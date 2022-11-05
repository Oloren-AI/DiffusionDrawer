
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import GATv2Conv

import numpy as np
import pytorch_lightning as pl

class Diffuser:
    
	def __init__(self, T: int, *args, **kwargs):
		self.T = T
		self.betas = torch.tensor([self.beta_t(t) for t in range(T)])
		self.alphas = 1 - self.betas
		self.alpha_bars = torch.cumprod(self.alphas, dim = 0)
	
	@abstractmethod
	def beta_t(self, t: int):
		# returns beta_t, variance at time t
		pass

	def sample(self, x: torch.Tensor):
		t = np.random.randint(0, self.T)
		return self._sample(x, t)

	def _sample(self, x: torch.Tensor, t: int):
		alpha_bar = self.alpha_bars[t].float().to(x.device)
		epsilon = torch.normal(0, 1, size=x.shape).float().to(x.device)
		x_ = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * epsilon
		return epsilon, x_

class LinearDiffuser(Diffuser):
    def __init__(self,*args, beta_lower: float = 10e-4, beta_upper: float = 0.02, **kwargs):
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        super().__init__(*args, **kwargs)
    
    def beta_t(self, t: int):
        return (t/ self.T) * (self.beta_upper - self.beta_lower) + self.beta_lower
		

class GATv2Model(pl.LightningModule):
	def __init__(self, feature_channels: int = 9,
            	diffuser: Diffuser = LinearDiffuser(T=1000),
            	hidden_channels: int = 8,
            	heads: int = 1):
    
		super().__init__()
		self.save_hyperparameters()
		self.diffuser = diffuser

		self.hidden_channels = hidden_channels
		self.heads = heads
		self.conv1 = GATv2Conv(feature_channels + 2, self.hidden_channels, edge_dim = 3, heads = self.heads)
		self.conv2 = GATv2Conv(self.hidden_channels * self.heads, 2, edge_dim = 3, heads = 1)
  
		self.loss = nn.MSELoss()
        
	def forward(self, data):
		x, edge_index, edge_attr, = data.x, data.edge_index, data.edge_attr
		x = x.float()
		edge_attr = edge_attr.float()

		x = self.conv1(x, edge_index, edge_attr)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attr)
		return x
    
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, batch, batch_idx):
		x, edge_index, edge_attr, = batch.x, batch.edge_index, batch.edge_attr
		x = x.float()
		edge_attr = edge_attr.float()

		theta = (torch.rand(1)*np.pi).float().to(x.device)
		rotation_matrix = torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]]).float().to(x.device)
		positions = torch.matmul(x[:, -2:], rotation_matrix)
  
  		# apply the forward process to x
		epsilon, x_ = self.diffuser.sample(positions)

		x[:, -2:] = x_

		x = self.conv1(x, edge_index, edge_attr)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attr)

		loss = F.mse_loss(x, epsilon)
  
		self.log('train_loss', loss)
  
		return loss

	def validation_step(self, batch, batch_idx):
		x, edge_index, edge_attr, = batch.x, batch.edge_index, batch.edge_attr
		x = x.float()
		edge_attr = edge_attr.float()
  
  		# apply the forward process to x
		x_, epsilon = self.diffuser.sample(x[:, -2:])
		x[:, -2:] = x_

		x = self.conv1(x, edge_index, edge_attr)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attr)

		loss = F.mse_loss(x, epsilon)
  
		self.log('val_loss', loss)