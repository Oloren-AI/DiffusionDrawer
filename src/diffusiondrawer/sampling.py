from rdkit import Chem
import torch
from .model import *
from .dataset import *

def set_conformer(mol, positions):
    mol = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        position = positions[i]
        conformer.SetAtomPosition(i, (position[0].item(), position[1].item(), 0))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    return mol

def get_positions(mol):
    conformer = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(i)
        positions.append([position[0], position[1]])
    positions = torch.tensor(positions)
    return positions

def sample_mol(mol, t, diffuser = LinearDiffuser(1000)):
    epsilon, pos =  diffuser._sample(get_positions(mol), t)
    mol = Chem.Mol(mol, True)
    mol.RemoveAllConformers()
    mol = set_conformer(mol, pos)
    return epsilon, mol

def run_diffusion(model, mol, T):
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (np.random.uniform(-1,1),np.random.uniform(-1,1), 0))
    mol.AddConformer(conformer)
    
    diffuser = LinearDiffuser(T=T)
    data = mol_to_data(mol)
    
    mols = []
    for i in tqdm(range(T-1,0,-1)):
        prev_pos = data.x[:, -2:]

        epsilon = model(data)
        
        alpha = diffuser.alphas[i]
        alpha_bar = diffuser.alpha_bars[i]
        beta = diffuser.betas[i]
        
        coef = (1-alpha)/torch.sqrt(1-alpha_bar)
        reconstruct = (prev_pos - coef*epsilon)/torch.sqrt(alpha)
        
        if i == 0:
            z = 0
        else:
            z = torch.rand(prev_pos.shape)
        
        data.x[:, -2:] = reconstruct + torch.sqrt(beta)*z
        mols.append(set_conformer(mol, data.x[:, -2:]))
            
    return mols