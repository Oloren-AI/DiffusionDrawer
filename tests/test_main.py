import pytest
import pickle

import pytorch_lightning as pl

from diffusiondrawer.dataset import load_directory, get_dataloaders
from diffusiondrawer.model import GATv2Model

__author__ = "davidzqhuang"
__copyright__ = "davidzqhuang"
__license__ = "MIT"

def test_load_directory():
    """Test load_directory"""
    load_directory("USPTO_mol_ref")
    
def test_training_step():
    """Test training_step"""
    model = GATv2Model(9)
    loader = pickle.load(open("tests/test_loader.pkl", "rb"))
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, loader)