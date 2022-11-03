import pytest
import pickle

from diffusiondrawer.dataset import load_directory

__author__ = "davidzqhuang"
__copyright__ = "davidzqhuang"
__license__ = "MIT"

def test_load_directory():
    """Test load_directory"""
    load_directory("USPTO_mol_ref")
    
def test_training_step():
    """Test training_step"""
    from diffusiondrawer.model import GATv2Model
    model = GATv2Model(9)
    batch = pickle.load(open("tests/test_batch.pkl", "rb"))
    model.training_step(batch, 0)