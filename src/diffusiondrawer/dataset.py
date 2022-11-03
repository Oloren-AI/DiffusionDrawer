import logging
import torch
import random
import pickle

logging.basicConfig(filename="diffusiondrawer.log")
logger = logging.getLogger(__name__)

import os
import numpy as np

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

from torch_geometric.data import Data
from torch_geometric.data import DataLoader


def mol_to_data(mol):
    
    # atoms
    atom_features_list = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos = mol.GetConformer().GetPositions()[i]
        atom_features_list.append(np.concatenate([atom_to_feature_vector(atom), [pos[0]/12, pos[1]/8]]))
    x = np.array(atom_features_list)
    
    # bonds
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.int64)

    data = Data()
        
    data.num_nodes = mol.GetNumAtoms()
    data.edge_index = torch.tensor(edge_index)
    data.edge_attr = torch.tensor(edge_attr)
    data.x = torch.tensor(x)
    
    positions = mol.GetConformer().GetPositions()
    
    data.y = torch.tensor([[pos[0]/12, pos[1]/8] for pos in positions])
    return data

def load_directory(directory):
    """
    loads directory of mol files
    
    Returns:
        List[Data]: list of Data objects
    """
    dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), directory)
    data_list = []
    for filename in os.listdir(dir):
        try:
            if filename.endswith(".mol") or filename.endswith(".MOL") or filename.endswith(".Mol"):
                mol = Chem.MolFromMolFile(os.path.join(dir, filename))
                data = mol_to_data(mol)
                data_list.append(data)
        except:
            logger.info("error loading file: " + filename)
    return data_list

def load_dataset():
    """
    loads dataset of mol file directories
    
    Returns:
        List[Data]: list of Data objects
    """
    
    fname = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), "dataset.pkl")
    
    if os.path.exists(fname):
        return pickle.load(open(fname, "rb"))
    else:
        dirs = ["CLEF_mol_ref", "JPO_mol_ref", "UOB_mol_ref", "USPTO_mol_ref"]
    
        out = list()
        for dir in dirs:
            out += load_directory(dir)
            
        pickle.dump(out, open(fname, "wb"))
        
        return out

def get_dataloaders(batch_size=4, shuffle=False, num_workers=8, n = None):
    data_list = load_dataset()
    
    # shuffle data_list
    random.shuffle(data_list)
    if n is not None:
        data_list = data_list[:n]
    
    # random split data_list into train, val, test
    train = data_list[:int(len(data_list)*0.8)]
    val = data_list[int(len(data_list)*0.8):int(len(data_list)*0.9)]
    test = data_list[int(len(data_list)*0.9):]
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, test_loader
    
    