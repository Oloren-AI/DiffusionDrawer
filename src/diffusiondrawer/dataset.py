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

from tqdm import tqdm

def mol_to_data(mol):
    
    # atoms
    atom_features_list = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        num_conf = mol.GetNumConformers()
        if num_conf > 0:
            pos = mol.GetConformer(num_conf-1).GetPositions()[i]
            pos_vec = [pos[0], pos[1]]
        else:
            pos_vec = [0, 0]
        atom_vec = np.zeros(119)
        atom_vec[atom.GetAtomicNum()] = 1
        atom_features_list.append(np.concatenate([atom_vec, pos_vec]))
    x = np.array(atom_features_list)
    max_pos = np.max(x[:, -2:])
    min_pos = np.min(x[:, -2:])
    x[:, -2:] = (x[:, -2:] - min_pos) / (max_pos - min_pos)
    x[:, -2:] = x[:, -2:]
    
    # bonds
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()]

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

def load_sdf(sdf_file_name,force=False):
    """
    loads sdf file
    
    Returns:
        List[Data]: list of Data objects
    """
    
    fname = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), f"{sdf_file_name[:-4]}.pkl")
    if os.path.exists(fname) and not force:
        return pickle.load(open(fname, "rb"))
    else:
        sdf_file_name = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), sdf_file_name)
        data_list = []
        suppl = Chem.SDMolSupplier(sdf_file_name)
        for mol in tqdm(suppl):
            if not mol is None:
                data = mol_to_data(mol)
                data_list.append(data)
            
        pickle.dump(data_list, open(fname, "wb"))
        
        return data_list

def load_directory(directory,force=False):
    """
    loads directory of mol files
    
    Returns:
        List[Data]: list of Data objects
    """
    fname = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), f"{directory}.pkl")
    if os.path.exists(fname) and not force:
        return pickle.load(open(fname, "rb"))
    else:
        dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/"), directory)
        data_list = []
        for filename in os.listdir(dir):
            try:
                if filename.endswith(".mol") or filename.endswith(".MOL") or filename.endswith(".Mol"):
                    mol = Chem.MolFromMolFile(os.path.join(dir, filename))
                    data = mol_to_data(mol)
                    data_list.append(data)
                elif filename.endswith(".sdf") or filename.endswith(".SDF") or filename.endswith(".Sdf"):
                    suppl = Chem.SDMolSupplier(os.path.join(dir, filename))
                    for mol in suppl:
                        data = mol_to_data(mol)
                        data_list.append(data)
            except:
                logger.info("error loading file: " + filename)
        pickle.dump(data_list, open(fname, "wb"))
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
def example1():
    mol = Chem.MolFromMolFile("data/USPTO_mol_ref/US07314511-20080101-C00002.MOL")

def get_dataloaders(batch_size=4, shuffle=False, num_workers=8, n = None, small = False, force=False):
    if small:
        data_list = load_directory("USPTO_mol_ref",force=force)
    else:
        data_list = []
    
        for dir in ["CLEF_mol_ref", "JPO_mol_ref", "UOB_mol_ref", "USPTO_mol_ref"]:
            data_list += load_directory(dir,force=force)
    
        for sdf in ["PubChem_CP.sdf"]:
            data_list += load_sdf(sdf,force=force)
    
        # shuffle data_list
        random.seed(42)
        random.shuffle(data_list)
        if n is not None:
            data_list = data_list[:n]
        
    print("Number of Molecules: " + str(len(data_list)))
    
    # random split data_list into train, val, test
    train = data_list[:int(len(data_list)*0.8)]
    val = data_list[int(len(data_list)*0.8):int(len(data_list)*0.9)]
    test = data_list[int(len(data_list)*0.9):]
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, test_loader
    
    