import os

import numpy as np
import pandas as pd 

from scipy import sparse as sp

import swifter

import pickle as pkl

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from openeye import oechem
from openeye import oegraphsim

import multiprocessing as mp

import functools

# RDK

# def get_rdk_mols(smiles):
#     # from rdkit import Chem

#     print ("building RDK chems from SMILES")
#     return smiles.swifter.apply(
#         lambda smi: Chem.MolFromSmiles(smi))


def rdk_maccs_wrapper(smi, ):
    mol = Chem.MolFromSmiles(smi)
    assert mol.GetNumAtoms() > 0
    return [bool(bit) 
        for bit in MACCSkeys.GenMACCSKeys(mol)]
    
def get_rdk_maccs(smiles, parallel=True):
    '''input is a vector of SMILES'''

    # mols = get_rdk_mols(X)

    print ("computing RDKit MACCS fingerprint")

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            rdk_maccs_fingerprints = p.map(
                rdk_maccs_wrapper,
                smiles
            )

    else:
        rdk_maccs_fingerprints = smiles.swifter.apply(
            rdk_maccs_wrapper)
        rdk_maccs_fingerprints = list(rdk_maccs_fingerprints.loc[smiles.index])

    return np.array(rdk_maccs_fingerprints)


def rdk_wrapper(smi, n_bits=1024):
    # from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    assert mol.GetNumAtoms() > 0
    return [bool(bit) 
        for bit in Chem.RDKFingerprint( mol , fpSize=n_bits )]

def get_rdk(smiles, n_bits=1024, parallel=True):
    '''input is a vector of SMILES'''

    # mols = get_rdk_mols(X)

    print ("computing RDKit topological fingerprint")

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            rdk_fingerprints = p.map(
                functools.partial(rdk_wrapper,
                    n_bits=n_bits),
                smiles
            )

    else:
        rdk_fingerprints = smiles.swifter.apply(lambda smi:
            rdk_wrapper(smi, n_bits=n_bits)
                )
        rdk_fingerprints = list(rdk_fingerprints.loc[smiles.index])

    return np.array(rdk_fingerprints)

def morg_wrapper(smi, 
    radius, 
    n_bits):
    # from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    assert mol.GetNumAtoms() > 0
    return [bool(bit) 
        for bit in AllChem.GetMorganFingerprintAsBitVect(mol, 
            radius=radius, 
            nBits=n_bits,
            useFeatures=True , )]

def get_morg(smiles, radius=2, n_bits=1024, parallel=True):
    '''input is a vector of SMILES'''

    # mols = get_rdk_mols(X) 

    print ("computing MORG fingerprint")
    print ("nbits:", n_bits)
    print ("radius:", radius)

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            morgan_fingerprints = p.map(
                functools.partial(morg_wrapper, 
                    radius=radius, 
                    n_bits=n_bits,
                    ),
                smiles    
            )

    else:
        morgan_fingerprints = smiles.swifter.apply(lambda smi:
            morg_wrapper( smi, 
                    radius=radius, 
                    n_bits=n_bits)
            )
        morgan_fingerprints = list(morgan_fingerprints.loc[smiles.index])

    return np.array(morgan_fingerprints)

#######################

# OPENEYE

def get_bit_string(fp):
    s = [fp.IsBitOn(b) for b in range(fp.GetSize())]
    assert any(s)
    return s

# def mol_wrapper(smi):
#     # from openeye import oechem
#     mol = oechem.OEGraphMol()
#     oechem.OESmilesToMol(mol, smi)
#     return mol

# def get_mols(smiles, parallel=True):
#     assert isinstance(smiles, pd.Series)

#     print("building list of OpenEye molecules")

#     if parallel:

#         with mp.Pool(processes=mp.cpu_count()) as p:
#             mols = p.map(mol_wrapper, 
#                     smiles) # order maintaained

#     else:
#         mols = [mol_wrapper(smi) for smi in smiles]

#     return mols

def circular_wrapper(smi, 
    num_bits = 1024,
    min_radius = 2,
    max_radius = 2):
    # from openeye import oegraphsim
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeCircularFP(fp, mol, 
        num_bits, 
        min_radius, max_radius, 
        oegraphsim.OEFPAtomType_DefaultPathAtom, 
        oegraphsim.OEFPBondType_DefaultPathBond)
    return get_bit_string(fp)

def get_circular(smiles, 
    num_bits = 1024,
    min_radius = 2,
    max_radius = 2,
    parallel=True):
    '''
    input is smiles
    '''
   
    # mols = get_mols(X, parallel=parallel)

    print ("computing circular fingerprint")

    if parallel:

        with mp.Pool(processes=mp.cpu_count()) as p:
            fps = p.map(functools.partial(
                circular_wrapper, num_bits=num_bits, min_radius=min_radius,
                max_radius=max_radius),
                smiles)

    else:

        fps = [
            circular_wrapper(smi, 
                num_bits=num_bits, min_radius=min_radius,
                max_radius=max_radius)
            for smi in smiles]

    return np.array(fps)

def maccs_wrapper(smi, ):
    # from openeye import oegraphsim
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeMACCS166FP(fp, mol,)
    return get_bit_string(fp)

def get_MACCs(smiles, parallel=True):

    print ("computing maccs fingerprint")

    # from openeye import oegraphsim

    # mols = get_mols(X, parallel=parallel)

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            fps = p.map(maccs_wrapper, smiles)

    else:
        fps = [maccs_wrapper(smi) for smi in smiles]

    return np.array(fps)

def compute_fp(smiles, fp, n_bits=1024):
    print ("computing", fp, "fingerpints for", 
        smiles.shape[0], "SMILES")
    if fp == "morg2":
        fps = get_morg(smiles, n_bits=n_bits)
    elif fp == "morg3":
        fps = get_morg(smiles, radius=3, n_bits=n_bits)
    elif fp == "rdk":
        fps = get_rdk(smiles, n_bits=n_bits)
    elif fp == "rdk_maccs":
        fps = get_rdk_maccs(smiles)
    elif fp == "maccs":
        fps = get_MACCs(smiles)
    elif fp == "circular":
        fps = get_circular(smiles)
    else:
        raise NotImplementedError

    print ("computed", fp, "fingerprint")
    
    return fps

def read_smiles(smiles_filename):
    print ("reading training compounds from", 
        smiles_filename)

    smiles = pd.read_csv(smiles_filename, header=None, 
        sep="\t", index_col=1)[0].astype("string")
    print ("number of training SMILES:", 
        smiles.shape[0])
    return smiles

def load_training_fingerprints(smiles, fp):
    fp_filename = os.path.join("data",
        "fingerprints", "{}.pkl".format(fp))
    print ("loading fingerprints from", 
        fp_filename)
    # return np.load(fp_filename)
    with open(fp_filename, "rb") as f:
        fingerprints = pkl.load(f)
    return np.array([fingerprints[smi] 
        for smi in smiles])

def load_labels():
    labels_filename = os.path.join("data",
        "targets.npz")
    print ("loading labels from", labels_filename)
    Y = sp.load_npz(labels_filename)
    print ("labels shape is", Y.shape)
    return Y # sparse format

def main():

    fps = ("morg2", "morg3", 
        "rdk", "rdk_maccs",
        "maccs", "circular", )

    training_smiles_filename = os.path.join("data", 
        "compounds.smi")
    print ("reading training smiles from", 
        training_smiles_filename)
    smiles = pd.read_csv(training_smiles_filename,header=None, 
        sep="\t", index_col=1)[0].astype("string")

    num_smiles = smiles.shape[0]
    print ("determining fingerprints for", 
        num_smiles, "compounds")

    fingerprint_directory = os.path.join("data", 
        "fingerprints")
    os.makedirs(fingerprint_directory, exist_ok=True)
    
    for fp in fps:
        
        fingerprints_filename = os.path.join(
            fingerprint_directory,
            "{}.pkl".format(fp))

        if os.path.exists(fingerprints_filename):
            continue

        fingerprints = compute_fp(smiles, fp)
        assert fingerprints.shape[0] == num_smiles
        fingerprints_dict = {
            smile: fp for smile, fp
                in zip(smiles, fingerprints)
        }

        print ("writing to", fingerprints_filename)
        # np.save(fingerprints_filename, fingerprints)
        with open(fingerprints_filename, "wb") as f:
            pkl.dump(fingerprints_dict, f, pkl.HIGHEST_PROTOCOL)

        print ("completed fingerprint", fp)
        print ()
        

if __name__ == "__main__":
    main()