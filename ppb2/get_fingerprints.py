import os

import gzip

import numpy as np
import pandas as pd 

from scipy import sparse as sp

import pickle as pkl

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from openeye import oechem
from openeye import oegraphsim

import multiprocessing as mp

import functools

from standardiser import standardise

from data_utils import read_smiles, load_labels

# RDK

def get_rdk_mol(smi, perform_standardisation=True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise Exception
    if mol is not None and perform_standardisation:
        try:
            mol = standardise.run(mol)
        except standardise.StandardiseException as e:
            pass
    return mol

def rdk_maccs_wrapper(smi, ):
    mol = get_rdk_mol(smi)
    if mol is None: 
        return mol
    assert mol.GetNumAtoms() > 0
    return sp.csr_matrix([bool(bit) 
        for bit in MACCSkeys.GenMACCSKeys(mol)])
    
def get_rdk_maccs(smiles, n_proc=8):
    '''input is a vector of SMILES'''

    print ("computing RDKit MACCS fingerprint")

    if n_proc>1:
        with mp.Pool(processes=n_proc) as p:
            rdk_maccs_fingerprints = p.map(
                rdk_maccs_wrapper,
                smiles
            )

    else:
        rdk_maccs_fingerprints = smiles.map(
            rdk_maccs_wrapper, )
        rdk_maccs_fingerprints = list(rdk_maccs_fingerprints.loc[smiles.index])

    rdk_maccs_fingerprints = filter(lambda fp: fp is not None,
        rdk_maccs_fingerprints)

    return sp.vstack(rdk_maccs_fingerprints)


def rdk_wrapper(smi, n_bits=1024):
    mol = get_rdk_mol(smi)
    if mol is None:
        return mol
    assert mol.GetNumAtoms() > 0
    return sp.csr_matrix([bool(bit) 
        for bit in Chem.RDKFingerprint( mol , fpSize=n_bits )])

def get_rdk(smiles, n_bits=1024, n_proc=8):
    '''input is a vector of SMILES'''

    print ("computing RDKit topological fingerprint")

    if n_proc>1:
        with mp.Pool(processes=n_proc) as p:
            rdk_fingerprints = p.map(
                functools.partial(rdk_wrapper,
                    n_bits=n_bits),
                smiles
            )

    else:
        rdk_fingerprints = smiles.map(lambda smi:
            rdk_wrapper(smi, n_bits=n_bits),
                )
        rdk_fingerprints = list(rdk_fingerprints.loc[smiles.index])

    rdk_fingerprints = filter(lambda fp: fp is not None,
        rdk_fingerprints)

    return sp.vstack(rdk_fingerprints)

def morg_wrapper(smi, 
    radius, 
    n_bits):
    mol = get_rdk_mol(smi)
    if mol is None:
        return mol
    assert mol.GetNumAtoms() > 0
    return sp.csr_matrix([bool(bit) 
        for bit in AllChem.GetMorganFingerprintAsBitVect(mol, 
            radius=radius, 
            nBits=n_bits,
            useFeatures=True , )])

def get_morg(smiles, radius=2, n_bits=1024, n_proc=8):
    '''input is a vector of SMILES'''

    print ("computing MORG fingerprint")
    print ("nbits:", n_bits)
    print ("radius:", radius)

    if n_proc>1:
        with mp.Pool(processes=n_proc) as p:
            morgan_fingerprints = p.map(
                functools.partial(morg_wrapper, 
                    radius=radius, 
                    n_bits=n_bits,
                    ),
                smiles    
            )

    else:
        morgan_fingerprints = smiles.map(lambda smi:
            morg_wrapper( smi, 
                    radius=radius, 
                    n_bits=n_bits),
            )
        morgan_fingerprints = list(morgan_fingerprints.loc[smiles.index])

    morgan_fingerprints = filter(lambda fp: fp is not None,
        morgan_fingerprints)

    return sp.vstack(morgan_fingerprints)

#######################

# OPENEYE

def get_bit_string(fp):
    s = [fp.IsBitOn(b) for b in range(fp.GetSize())]
    assert any(s)
    return sp.csr_matrix(s)


def circular_wrapper(smi, 
    num_bits = 1024,
    min_radius = 2,
    max_radius = 2):

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
    n_proc=8):
    '''
    input is smiles
    '''
   
    print ("computing circular fingerprint")

    if n_proc>1:

        with mp.Pool(processes=n_proc) as p:
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

    return sp.vstack(fps)

def maccs_wrapper(smi, ):

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeMACCS166FP(fp, mol,)
    return get_bit_string(fp)

def get_MACCs(smiles, n_proc=8):

    print ("computing maccs fingerprint")

    if n_proc>1:
        with mp.Pool(processes=n_proc) as p:
            fps = p.map(maccs_wrapper, smiles)

    else:
        fps = [maccs_wrapper(smi) for smi in smiles]

    return sp.vstack(fps)

def compute_fp(smiles, all_fp, n_bits=1024, n_proc=4):

    if all_fp == "all": # ensemble from concatenation of all fps
        all_fp = ("morg2", "morg3", 
        "rdk", "rdk_maccs",
        "maccs", "circular", )

    if not isinstance(all_fp, tuple):
        if not isinstance(all_fp, list):
            all_fp = [all_fp]   
        assert isinstance(all_fp, list)
        all_fp = tuple(all_fp)

    fps = []
    for fp in all_fp:
        print ("computing", fp, "fingerpints for", 
            smiles.shape[0], "SMILES",
            "using", n_proc, "cores")
        if fp == "morg2":
            fps.append(get_morg(smiles, n_bits=n_bits, n_proc=n_proc))
        elif fp == "morg3":
            fps.append(get_morg(smiles, radius=3, n_bits=n_bits, n_proc=n_proc))
        elif fp == "rdk":
            fps.append(get_rdk(smiles, n_bits=n_bits, n_proc=n_proc))
        elif fp == "rdk_maccs":
            fps.append(get_rdk_maccs(smiles, n_proc=n_proc))
        elif fp == "maccs":
            fps.append(get_MACCs(smiles, n_proc=n_proc))
        elif fp == "circular":
            fps.append(get_circular(smiles, n_proc=n_proc))
        else:
            raise NotImplementedError

        print ("computed", fp, "fingerprint")
        
    return sp.hstack(fps, format="csr")


def load_training_fingerprints(smiles, fp):
    fp_filename = os.path.join("data",
        "fingerprints", "{}.pkl.gz".format(fp))
    print ("loading fingerprints from", 
        fp_filename)
    if fp_filename.endswith(".pkl.gz"):
        f = gzip.open(fp_filename, "rb")
    else:
        f = open(fp_filename, "rb") 
    fingerprints = pkl.load(f)
    f.close()
    return sp.vstack([fingerprints[smi] 
        for smi in smiles])

def main():

    fps = ("morg2", "morg3", 
        "rdk", "rdk_maccs",
        "maccs", "circular", 
        "all")

    training_smiles_filename = os.path.join("data", 
        "compounds.smi.gz")
    smiles = read_smiles(training_smiles_filename)

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

        fingerprints = compute_fp(smiles, fp, n_proc=8)
        assert fingerprints.shape[0] == num_smiles
        fingerprints_dict = {
            smile: fp for smile, fp
                in zip(smiles, fingerprints)
        }

        print ("writing to", fingerprints_filename)
        with open(fingerprints_filename, "wb") as f:
            pkl.dump(fingerprints_dict, f, pkl.HIGHEST_PROTOCOL)

        print ("completed fingerprint", fp)
        print ()

if __name__ == "__main__":
    main()