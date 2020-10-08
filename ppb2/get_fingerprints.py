import os

import numpy as np
import pandas as pd 

from scipy import sparse as sp

import swifter

import pickle as pkl

# from rdkit import Chem
# from rdkit.Chem import AllChem

# from openeye import oechem
# from openeye import oegraphsim

import multiprocessing as mp

import functools

# RDK

def get_chems(smiles):
    from rdkit import Chem

    print ("building RDK chems from SMILES")
    return smiles.swifter.apply(
        lambda smi: Chem.MolFromSmiles(smi))

def rdk_wrapper(x, n_bits):
    from rdkit import Chem
    return [bool(y) 
            for y in Chem.RDKFingerprint( x , fpSize=n_bits )]

def get_rdk(X, n_bits=1024, parallel=True):
    '''X is a vector of SMILES'''

    chems = get_chems(X)

    print ("computing RDKit fingerprint")

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            rdk_fingerprints = p.map(
                functools.partial(rdk_wrapper,
                    n_bits=n_bits),
                chems
            )

    else:
        rdk_fingerprints = list(chems.swifter.apply(lambda x:
            rdk_wrapper(x, n_bits=n_bits)
                if x else None))

    return np.array(rdk_fingerprints)

def morg_wrapper(chem, 
        radius, 
        n_bits):
    from rdkit.Chem import AllChem
    return [bool(bit) 
        for bit in AllChem.GetMorganFingerprintAsBitVect(chem, 
            radius=radius, 
            nBits=n_bits,
            useFeatures=True , )]

def get_morg(X, radius=2, n_bits=1024, parallel=True):
    '''X is a vector of SMILES'''

    chems = get_chems(X) 

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
                chems    
            )

    else:
        morgan_fingerprints = list(chems.swifter.apply(lambda x:
            morg_wrapper( x, 
                    radius=radius, 
                    useFeatures=True ,
                    nBits=n_bits)
                if x else None
            ))

    return np.array(morgan_fingerprints)

#######################

# OPENEYE

def get_bit_string(fp):
    s = [fp.IsBitOn(b) for b in range(fp.GetSize())]
    assert any(s)
    return s

def mol_wrapper(smi):
    from openeye import oechem
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    return mol

def get_mols(smiles, parallel=True):
    assert isinstance(smiles, pd.Series)

    print("building list of OpenEye molecules")

    if parallel:

        with mp.Pool(processes=mp.cpu_count()) as p:
            mols = p.map(mol_wrapper, 
                    smiles)

    else:
        mols = [mol_wrapper(smi) for smi in smiles]

    return mols

def circular_wrapper(mol, 
    num_bits = 1024,
    min_radius = 2,
    max_radius = 2):
    from openeye import oegraphsim
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeCircularFP(fp, mol, 
            num_bits, 
            min_radius, max_radius, 
            oegraphsim.OEFPAtomType_DefaultPathAtom, 
            oegraphsim.OEFPBondType_DefaultPathBond)
    return get_bit_string(fp)

def get_circular(X, 
    num_bits = 1024,
    min_radius = 2,
    max_radius = 2,
    parallel=True):
   
    mols = get_mols(X, parallel=parallel)

    print ("computing circular fingerprint")

    if parallel:

        with mp.Pool(processes=mp.cpu_count()) as p:
            fps = p.map(functools.partial(
                circular_wrapper, num_bits=num_bits, min_radius=min_radius,
                max_radius=max_radius),
                mols)

    else:

        fps = [
            circular_wrapper(mol, 
                num_bits=num_bits, min_radius=min_radius,
                max_radius=max_radius)
            for mol in mols]

    return np.array(fps)

def maccs_wrapper(mol, ):
    from openeye import oegraphsim
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeMACCS166FP(fp, mol,)
    return get_bit_string(fp)


def get_MACCs(X, parallel=True):

    print ("computing maccs fingerprint")

    from openeye import oegraphsim

    mols = get_mols(X, parallel=parallel)

    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as p:
            fps = p.map(maccs_wrapper, mols)

    else:
        fps = [maccs_wrapper(mol) for mol in mols]

    return np.array(fps)

def compute_fp(X, fp, n_bits=1024):
    print ("Computing", fp, "fingerpints for", X.shape[0],
        "SMILES")
    if fp == "morg2":
        X = get_morg(X, n_bits=n_bits)
    elif fp == "morg3":
        X = get_morg(X, radius=3, n_bits=n_bits)
    elif fp == "rdk":
        X = get_rdk(X, n_bits=n_bits)
    elif fp == "maccs":
        X = get_MACCs(X)
    elif fp == "circular":
        X = get_circular(X)
    else:
        raise NotImplementedError

    print ("Computed fingerprint")
    print ("Shape is", X.shape)
    
    return X

def load_training_fingerprints(X, fp):
    fp_filename = os.path.join("data",
        "fingerprints", "{}.pkl".format(fp))
    print ("loading fingerprints from", 
        fp_filename)
    # return np.load(fp_filename)
    with open(fp_filename, "rb") as f:
        fingerprints = pkl.load(f)
    return np.array([fingerprints[x] 
        for x in X])

def load_labels():
    labels_filename = os.path.join("data",
        "targets.npz")
    print ("loading labels from", labels_filename)
    Y = sp.load_npz(labels_filename)
    print ("labels shape is", Y.shape)
    return Y # sparse format

def main():

    fps = ("morg2", "morg3", 
        "rdk", "maccs", "circular", )

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
        "fingerprints_test")
    os.makedirs(fingerprint_directory, exist_ok=True)
    
    for fp in fps:
        
        fingerprints_filename = os.path.join(fingerprint_directory,
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
        

if __name__ == "__main__":
    main()