
import os 

import numpy as np
import pandas as pd
import scipy.sparse as sp 

from skmultilearn.model_selection import IterativeStratification

from rdkit import Chem

def sdf_to_smiles(filename, smiles_name="SMILES"):
    assert filename.endswith(".sdf")
    print ("reading molecules from sfd file:", filename)
    mol_df = Chem.PandasTools.LoadSDF(filename, 
        smilesName=smiles_name, molColName=None)
    print ("removing molecules with missing SMILES")
    mol_df = mol_df[~mol_df[smiles_name].isnull()]
    print ("identified", mol_df.shape[0], "molecules")
    return mol_df
    

def valid_smiles(smi):
    smi = smi["SMILES"]
    assert smi is not None
    try:
        return Chem.MolFromSmiles(smi) is not None
    except TypeError:
        print ("typeerror", smi)
        assert False, smi
        return False

def write_smiles(smiles, smiles_filename):
    print ("writing smiles to", smiles_filename)
    with open(smiles_filename, "w") as f:
        for compound, smi in smiles.items():
            f.write("{}\t{}\n".format(smi, compound))
            
def read_smiles(smiles_filename, remove_invalid=True):
    print ("reading compounds from", 
        smiles_filename)

    smiles = pd.read_csv(smiles_filename, header=None, 
        sep="\t", index_col=1)#[0]#.astype("string")
    smiles.columns = ["SMILES"] + list(smiles.columns)[1:]

    assert len(set(smiles.index)) == smiles.shape[0], \
        (len(set(smiles.index)), smiles.shape[0])

    if remove_invalid:
        print ("removing invalid SMILES")
        valid = smiles.apply(valid_smiles, axis=1)
        smiles = smiles[valid]

    print ("number of training SMILES:", 
        smiles.shape[0])

    return smiles["SMILES"]

def load_labels(labels_filename):
    print ("loading labels from", labels_filename)
    Y = sp.load_npz(labels_filename)
    print ("labels shape is", Y.shape)
    return Y # sparse format

def split_data(X, Y, n_splits=5, output_dir="splits"):
    split = IterativeStratification(n_splits=n_splits, order=1, random_state=0, )

    splits = []
    for split_no, (train_idx, test_idx) in enumerate(split.split(X, Y)):
        print ("processing fold", split_no+1, "/", n_splits)
       
        X_train = X[train_idx]
        X_test = X[test_idx]

        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        assert not Y_train.all(axis=-1).any()
        assert not (1-Y_train).all(axis=-1).any()

        assert not Y_test.all(axis=-1).any()    
        assert not (1-Y_test).all(axis=-1).any()

        split_dir = os.path.join(output_dir,
            "split_{}".format(split_no))
        os.makedirs(split_dir, exist_ok=True)

        write_smiles(X_train, os.path.join(split_dir, "train.smi"))
        write_smiles(X_test, os.path.join(split_dir, "test.smi"))

        sp.save_npz(os.path.join(split_dir, "train.npz"), 
            sp.csr_matrix(Y_train))
        sp.save_npz(os.path.join(split_dir, "test.npz"), 
            sp.csr_matrix(Y_test))


def filter_data(X, Y, 
    min_actives=10, 
    min_hits=10):
    print ("filtering data to ensure",
        "at least", min_actives, "compounds hit each target and",
        "each compound hits", min_hits, "targets")

    n_compounds = Y.shape[0]
    n_targets = Y.shape[1]

    print ("num compounds before filtering:", n_compounds)
    print ("num targets before filtering:", n_targets)
    print ("number of relationships: before filtering", Y.sum())

    # valid_compounds = np.ones(n_compounds, dtype=bool)
    # valid_targets = np.ones(n_targets, dtype=bool)

    while (Y.sum(axis=0) < min_actives).any() or\
        (Y.sum(axis=1) < min_hits).any():

        # filter targets
        idx = np.logical_and(
            Y.sum(axis=0) >= min_actives, 
            (1-Y).sum(axis=0) >= min_actives)
        Y = Y[:,idx]

        # filter compounds
        idx = Y.sum(axis=1) >= min_hits
        X = X[idx]
        Y = Y[idx, ]

    assert (Y.sum(axis=0) >= min_actives).all()
    assert ((1-Y).sum(axis=0) >= min_actives).all()
    assert (Y.sum(axis=1) >= min_hits).all()
    assert ((1-Y).sum(axis=1) >= min_hits).all()

    print ("num compounds after filtering:", X.shape[0])
    print ("num targets after filtering:", Y.shape[1])
    print ("number of relationships after filtering:", Y.sum())

    return X, Y

def main(): # run this to generate training splits

    # n_splits = 5

    # training_smiles_filename = os.path.join("data", 
    #     "compounds.smi")
    # assert os.path.exists(training_smiles_filename)
    # X = read_smiles(training_smiles_filename)

    # labels_filename = os.path.join("data", "targets.npz")
    # assert os.path.exists(labels_filename)
    # Y = load_labels(labels_filename).A

    # # filter out compounds that hit at less than min_hits targets
    # min_hits = 1

    # # remove any targets that are hit/not hit by less than min_actives compounds
    # min_actives = n_splits # at least one positive example of the class in each split

    # X, Y = filter_data(X, Y, 
    #     min_actives=min_actives, 
    #     min_hits=min_hits)

    # assert X.shape[0] == Y.shape[0]
    # assert Y.any(axis=1).all()
    # assert (1-Y).any(axis=1).all()
    # assert not Y.all(axis=0).any()
    # assert not (1-Y).all(axis=0).any()
    # assert (Y.sum(axis=0) >= min_actives).all()
    # assert (Y.sum(axis=1) >= min_hits).all()

    # output_dir = os.path.join("splits")
    # split_data(X, Y, 
    #     n_splits=n_splits, 
    #     output_dir=output_dir)

    # process test data
    test_smiles_filename = os.path.join("splits",
        "complete", 
        "test_original.smi")
    assert os.path.exists(test_smiles_filename)
    X = read_smiles(test_smiles_filename)

    test_labels_filename = os.path.join("splits",
        "complete", 
        "test_original.npz")
    assert os.path.exists(test_labels_filename)
    Y = load_labels(test_labels_filename).A

    # # filter out compounds that hit at less than min_hits targets
    min_hits = 1

    # keep targets with no actives
    min_actives = 0 

    X, Y = filter_data(X, Y, 
        min_actives=min_actives, 
        min_hits=min_hits)

    assert X.shape[0] == Y.shape[0]
    assert Y.any(axis=1).all()
    assert (1-Y).any(axis=1).all()

    assert (Y.sum(axis=0) >= min_actives).all()
    assert (Y.sum(axis=1) >= min_hits).all()

    output_dir = os.path.join("splits", 
        "complete")
    test_smiles_filename = os.path.join(output_dir,
        "test.smi")
    write_smiles(X, test_smiles_filename)

    test_labels_filename = os.path.join(output_dir,
        "test.npz")
    sp.save_npz(test_labels_filename, 
        sp.csr_matrix(Y))

if __name__ == "__main__":
    main()