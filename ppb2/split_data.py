
import os 

import pandas as pd
import scipy.sparse as sp 

from skmultilearn.model_selection import IterativeStratification

def write_smiles(smiles, smiles_filename):
    with open(smiles_filename, "w") as f:
        for compound, smi in smiles.items():
            f.write("{}\t{}\n".format(smi, compound))
            
def read_smiles(smiles_filename):
    print ("reading compounds from", 
        smiles_filename)

    smiles = pd.read_csv(smiles_filename, header=None, 
        sep="\t", index_col=1)[0].astype("string")
    print ("number of training SMILES:", 
        smiles.shape[0])
    return smiles

def load_labels(labels_filename):
    # labels_filename = os.path.join("data",
        # "targets.npz")
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


def filter_training_data(X, Y, 
    min_compounds=10, 
    min_target_hits=10):
    print ("filtering training data to ensure",
        "at least", min_compounds, "compounds hit each target and",
        "each compound hits", min_target_hits, "targets")

    while (Y.sum(axis=0) < min_compounds).any() or\
        (Y.sum(axis=1) < min_target_hits).any():

        # filter targets
        idx = np.logical_and(
            Y.sum(axis=0) >= min_compounds, 
            (1-Y).sum(axis=0) >= min_compounds)
        Y = Y[:,idx]

        # filter compounds
        idx = Y.sum(axis=1) >= min_target_hits
        X = X[idx]
        Y = y[idx, ]

    assert (Y.sum(axis=0) >= min_compounds).all()
    assert ((1-Y).sum(axis=0) >= min_compounds).all()
    assert (Y.sum(axis=1) >= min_target_hits).all()
    assert ((1-Y).sum(axis=1) >= min_target_hits).all()

    print ("num compounds after filtering:", X.shape[0])
    print ("num targets after filtering:", Y.shape[1])
    print ("number of relationships:", Y.sum())

    return X, Y

def main():

    training_smiles_filename = os.path.join("data", 
        "compounds.smi")
    assert os.path.exists(training_smiles_filename)
    X = read_smiles(training_smiles_filename)

    labels_filename = os.path.join("data", "targets.npz")
    assert os.path.exists(labels_filename)
    Y = load_labels(labels_filename).A

    # filter out compounds that hit at less than min_hits targets
    min_hits = 1

    # remove any targets that are hit/not hit by less than min_compounds compounds
    min_compounds = 5

    X, Y = filter_training_data(X, Y, 
        min_compounds=min_compounds, 
        min_target_hits=min_hits)

    assert X.shape[0] == Y.shape[0]
    assert Y.any(axis=1).all()
    assert (1-Y).any(axis=1).all()
    assert not Y.all(axis=0).any()
    assert not (1-Y).all(axis=0).any()
    assert (Y.sum(axis=0) >= min_compounds).all()
    assert (Y.sum(axis=1) >= min_hits).all()

    output_dir = os.path.join("splits")
    split_data(X, Y, n_splits=5, 
        output_dir=output_dir)

if __name__ == "__main__":
    main()