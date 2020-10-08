import os 
import argparse 

import numpy as np
import pandas as pd

import pickle as pkl 

from get_fingerprints import load_labels
from models import PPB2, StackedPPB2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_id", type=int)

    parser.add_argument("--fp", default="morg2",
        choices=["mqn", "xfp", "ecfp4", 
        "morg2", "morg3", "rdk",
        "circular", "maccs"])

    parser.add_argument("--model", default="nb",
        choices=["nn", "nb", "nn+nb", 
        "lr", "svc", "bag", "stack"])

    return parser.parse_args()

def main():

    args = parse_args()

    print ("Fingerprint type:", args.fp)
    print ("Model type:", args.model)

    data_dir = os.path.join("data", )
    
    training_data_filename = os.path.join(data_dir,
        "compounds.smi") # chembl 22
    print ("reading training compounds from", 
        training_data_filename)

    X = pd.read_csv(training_data_filename, header=None, 
        sep="\t", index_col=1)[0].astype("string")
    print ("number of training SMILES:", 
        X.shape[0])
    Y = load_labels()

    # from sklearn.model_selection import train_test_split

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100)

    n_targets = Y.shape[1]

    target_id = args.target_id
    assert target_id < n_targets

    # map id to target name
    # with open("data/id_to_gene_symbol.pkl", "rb") as f:
        # id_to_targets = pkl.load(f)

    # for target_id in range(n_targets):

    # target_name = id_to_targets[target_id]

    print ("training model for target_id", 
        target_id,  )
    labels = Y[:, target_id].A.flatten()

    if args.model == "stack":
        model = StackedPPB2()
    else:
        model = PPB2(fp=args.fp, 
            model_name=args.model, )

    model_dir = os.path.join("models", )
    os.makedirs(model_dir, exist_ok=True)

    model_filename = os.path.join(model_dir, 
        "target-{}-{}-{}.pkl".format(target_id, 
                args.fp, args.model))

    if not os.path.exists(model_filename):

        print ("fitting PPB2 model")
        model.fit(X, labels)

        print ("pickling trained model to", 
            model_filename)
        with open(model_filename, "wb") as f:
            pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

