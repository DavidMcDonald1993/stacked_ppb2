import os 
import argparse 

import numpy as np
import pandas as pd

import pickle as pkl 

from get_fingerprints import read_smiles, load_labels
from models import PPB2, StackedPPB2

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--target_id", type=int)

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
    X = read_smiles(training_data_filename)
   
    Y = load_labels().A

    # idx = range(1000)

    # X = X[idx]
    # Y = Y[idx, ]

    # idx = np.logical_and(Y.any(0), (1-Y).any())
    # Y = Y[:,idx]

    if args.model == "stack":
        model = StackedPPB2()
    else:
        model = PPB2(fp=args.fp, 
            model_name=args.model, )

    model_dir = os.path.join("models", )
    os.makedirs(model_dir, exist_ok=True)

    if args.model == "stack":
        model_filename = os.path.join(model_dir, 
            "{}.pkl".format(args.model))
    else:
        model_filename = os.path.join(model_dir, 
            "{}-{}.pkl".format(args.fp, args.model))

    if not os.path.exists(model_filename):

        print ("fitting PPB2 model")
        model.fit(X, Y)

        print ("pickling trained model to", 
            model_filename)
        with open(model_filename, "wb") as f:
            pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

