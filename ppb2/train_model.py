import os 
import argparse 

import numpy as np
import pandas as pd

import pickle as pkl 

from data_utils import read_smiles, load_labels
from models import build_model, save_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--compounds")
    parser.add_argument("--targets")
    parser.add_argument("--path")
    parser.add_argument("--n_proc", default=4, type=int)

    parser.add_argument("--model", default=["morg2-nn+nb"], nargs="+")

    return parser.parse_args()

def main():

    args = parse_args()

    X = read_smiles(args.compounds)
    Y = load_labels(args.targets)
    if not isinstance(Y, np.ndarray):
        Y = Y.A

    model = build_model(args)

    model_dir = os.path.join(args.path, )
    os.makedirs(model_dir, exist_ok=True)

    if args.model[0] == "stack":
        model_filename = os.path.join(model_dir, 
            "stack-({}).pkl".format("&".join((name
                    for name, _ in model.classifiers))))
    else:
        model_filename = os.path.join(model_dir, 
            "{}.pkl".format(args.model[0]))

    if not os.path.exists(model_filename):

        print ("fitting PPB2 model")
        model.fit(X, Y)

        save_model(model, model_filename)

if __name__ == "__main__":
    main()

