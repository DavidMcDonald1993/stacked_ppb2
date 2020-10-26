import os
import argparse

import numpy as np
import pandas as pd 

import pickle as pkl 

from get_fingerprints import read_smiles

from pathlib import Path

from models import load_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query",)
    parser.add_argument("--model",)
    parser.add_argument("--output")

    parser.add_argument("--n_proc", default=8, type=int)

    parser.add_argument("-k", 
        default=200, 
        type=int)

    return parser.parse_args()

def main():

    args = parse_args()
    assert args.query is not None
    assert args.model is not None
    assert args.output is not None


    # load target name mappings
    target_mapping_filename = os.path.join("data",
        "id_to_uniprot.pkl")
    print ("reading target mapping from",
        target_mapping_filename)
    with open(target_mapping_filename, "rb") as f:
        target_mapping = pkl.load(f)


    model_filename = args.model 
    model = load_model(model_filename)
    model.set_n_proc(args.n_proc)

    # load queries (SMILES format)
    query_filename = args.query 
    print ("reading_queries from", query_filename)
    queries = read_smiles(query_filename)

    query_index = queries.index 
    n_queries = queries.shape[0]
    print ("number of queries:", n_queries)

    prediction = model.predict(queries)
    prediction_probs = model.predict_proba(queries)
    n_targets = prediction.shape[1]
    targets = [target_mapping[t] 
            for t in range(n_targets)]

    if not isinstance(prediction, np.ndarray):
        prediction = prediction.A

    prediction = pd.DataFrame(prediction, 
        index=query_index,
        columns=targets)
    prediction_probs = pd.DataFrame(prediction_probs,
        index=query_index,
        columns=targets)

    output_dir = os.path.join(
        args.output, 
        "{}-{}".format(Path(model_filename).stem, Path(query_filename).stem)
        )
    os.makedirs(output_dir, exist_ok=True)

    prediction_filename = os.path.join(output_dir,
        "predictions.csv")
    print ("writing predictions to", prediction_filename)
    prediction.to_csv(prediction_filename)
    
    probs_filename = os.path.join(output_dir,
        "probs.csv")
    print ("writing probs to", probs_filename)
    prediction_probs.to_csv(probs_filename)

    # rank top n targets for each query
    n = 100

    prediction_probs = prediction_probs.values
    idx = prediction_probs.argsort(axis=-1,)[:, ::-1][:,:n]
    predictions_ranked_filename = os.path.join(output_dir,
        "top_{}_targets.tsv".format(n))
    print ("writing top", n, "targets for each query to",
        predictions_ranked_filename)
    with open(predictions_ranked_filename, "w") as f:
        f.write("Query Compound\t{}\n".format(
            "\t".join(("Target {}".format(i+1) 
                for i in range(n)))))
        for i, row in enumerate(idx):
            f.write("{}".format(query_index[i]))
            for target_id in row:
                prob = prediction_probs[i, target_id]
                if prob > 0:
                    f.write("\t{}".format(target_mapping[target_id], ))
            f.write("\n")


if __name__ == "__main__":
    main()