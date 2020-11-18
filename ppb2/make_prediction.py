import os
import argparse

import numpy as np
import pandas as pd 

import pickle as pkl 

from pathlib import Path

from models import load_model
from data_utils import read_smiles, sdf_to_smiles

from db_utils import write_hits_to_db, write_probs_to_db, write_pathway_enrichment_to_db
from enrichment_analysis import perform_enrichment_analysis

def write_hits(prediction, output_dir):
    # write target predictions to file
    assert isinstance(prediction, pd.DataFrame)
    prediction = prediction.astype(bool)
    if isinstance(prediction, pd.DataFrame):
        query_index = prediction.index
        targets = prediction.columns
        prediction = prediction.values
    hit_dir = os.path.join(output_dir,
        "hits")
    os.makedirs(hit_dir, exist_ok=True)
    for i, row in enumerate(prediction):
        query = query_index[i]
        hits = targets[row]
        # ignore empty?
        query_hit_filename = os.path.join(
            hit_dir, "{}_hits.txt".format(query))
        print ("writing hits for query", query,
            "to", query_hit_filename)
        with open(query_hit_filename, "w") as f:
            f.write("\n".join(hits))

def write_top_k_hits(probs, output_dir, k=100):
    # rank top k hits for each query
    assert isinstance(probs, pd.DataFrame)
    if isinstance(probs, pd.DataFrame):
        query_index = probs.index
        targets = probs.columns
        probs = probs.values
    idx = probs.argsort(axis=-1,)[:, ::-1][:,:k]

    predictions_ranked_filename = os.path.join(output_dir,
        "top_{}_targets.tsv".format(k))
    print ("writing top", k, "targets for each query to",
        predictions_ranked_filename)
    with open(predictions_ranked_filename, "w") as f:
        f.write("Query Compound\t{}\n".format(
            "\t".join(("Target {}".format(i+1) 
                for i in range(n)))))
        for i, row in enumerate(idx):
            f.write("{}".format(query_index[i]))
            for target_id in row:
                prob = probs[i, target_id]
                if prob > 0:
                    f.write("\t{}".format(target_mapping[target_id], ))
            f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query",)
    parser.add_argument("--model",)
    parser.add_argument("--output", default="predictions")

    parser.add_argument("--n_proc", default=4, type=int)

    parser.add_argument("-k", 
        default=200, 
        type=int)
    
    parser.add_argument("--write-hits", action="store_true")
    parser.add_argument("--write-probs", action="store_true")
    parser.add_argument("--write-top-hits", action="store_true")
    parser.add_argument("--perform_enrichment", action="store_true")

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
    if hasattr(model, "set_k"):
        model.set_k(args.k)
    if hasattr(model, "n_proc"):
        model.set_n_proc(args.n_proc)

    # load queries (SMILES format)
    query_filename = args.query 
    print ("reading_queries from", query_filename)
    if query_filename.endswith(".sdf"):
        print ("converting from .SDF")
        queries = sdf_to_smiles(query_filename, 
            # index="coconut_id" #TODO
            )
    else:
        print ("reading SMILES")
        queries = read_smiles(query_filename)

    query_index = queries.index 
    n_queries = queries.shape[0]
    print ("number of queries:", n_queries)

    output_dir = os.path.join( # add model name to output dir
        args.output, 
        "{}-{}".format(Path(model_filename).stem, Path(query_filename).stem)
        )
    os.makedirs(output_dir, exist_ok=True)

    prediction = model.predict(queries).astype(int)
    
    n_targets = prediction.shape[1]
    targets = np.array([target_mapping[t] 
        for t in range(n_targets)])

    if not isinstance(prediction, np.ndarray):
        prediction = prediction.A

    prediction = pd.DataFrame(prediction, 
        index=query_index,
        columns=targets)

    prediction_filename = os.path.join(output_dir,
        "predictions.csv.gz")
    print ("writing predictions to", prediction_filename)
    prediction.to_csv(prediction_filename)

    if args.write_hits:
        # write_hits(prediction, output_dir)
        write_hits_to_db(model_filename, prediction)

    # identify target hits
    if args.perform_enrichment:
        targets_hit = targets[prediction.any(axis=0)]
        hit_filename = os.path.join(output_dir,
            "all_hits.txt")
        print ("writing hits to", hit_filename)
        with open(hit_filename, "w") as f:
            f.write("\n".join(targets_hit))

        enrichment, found, not_found = \
            perform_enrichment_analysis(hit_filename,
            csv_filename=os.path.join(output_dir, "enrichment.csv"),
            found_filename=os.path.join(output_dir, "found.txt"),
            not_found_filename=os.path.join(output_dir, "not_found.txt"),
            pdf_filename=os.path.join(output_dir, "enrichment.pdf"),
            )

        write_pathway_enrichment_to_db(model_filename,
            targets_hit,
            enrichment,
            found,
            not_found)

    del prediction

    prediction_probs = model.predict_proba(queries)

    prediction_probs = pd.DataFrame(prediction_probs,
        index=query_index,
        columns=targets)

    probs_filename = os.path.join(output_dir,
        "probs.csv.gz")
    print ("writing probs to", probs_filename)
    prediction_probs.to_csv(probs_filename)

    if args.write_top_hits:
        write_top_k_hits(prediction_probs, output_dir, k=100)
 
    if args.write_probs:
        write_probs_to_db(model_filename, prediction_probs)

if __name__ == "__main__":
    main()