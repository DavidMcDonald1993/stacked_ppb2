import os
import argparse

import numpy as np
import pandas as pd 

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.metrics import pairwise_distances
# from sklearn.multiclass import OneVsRestClassifier

# from scipy import sparse as sp

import pickle as pkl 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp", default="morg2",
        choices=["mqn", "xfp", "ecfp4", 
        "morg2", "morg3", "rdk", 
        "circular", "maccs"])

    parser.add_argument("--model", default="nb",
        choices=["nn", "nb", "nn+nb", 
            "lr", "svc", "bag", "stack"])

    parser.add_argument("--target_id", type=int)

    parser.add_argument("--query",)

    parser.add_argument("-k", default=200, 
        type=int)

    return parser.parse_args()

def main():

    args = parse_args()

    # data_dir = os.path.join("data", )
    
    # X = load_fingerprints(args.fp,
    #     fingerprint_dir=data_dir)
    # print ("training fingerprints shape is", 
    #     X.shape)

    # labels_filename = os.path.join(data_dir,
    #     "targets.npz")
    # print ("loading labels from", labels_filename)
    # Y = sp.load_npz(labels_filename).A
    # print ("labels shape is", Y.shape)

    # n_targets = Y.shape[1]

    target_id = args.target_id

    model_filename = os.path.join("models", 
        "target-{}-{}-{}.pkl".format(target_id, 
                args.fp, args.model))
    assert os.path.exists(model_filename)
    print ("reading model from", model_filename)
    with open(model_filename, "rb") as f:
        model = pkl.load(f)

    # load queries (SMILES format)
    query_filename = args.query 
    print ("reading_queries from", query_filename)
    queries = pd.read_csv(query_filename, header=None, 
        sep="\t", index_col=1)[0].astype("string")

    query_index = queries.index 
    n_queries = queries.shape[0]
    print ("number of queries:", n_queries)

    prediction = model.predict(queries)
    prediction_probs = model.predict_proba(queries)
    prediction_log_probs = model.predict_log_proba(queries)

    hits = query_index[prediction.astype(bool)]

    id_to_target_filename = os.path.join("data", 
        "id_to_gene_symbol.pkl")
    with open(id_to_target_filename, "rb") as f:
        id_to_target = pkl.load(f)

    print ("hits for", id_to_target[target_id], ":")
    print (hits)

    # columns = [id_to_target[i] for i in range(n_targets)]

    # id_to_compound_filename = os.path.join("data", 
    #     "id_to_compound.pkl")
    # with open(id_to_compound_filename, "rb") as f:
    #     id_to_compound = pkl.load(f)
    # # index = [id_to_compound[i] for i in range(n_queries)]

    # filename = os.path.join(".", 
    #      "{}-{}".format(args.fp, args.model, args.k))
    # if "nn" in args.model:
    #     filename += "-k={}".format(args.k)
    # predictions_filename = filename + "_predictions.csv"
    # print ("writing predictions to", predictions_filename)
    # predictions = pd.DataFrame(predictions,
    #     index=query_index
    #     )
    # predictions.columns = [id_to_target[i] 
    #     for i in predictions.columns]
    # predictions.to_csv(predictions_filename)

    # n = 100
    # # rank top n targets for each query
    # predictions = predictions.values
    # idx = predictions.argsort(axis=-1,)[:, ::-1][:,:n]
    # predictions_ranked_filename = filename + \
    #     "_top_{}_targets.tsv".format(n)
    # print ("writing top", n, "targets for each query to",
    #     predictions_ranked_filename)
    # with open(predictions_ranked_filename, "w") as f:
    #     f.write("Query Compound\t{}\n".format(
    #         "\t".join(("Target {}".format(i+1) 
    #             for i in range(n)))))
    #     for i, row in enumerate(idx):
    #         f.write("{}".format(query_index[i]))
    #         for target_id in row:
    #             prob = predictions[i, target_id]
    #             if prob > 0:
    #                 f.write("\t{}".format(id_to_target[target_id], ))
    #         f.write("\n")


if __name__ == "__main__":
    main()