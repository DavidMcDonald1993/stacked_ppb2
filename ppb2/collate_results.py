import os 

import itertools

import pandas as pd 

import pickle as pkl


def main():

    fps = ("morg2", "morg3", "rdk", "circular", "maccs")
    models = ("nb", "nn", "nn+nb", "lr", "bag", "svc")

    results_dir = os.path.join("results", )

    collated_results = []

    for fp, model in itertools.product(fps, models):
        filename = os.path.join(results_dir, 
            "{}-{}-results.pkl".format(fp, model))
        print ("reading", filename)
        assert os.path.exists(filename)

        with open(filename, "rb") as f:
            results = pkl.load(f)

        # df = pd.read_csv(filename, index_col=0)
        # df.index = df.index.map(lambda x: fp+"-"+x)
        collated_results.append(results)

    collated_results = pd.DataFrame(collated_results, axis=0)
    collated_results_filename = os.path.join(results_dir, 
        "PPB2-collated_results.csv")
    print("writing to", collated_results_filename)
    collated_results.to_csv(collated_results_filename)

if __name__ == "__main__":
    main()