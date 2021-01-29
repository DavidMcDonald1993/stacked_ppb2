import os 

import itertools

import pandas as pd 

import pickle as pkl

import glob

def main():

    # fps = ("morg2", "morg3", "rdk", "rdk_maccs", "circular", "maccs")
    # models = ("nb", "nn", "nn+nb", "lr", "bag", "svc", )

    results_dir = os.path.join("results_bluebear", )

    collated_results = []
    for filename in glob.iglob(os.path.join(results_dir,
        "*.pkl")):

        base = os.path.basename(filename)
        base_split = base.split("-")

        print ("reading", filename )

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                results = pkl.load(f)

            collated_results.append(results)

    collated_results = pd.DataFrame(collated_results, )
    collated_results_filename = os.path.join(results_dir, 
        "PPB2-collated_results.csv")
    print("writing to", collated_results_filename)
    collated_results.to_csv(collated_results_filename)

if __name__ == "__main__":
    main()