import os 

import itertools

import pandas as pd 


def main():

    fps = ("ecfp4", "morg2", "circular", "MACCS")
    models = ("nb", "nn", "nn+nb")

    results_dir = os.path.join(".")

    collated_results = []

    for fp, model in itertools.product(fps, models):
        filename = os.path.join(results_dir, 
            "{}-{}-results.csv".format(fp, model))
        print ("reading", filename)
        assert os.path.exists(filename)

        df = pd.read_csv(filename, index_col=0)
        df.index = df.index.map(lambda x: fp+"-"+x)
        collated_results.append(df)

    collated_results = pd.concat(collated_results, axis=0)
    collated_results_filename = os.path.join(results_dir, 
        "PPB2-collated_results.csv")
    print("writing to", collated_results_filename)
    collated_results.to_csv(collated_results_filename)

if __name__ == "__main__":
    main()