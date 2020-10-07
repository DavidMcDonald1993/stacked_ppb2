import os 
import glob 

import pickle as pkl

import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import count

from scipy import sparse as sp

def construct_compound_to_targets(df):
    print ("constructing active to target mapping")
    compounds_to_targets = defaultdict(list)
    for _, row in df.iterrows():
        compounds_to_targets[row["ChEMBL compound ID"]]\
            .append(row["Target ID"])
    return compounds_to_targets
    
def to_array(column, delimiter=";"):
    assert isinstance(column, pd.Series)
    if delimiter is not None:
        arr = np.array([list(filter(None, x.split(delimiter) ))
                for x in column.values], dtype=np.int16)
    else:
        arr = np.array([ [y for y in x]
                for x in column.values], dtype=np.int16)
    if np.array_equal(arr, arr.astype(bool)):
        print ("converting to bool")
        arr = arr.astype(bool)
        # arr = sp.csr_matrix(arr)
    return arr 

def filter_df(df,
    source_organisms={"Homo sapiens", "Rattus norvegicus"},
    min_actives=10,
    max_nM=10000):

    # filter by target organism
    df = df.loc[df["Source of target"].isin(source_organisms)]

    # filter by bioactivity
    df = df.loc[~df["Bioactivity of compound"].\
        str.contains(">")
    ]
    df = df.loc[
        df["Bioactivity of compound"].str.contains("IC50_nM") |
        df["Bioactivity of compound"].str.contains("EC50_nM") |
        df["Bioactivity of compound"].str.contains("Ki_nM") |
        df["Bioactivity of compound"].str.contains("Kd_nM") 
    ]
    df = df[df.apply(lambda row: 
        float(row["Bioactivity of compound"].split("_")[-1]) <= max_nM,
        axis=1)]

    df = df.groupby('Target ID').filter(
        lambda x: x['Target ID'].count() >= min_actives)

    return df

def main():

    data_dir = os.path.join("data", "original")

    # filter by target type
    with open(os.path.join(data_dir, 
        "target_mapping.pkl"), "rb") as f:
        target_mapping = pkl.load(f)

    targets = (os.path.basename(f).split(".")[0]
        for f in glob.glob(os.path.join(data_dir,
            "*.smi.gz")))

    targets = filter(lambda target:
        target in target_mapping \
            and target_mapping[target]=="SINGLE PROTEIN", 
        targets)
    
    data_files = (os.path.join(data_dir, target + ".smi.gz")
        for target in targets)

    min_actives = 10

    print ("reading actives and filtering")
    df = pd.concat([
        filter_df(
            pd.read_csv(data_file, index_col=0, sep="\t"),
            min_actives=min_actives)
        for data_file in data_files
    ], axis=0)
    print ()

    # build mapping from compound to target list
    compounds_to_targets = construct_compound_to_targets(df)

    output_dir = os.path.join("data", 
        "min_actives={}".format(min_actives))
    if not os.path.exists(output_dir):
        print ("making directory", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # output mapping to file
    compounds_to_targets_filename = os.path.join(output_dir,
        "compounds_to_targets.pkl")
    print ("writing compounds to targets to", 
        compounds_to_targets_filename)
    with open(compounds_to_targets_filename, "wb") as f:
        pkl.dump(compounds_to_targets, f, pkl.HIGHEST_PROTOCOL)

    num_compounds = len(compounds_to_targets)
    num_targets = len(set(df["Target ID"]))

    print ("number of compounds:", num_compounds)
    print ("number of targets:", num_targets)
    print ()

    # remove unnecessary columns
    columns_of_interest = [
        "ChEMBL compound ID",
        "SMILES", 
        "Xfp fingerprint",
        "MQN fingerprint",
        "SMIfp fingerprint",
        "Daylight type substructure fingerprint (Sfp)",
        "Extended connectivity fingerprint (ECfp4)",
        "APfp fingerprint",
    ]
    assert all([column in df.columns 
        for column in columns_of_interest])

    print ("creating compound info dataframe")
    df = df[columns_of_interest].drop_duplicates()
    df = df.set_index("ChEMBL compound ID")

    assert len(set(df.index)) == df.shape[0]

    compounds_filename = os.path.join(output_dir,
        "compounds.csv")
    print ("writing compounds to", compounds_filename)
    df.to_csv(compounds_filename)

    print ("writing npy files for each fingerprint")
    for column in df.columns:
        if "fingerprint" in column:
            print ("processing fingerprint",
                column)
            filename = os.path.join(output_dir,
                column.replace(" ", "_") + ".npy")
            array = to_array(df[column])
            print ("writing to", filename)
            np.save(filename, array)

    # write compound smiles
    smiles_filename = os.path.join(output_dir,
        "compounds.smi")
    print ("writing SMILES to", smiles_filename)
    with open(smiles_filename, "w") as f:
        for compound, smi in df["SMILES"].items():
            f.write("{}\t{}\n".format(smi, compound))

    print ("converting labels to sparse matrix")
    compounds_to_id = {compound: i 
        for i, compound in enumerate(df.index)}
    targets_to_id = defaultdict(count().__next__)

    indices = [
        (compounds_to_id[compound], targets_to_id[target]) 
        for compound, targets in compounds_to_targets.items()
        for target in targets]
    data = [True] * len(indices)
    shape = num_compounds, num_targets

    targets = sp.csr_matrix(
        (data, tuple(zip(*indices))), 
        shape=shape,
        dtype=bool)
    target_filename = os.path.join(output_dir,
        "targets.npz")
    print ("writing sparse targets to", target_filename)
    sp.save_npz(target_filename, targets)

    id_to_compound = {v: k 
        for k, v in compounds_to_id.items()}
    id_to_target = {v: k 
        for k, v in targets_to_id.items()}

    id_to_compound_filename = os.path.join(output_dir,
        "id_to_compound.pkl")
    id_to_target_filename = os.path.join(output_dir,
        "id_to_target.pkl")

    print ("saving id_to_compound to",
        id_to_compound_filename)
    with open(id_to_compound_filename, "wb") as f:
        pkl.dump(id_to_compound, f, pkl.HIGHEST_PROTOCOL)

    print ("saving id_to_target to",
        id_to_target_filename)
    with open(id_to_target_filename, "wb") as f:
        pkl.dump(id_to_target, f, pkl.HIGHEST_PROTOCOL)
  
if __name__ == "__main__":
    main()
