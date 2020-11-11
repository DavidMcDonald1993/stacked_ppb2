
from datetime import datetime

from pymongo import MongoClient

import numpy as np
import pandas as pd

HOST = "192.168.0.49"
PORT = 27017
DB = "COCONUT"

def connect_to_db():
    print ("connecting to MongoDB database", 
        DB, "using host",
        HOST, "and port", PORT)
    client = MongoClient(HOST, PORT)
    return client[DB]

def count_documents(collection, filter={}):
    return collection.count_documents(filter=filter)

def clear_collection(collection, filter={}):
    collection.remove(filter=filter)

def add_compounds(compounds):
    assert isinstance(compounds, list)
    db = connect_to_db()

    compound_collection = db["compounds"]

    print (compound_collection.count_documents(filter={}))

    db.client.close()

def write_hits_to_db(model_name, predictions,
    collection="hits"):
    assert isinstance(model_name, str)
    assert isinstance(predictions, pd.DataFrame)

    db = connect_to_db()

    print ("writing hits to MongoDB:", DB, 
        "collection:", collection)

    hit_collection = db[collection]

    records = []

    print ("determining hits")

    for compound, row in predictions.iterrows():
        
        for target in row.index:
            if row[target]:
                records.append({
                    # "compound_chembl_id": compound,
                    "compound": compound,
                    # "target_chembl_id": target,
                    "target": target,
                    "model": model_name,
                    "time": str(datetime.now())
                })

    print ("inserting", len(records), "records")
    hit_collection.insert_many(records)

    db.client.close()


def write_probs_to_db(model_name, probs,
    collection="probabilities"):
    assert isinstance(model_name, str)
    assert isinstance(probs, pd.DataFrame)
    db = connect_to_db()

    print ("writing hit probabilities to MongoDB:",
        DB, "collection:", collection)

    prob_collection = db[collection]

    records = []

    print ("identifying non-zero probabilities")

    for compound, row in probs.iterrows():
        
        for target in row.index:
            if row[target] > 0:
                records.append({
                    # "compound_chembl_id": compound,
                    "compound": compound,
                    # "target_chembl_id": target,
                    "target": target,
                    "probability": row[target],
                    "model": model_name,
                    "time": str(datetime.now())
                })

    print ("inserting", len(records), "records")
    prob_collection.insert_many(records)

    db.client.close()

def main():

    db = connect_to_db()

    hit_collection = db["hits"]

    print (count_documents(hit_collection))

    # predictions = pd.DataFrame(
    #     [[0, 0, 1], 
    #     [1, 1, 1],
    #     [1, 0, 0]], 
    #     index=["compound_1", "compound_2", "compound_3"],
    #     columns=["target_1", "target_2", "target_3"],
    #     dtype=bool)

    # write_hits_to_db("maccs-nb", predictions)

    # np.random.seed(0)

    # probs = pd.DataFrame(
    #     np.random.rand(3, 3), 
    #     index=["compound_1", "compound_2", "compound_3"],
    #     columns=["target_1", "target_2", "target_3"],
    #     )

    # write_probs_to_db("maccs-nb", probs)

    
if __name__ == "__main__":
    main()