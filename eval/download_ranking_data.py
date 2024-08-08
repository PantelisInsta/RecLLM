# Download training data from feature store

# Set up the environment
import setup_env

# import relevant modules
from llm4crs.utils.feature_store import fetch_recall_rank_features
import pandas as pd
import ast
import json
import os

# Set up environment variables that define data paths
from llm4crs.environ_variables import *

# Save data directory
DATA_DIR = f"{os.getcwd()}/data"
data = []

# Parameters
N = 1000             # number of data points to fetch
RETAILER_ID = 12     # retailer id
features = ['ITEMS'] # features to fetch

# load search terms for ranking tool (csv file)
search_terms = pd.read_csv(RANKING_SEARCH_TERMS_FILE)

# randomly select N search terms
search_terms = search_terms.sample(N)

# fetch data for each search term and save question, item ids, names and target
# to json file 

for i, term in search_terms.iterrows():
    
    d = {}
    d['question'] = term[0]

    print(f"Fetching data for search term: {term}")
    rank_data = fetch_recall_rank_features(term[0], retailer_id=RETAILER_ID, features=features)
    # parse items
    try:
        items = list(rank_data['ITEMS'].values[0])
    except:
        continue
    
    parsed_items = []
    for item in items:
        parsed_items.append(ast.literal_eval(item.strip('"')))
    items = pd.DataFrame(parsed_items)

    # remove duplicate items
    items = items.drop_duplicates(subset='product_id')

    # if none of the items are converted, skip
    if items['converted'].sum() == 0:
        continue

    # get ids of items
    d['id'] = items['product_id'].values.tolist()
    # get converted list
    d['target'] = items['converted'].values.tolist()
    # get item names
    d['name'] = items['product_name'].values.tolist()

    data.append(d)

# save to jsonl file
with open(f"{DATA_DIR}/ranking_data.jsonl", 'w') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')