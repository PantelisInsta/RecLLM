# Set up the environment
import setup_env

# import relevant modules
import pandas as pd
import ast
from llm4crs.utils.feature_store import *

term = "milk"
retailer_id = 12

# fetch items from the feature store
# retrieval features
rec = fetch_recommendation_features(term,retailer_id,'substitute',features=['product_ids','product_names'])
# make dataframe
ids = rec.product_ids.values[0]
ids = [int(x) for x in ids]
names = list(rec.product_names.values[0])
items_rec = pd.DataFrame({'product_id':ids,'product_name':names})

# rank features
rank = fetch_recall_rank_features(term,retailer_id,features=['ITEMS'])
data = list(rank['ITEMS'].values[0])

parsed_data = []
for item in data:
    # Remove the outer quotes and use ast.literal_eval to safely convert to dict
    dict_item = ast.literal_eval(item.strip('"'))
    parsed_data.append(dict_item)

# make dataframe
items_rank = pd.DataFrame(parsed_data)

# create column in rank that is 1 if the product is in the retrieval list, 0 otherwise
items_rank['in_retrieval'] = items_rank['product_id'].apply(lambda x: 1 if x in ids else 0)
# sort by relevance score
items_rank = items_rank.sort_values(by='relevance_score',ascending=False)

# print items rank dataframe
print(items_rank)

print(f"Number of items in retrieval: {items_rec.shape[0]}")
print(f"Number of items in rank that are in retrieval: {items_rank['in_retrieval'].sum()}")
print(f"Number of items in rank: {items_rank.shape[0]}")