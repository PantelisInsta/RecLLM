import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import subprocess
import os

# zsh script that sources environment and sets env variables
env_output = subprocess.check_output(['zsh', '-c', 'source setup_env_var.zsh && env'])

# Update the Python process environment
for line in env_output.decode().split('\n'):
    if line and '=' in line:
        key, value = line.split('=', 1)
        os.environ[key] = value

from second_pass_search_ranker.utils import feature_store_utils

# Fetch data from feature store
def fetch_recall_rank_features(term, retailer_id, features):
    builders = []
    feature_builder = feature_store_utils.get_builder(
            shard_entity_names=['retailer_id', 'term'],
            shard_entity_values=[[retailer_id, term]],
            features=features,
            source_name='search_llm_agent_query_recall_ranking',
            secondary_entity_names=[],
            source_version=2
    )
    builders.append(feature_builder)
    df = feature_store_utils.get_data_from_builders(builders)
    return df


def fetch_recommendation_features(term, retailer_id, content_type, features):
    builders = []
    feature_builder = feature_store_utils.get_builder(
            shard_entity_names=['retailer_id','term', 'content_type'],
            shard_entity_values=[[retailer_id,term,content_type]],
            features=features,
            source_name='search_llm_agent_content_query_to_product_mapping',
            secondary_entity_names=[],
            source_version=2
    )
    builders.append(feature_builder)
    df = feature_store_utils.get_data_from_builders(builders)
    return df


features = ['child_search_id','retailer_name','query_classification','items']
df_rank = fetch_recall_rank_features('milk', 12, features)[0]
print(df_rank)

features = ['retailer_name', 'subtitle', 'product_ids', 'product_names']
df_rec = fetch_recommendation_features('milk', 12, 'substitute', features)[0]
print(df_rec)