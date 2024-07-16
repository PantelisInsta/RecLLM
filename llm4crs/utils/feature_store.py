from second_pass_search_ranker.utils import feature_store_utils


def fetch_recall_rank_features(term, retailer_id, features):
    """
    Fetches recall rank features from the feature store for a given
    search term and retailer_id.

    Args:
        term (str): search term
        retailer_id (int): retailer_id
        features (list): list of features to fetch
    """

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
    df = feature_store_utils.get_data_from_builders(builders)[0]

    return df


def fetch_recommendation_features(term, retailer_id, content_type, features):
    """
    Fetches recommendation features from the feature store for a given
    search term, retailer_id and content_type.

    Args:
        term (str): search term
        retailer_id (int): retailer_id
        content_type (str): content type (could be 'substitute', 'complement', 'theme')
        features (list): list of features to fetch
    """
    
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
    df = feature_store_utils.get_data_from_builders(builders)[0]

    return df