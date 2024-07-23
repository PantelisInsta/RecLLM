# Ranking tool based on instacart feature store

from llm4crs.utils.feature_store import fetch_recall_rank_features
import pandas as pd
import ast

class RankFeatureStoreTool:
    """
    Defines a tool that fetches items from the feature store for ranking. The tool receives a search term,
    fetching ranking recommendations from the feature store, keeps only recommended items that are in the 
    candidate bus, picks the top recommendations these, and updates the candidate bus with the new candidates.
    
    Args:
        name (str): The name of the tool.
        desc (str): The description of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        fetch_tool (class): The tool used for fetching items.
        features (list): The features to fetch.
        retailer_id (int): The retailer id.
        top_k (int): The number of top recommendations to keep.
    """
        
    def __init__(self, name, desc, item_corpus, buffer, fetch_tool, top_k=5,
                  features=['ITEMS'], retailer_id=12):
        
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.fetch_tool = fetch_tool
        self.features = features
        self.retailer_id = retailer_id
        self.top_k = top_k

    def fetch_rank_items(self):
        """
        Fetches items from the feature store for a given search term.
        Returns a list of feature store product indexes.
        """

        # Get data
        rank = fetch_recall_rank_features(self.term,retailer_id=self.retailer_id,features=self.features)
        data = list(rank['ITEMS'].values[0])

        # Parse data to extract items
        parsed_data = []
        for item in data:
            # Remove the outer quotes and use ast.literal_eval to safely convert to dict
            dict_item = ast.literal_eval(item.strip('"'))
            parsed_data.append(dict_item)

        # make dataframe
        self.items_rank = pd.DataFrame(parsed_data)

    # Check which items are in the candidate bus, and only keep those
    def filter_rank_items(self):
        """
        Keeps recommended items that are in the candidate bus, and includes indexes of the items in the candidate bus
        to the rank dataframe.
        """

        # Extract product indexes from product_ids column and convert to list of integers
        product_indexes = self.items_rank['product_id'].tolist()

        # Convert from instacart product IDs to internal product IDs using the corpus
        ids = [self.item_corpus.convert_index_2_id(idx) for idx in product_indexes] 

        # Add ids list to the self.items_rank dataframe
        self.items_rank['id'] = ids

        # Filter out items that are not in the candidate bus
        ids = [idx for idx in ids if idx in self.buffer.memory]
        
        # Update items in rank
        self.items_rank = self.items_rank[self.items_rank['id'].isin(ids)]

    # Run the tool

    def run(self, term=None):
        """
        Updates the candidate bus with recommended candidates from feature store.
        """

        # if a term is not provided use the fetch tool term
        if term is None:
            term = self.fetch_tool.term
        self.term = term

        # Fetch items from rank feature store
        self.fetch_rank_items()

        # Filter items to make sure they are in the candidate bus
        self.filter_rank_items()

        # Pick top k items according to relevance score
        self.items_rank = self.items_rank.sort_values(by='relevance_score', ascending=False)
        self.items_rank = self.items_rank.head(self.top_k)

        # Update the candidate bus with the new candidates
        ids = self.items_rank['id'].tolist()
        self.buffer.push("Feature store ranking tool",ids)

        # Return message with ids
        return f"Here are the recommended candidate ids: [{','.join(map(str, ids))}]."