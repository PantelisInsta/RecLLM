import pandas as pd
import numpy as np

from llm4crs.utils.feature_store import fetch_recommendation_features
from llm4crs.utils import SentBERTEngine

FEATURES = ['retailer_name', 'subtitle', 'product_ids', 'product_names']

class FetchFeatureStoreItemsTool:
    """
    Defines a tool to fetch items from the feature store. The tool receives a search term,
    fetches items from the feature store, converts the feature store indexes to item ids,
    and updates the candidate bus with the new candidates.

    Args:
        name (str): The name of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        desc (str): The description of the tool.
        terms (str): Directory to a csv file containing feature store terms for fuzzy search.
        content_type (str): The content type from the feature store.
        features (list): The features to fetch.
        retailer_id (int): The retailer id.
    """

    def __init__(self, name, item_corpus, buffer, desc, terms=None, content_type='substitute',
                  features=FEATURES, retailer_id=12):
        
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.content_type = content_type
        self.features = features
        self.retailer_id = retailer_id

        # if terms is not none, define a sentence transformer engine for fuzzy search
        if terms:
            # terms is a directory to a csv file. load that file
            terms = pd.read_csv(terms)
            # convert terms to ndarray
            self.terms = np.array(terms).flatten()
            
            self.engine = SentBERTEngine(self.terms, 
                                         list(range(len(self.terms))), 
                                         model_name="thenlper/gte-base", 
                                         case_sensitive=False)

    def fetch_items(self, term):
        """
        Fetches items from the feature store for a given search term.
        Returns a list of feature store product indexes.
        """

        # Get dataframe of items
        df = fetch_recommendation_features(term, 12, 'substitute', FEATURES)

        # Extract product indexes from product_ids column and convert to list of integers
        product_indexes = list(df['product_ids'].values[0])
        product_indexes = [int(x) for x in product_indexes]        

        return product_indexes
    
    def convert_index_2_id(self, indexes):
        """
        Converts feature store indexes to item ids.
        """
        return self.item_corpus.convert_index_2_id(indexes)
    
    def run(self, term):
        """
        Updates the candidate bus with new candidates.
        """
        # If term is not in terms, run fuzzy engine
        if self.terms is not None and term not in self.terms:
            term = self.fuzzy_search(term)
        # Fetch items from feature store
        indexes = self.fetch_items(term)
        ids = self.item_corpus.convert_index_2_id(indexes)
        # Update candidate bus with push method
        self.buffer.push("Fetch feature store items tool",ids)
        
        # Return message with ids
        return f"Here are candidates id searched with the term: [{','.join(map(str, ids))}]."
        

    def fuzzy_search(self, term):
        """
        Searches for the most similar term in the terms list.
        """
        return self.engine(term,topk=1,return_doc=True)[0]