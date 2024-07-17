# Tool to retrieve items from instacart feature store

from llm4crs.utils.feature_store import fetch_recommendation_features
from llm4crs.utils import SentBERTEngine

FEATURES = ['retailer_name', 'subtitle', 'product_ids', 'product_names']
FEATURE_STORE_QUERIES = None


class fetchFeatureStore:
    """
    Defines a tool to fetch items from the feature store. The tool receives a search term,
    fetches items from the feature store, converts the feature store indexes to item ids,
    and updates the candidate bus with the new candidates.
    """

    def __init__(self, item_corpus, candidate_bus, terms=FEATURE_STORE_QUERIES, content_type='substitute',
                  features=FEATURES, retailer_id=12):
        
        self.item_corpus = item_corpus # full corpus of items
        self.candidate_bus = candidate_bus # candidate bus to store candidates
        self.content_type = content_type # content type from feature store
        self.features = features    # features to fetch
        self.retailer_id = retailer_id  # retailer
        self.terms = terms  # contains all search terms in feature store, for fuzzy search if there is no exact term match

        # if terms is not none, define a sentence transformer engine for fuzzy search
        if terms:
            self.engine = SentBERTEngine(terms, 
                                         list(range(len(terms))), 
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
        if term not in self.terms:
            term = self.fuzzy_search(term)
        # Fetch items from feature store
        indexes = self.fetch_items(term)
        ids = self.item_corpus.convert_index_2_id(indexes)
        # Update candidate bus with push method
        self.candidate_bus.push("Fetch feature store items tool",ids)
        

    def fuzzy_search(self, term):
        """
        Searches for the most similar term in the terms list.
        """
        return self.engine(term, topk=1)[0]