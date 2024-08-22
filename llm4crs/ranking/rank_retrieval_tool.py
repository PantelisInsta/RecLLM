# Tool that uses ranking API to fetch items, recommends top k items for that item category,
# and updates the candidate bus with the new candidates.

# imports
import pandas as pd
import numpy as np
from loguru import logger
import ast

from llm4crs.utils.feature_store import fetch_recall_rank_features_user_seq
from llm4crs.utils import SentBERTEngine

class RankRetrievalFeatureStoreTool:

    """
    Defines a tool that fetches items from the feature store for ranking. The tool receives a search term,
    fetches ranking recommendations from the feature store, keeps only the top k most relevant items, 
    and updates the candidate bus with the new candidates.

    Args:
        name (str): The name of the tool.
        desc (str): The description of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        terms (str): The terms to use for fuzzy search.
        top_k (int): The number of top recommendations to keep.
        features (list): The features to fetch.
        retailer_id (int): The retailer id.
    """

    def __init__(self, name, desc, item_corpus, buffer, terms=None, top_k=5,
                  features=['ITEMS'], retailer_id=12):
        
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.features = features
        self.retailer_id = retailer_id
        self.top_k = top_k

        # if terms is not none, define a sentence transformer engine for fuzzy search
        if terms:
            # terms is a directory to a csv file. load that file
            terms = pd.read_csv(terms)
            # convert terms to ndarray
            self.terms = np.array(terms['TERM']).flatten()
            
            self.engine = SentBERTEngine(self.terms, 
                                         list(range(len(self.terms))), 
                                         model_name="thenlper/gte-base", 
                                         case_sensitive=False)
        else:
            self.terms = None


    def fuzzy_search(self, term):
        """
        Searches for the most similar term in the terms list.
        """

        logger.debug(f"Ranking tool rewrite search term: {term}")
        new_term = self.engine(term,topk=1,return_doc=True)[0]
        logger.debug(f"New term: {new_term}")

        return new_term


    def fetch_rank_items(self):
        """
        Fetches items from the feature store for a given search term.
        Returns a list of feature store product indexes.
        """

        # Get data
        rank = fetch_recall_rank_features_user_seq(self.term,retailer_id=self.retailer_id,features=self.features)
        data = list(rank['ITEMS'].values[0])

        # Parse data to extract items
        parsed_data = []
        for item in data:
            # Remove the outer quotes and use ast.literal_eval to safely convert to dict
            dict_item = ast.literal_eval(item.strip('"'))
            parsed_data.append(dict_item)

        # make dataframe
        self.items_rank = pd.DataFrame(parsed_data)

        # remove duplicate products
        self.items_rank = self.items_rank.drop_duplicates(subset='product_id')


    def run(self, term):
        """
        Fetches ranking recommendations from the feature store for a given search term, 
        keeps only the top k most relevant items, and updates the candidate bus with the new candidates.
        """

        # If term is not in terms, run fuzzy engine
        if self.terms is not None and term not in self.terms:
            new_term = self.fuzzy_search(term)
            self.term = new_term
        else:
            self.term = term

        # fetch items
        self.fetch_rank_items()

        # Pick top k items according to relevance score
        self.items_rank = self.items_rank.sort_values(by='relevance_score', ascending=False)
        self.items_rank = self.items_rank.head(self.top_k)

        # get product ids
        item_idx = self.items_rank['product_id'].values

        # convert product ids to internal ids
        ids = [self.item_corpus.convert_index_2_id(int(idx)) for idx in item_idx]

        # remove None values
        ids = [idx for idx in ids if idx is not None]

        # update buffer
        self.buffer.push("Feature store ranking tool",ids)

        # store items in basket and clean buffer
        self.buffer.store_and_clear(term)

        return f"Here are the recommended candidate ids: [{','.join(map(str, ids))}]."