# Ranking tool based on instacart feature store

from llm4crs.utils.feature_store import fetch_recall_rank_features
from llm4crs.utils import SentBERTEngine
from loguru import logger
import pandas as pd
import numpy as np
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
        
    def __init__(self, name, desc, item_corpus, buffer, fetch_tool, terms=None, top_k=5,
                  features=['ITEMS'], retailer_id=12):
        
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.fetch_tool = fetch_tool
        self.features = features
        self.retailer_id = retailer_id
        self.top_k = top_k

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
        else:
            self.terms = None

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

        # remove duplicate products
        self.items_rank = self.items_rank.drop_duplicates(subset='product_id')

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

        # Remove rows with None as id
        self.items_rank = self.items_rank.dropna(subset=['id'])

        # Filter out items that are not in the candidate bus
        ids = [idx for idx in ids if idx in self.buffer.memory]
        
        # Update items in rank
        self.items_rank = self.items_rank[self.items_rank['id'].isin(ids)]


    def run(self, term='null'):
        """
        Updates the candidate bus with recommended candidates from feature store.
        """

        # if a term is not provided use the fetch tool term
        if term == 'null':
            term = self.fetch_tool.term

        # Rewrite search term
        if self.terms is not None and term not in self.terms:
            new_term = self.fuzzy_search(term)
            info += f"System: Fuzzy search replaced term '{term}' with '{new_term}' in the ranking tool.\n"
            self.term = new_term
        else:
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
        # convert ids to list of integers
        ids = [int(x) for x in ids]
        self.buffer.push("Feature store ranking tool",ids)

        # Return message with ids
        return f"Here are the recommended candidate ids: [{','.join(map(str, ids))}]."
    
    
    def fuzzy_search(self, term):
        """
        Searches for the most similar term in the terms list.
        """

        logger.debug(f"Ranking tool rewrite search term: {term}")
        new_term = self.engine(term,topk=1,return_doc=True)[0]
        logger.debug(f"New term: {new_term}")

        return new_term