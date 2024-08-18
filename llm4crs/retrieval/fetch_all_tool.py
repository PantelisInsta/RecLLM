# Tool that fetches items from both ranking and recommendation APIs

# imports
import pandas as pd
import numpy as np
from loguru import logger
import ast

from llm4crs.utils.feature_store import fetch_retrieval_features, fetch_recall_rank_features
from llm4crs.utils import SentBERTEngine

FEATURES_REC = ['retailer_name', 'products']
FEATURES_RANK = ['ITEMS']

QUERY_CANDIDATE_INFO = ['global_ctr','retailer_ctr','relevance_score', 'exact_match', 'strong_substitute',
                        'weak_subsitute', 'close_complement', 'remote_complement']
QUERY_CANDIDATE_INFO_DESCRIPTIONS = ['Global click-through rate','Retailer click-through rate','Relevance score',
                        'Exact match', 'Strong substitute', 'Weak substitute', 'Close complement', 'Remote complement']

class FetchAllTool:
    """
    Defines a tool to fetch items from the recommendation and ranking APIs of the feature store.
    The tool receives a search term, rewrites the search term to match the terms in the recommendation
    and ranking APIs, fetches items from the feature store, converts the feature store indexes to item ids,
    removes duplicates and updates the candidate bus with the new candidates.

    Args:
        name (str): The name of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        desc (str): The description of the tool.
        use_reco_tool (bool): Whether to use the recommendation tool for fetching items.
        terms_rec (str): Directory to a csv file containing recommendation API terms for fuzzy search.
        terms_rank (str): Directory to a csv file containing ranking API terms for fuzzy search.
        retailer_id (int): The retailer id.
    """

    def __init__(self, name, desc, item_corpus, buffer, use_reco_tool = True, 
                 terms_rec=None, terms_rank=None, retailer_id=12):
        
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.use_reco_tool = use_reco_tool
        self.retailer_id = retailer_id

        # if terms_rec is not none, define a sentence transformer engine for fuzzy search
        if terms_rec and use_reco_tool:
            # terms is a directory to a csv file. load that file
            terms_rec = pd.read_csv(terms_rec)
            # convert terms to ndarray
            self.terms_rec = np.array(terms_rec).flatten()
            
            self.engine_rec = SentBERTEngine(self.terms_rec, 
                                         list(range(len(self.terms_rec))), 
                                         model_name="thenlper/gte-base", 
                                         case_sensitive=False)
        else:
            self.terms_rec = None

        # if terms_rank is not none, define a sentence transformer engine for fuzzy search
        if terms_rank:
            # terms is a directory to a csv file. load that file
            terms_rank = pd.read_csv(terms_rank)
            # convert terms to ndarray
            self.terms_rank = np.array(terms_rank['TERM']).flatten()
            
            self.engine_rank = SentBERTEngine(self.terms_rank, 
                                         list(range(len(self.terms_rank))), 
                                         model_name="thenlper/gte-base", 
                                         case_sensitive=False)
        else:
            self.terms_rank = None

        
    def fetch_rec_items(self, term, content_type= ['substitute','complementary','theme']):
        """
        Fetches items from the recommendation API of the feature store for a given search term.
        Returns a list of feature store product indexes.
        """

        # Get dataframe of items
        df = fetch_retrieval_features(term, 12, content_type, FEATURES_REC)
        
        # for every content type, extract product indexes from product_ids column and convert to list of integers
        idx = []
        for i in range(len(content_type)):
           # get list where each element corresponds to a subquery
            subqueries = df[i]['products'].values[0]
            # Check if there are no products for this content type
            if subqueries is None:
                continue
            # iterate over subquery to extract product indexes 
            for j in range(len(subqueries)):
                subquery = ast.literal_eval(subqueries[j])
                # extract product indexes from product_ids column
                idx += list(subquery['product_ids'])
        # Convert to list of integers
        idx = [int(x) for x in idx]
        # Make sure there are no duplicates
        idx = list(set(idx))

        return idx

    def fetch_rank_items(self, term):
        """
        Fetches items from the ranking API of the feature store for a given search term.
        Returns a list of feature store product indexes.
        """

        # Get data
        rank = fetch_recall_rank_features(term,retailer_id=self.retailer_id,features=FEATURES_RANK)
        data = list(rank['ITEMS'].values[0])

        # Parse data to extract items
        parsed_data = []
        for item in data:
            # Remove the outer quotes and use ast.literal_eval to safely convert to dict
            dict_item = ast.literal_eval(item.strip('"'))
            parsed_data.append(dict_item)

        # make dataframe
        items_rank = pd.DataFrame(parsed_data)

        # remove duplicate products
        items_rank = items_rank.drop_duplicates(subset='product_id')

        # convert product_id (index) to candidate buffer id
        items_rank['id'] = items_rank['product_id'].apply(self.item_corpus.convert_index_2_id)
        
        # get dictionary of dictionaries with detailed information about the items
        query_candidate_info = {row['id']: self.get_query_candidate_info(row) for _, row in items_rank.iterrows()}

        # save detailed candidate-query information to candidate buffer
        self.buffer.update_query_candidate_info(query_candidate_info)

        # return product ids column of items_rank as list of integers
        return items_rank.product_id.tolist()

    def get_query_candidate_info(self, data, columns = QUERY_CANDIDATE_INFO,
                                descriptions = QUERY_CANDIDATE_INFO_DESCRIPTIONS):
        """
        Parses a Series object to extract all relevant information about item-query interactions.
        """
        info = {}
        for i, column in enumerate(columns):
            try:
                info[descriptions[i]] = data[column]
            except:
                pass

        return info

    def convert_index_2_id(self, indexes):
        """
        Converts feature store indexes to item ids.
        """
        return self.item_corpus.convert_index_2_id(indexes)
    
    def fuzzy_search(self, term, engine, API = "Recommendation"):
        """
        Fuzzy searches for a term in the recommendation or ranking API terms.
        Returns the most similar term.
        """
        logger.debug(f"{API} tool rewrite search term: {term}")
        new_term = engine(term,topk=1,return_doc=True)[0]
        logger.debug(f"New term: {new_term}")

        return new_term
    
    def run(self, term):
        """
        Updates the candidate bus with new candidates from the recommendation and ranking APIs of the feature store.
        """
        info = ""
        if self.use_reco_tool:
            # If term is not in terms, run fuzzy engine
            if self.terms_rec is not None and term not in self.terms_rec:
                new_term = self.fuzzy_search(term, self.engine_rec, API = "Recommendation")
                info += f"System: Fuzzy search replaced term '{term}' with '{new_term}' in the recommendation tool.\n"
                self.term_rec = new_term
            else:
                self.term_rec = term
            
            # Fetch items from recommendation API
            try:
                rec_indexes = self.fetch_rec_items(self.term_rec)
            except:
                rec_indexes = []
                info += f"System: No items found in the recommendation tool with the term '{term}'.\n"
        else:
            rec_indexes = []
            info += f"System: The recommendation tool is disabled.\n"
        
        # If term is not in terms, run fuzzy engine
        if self.terms_rank is not None and term not in self.terms_rank:
            new_term = self.fuzzy_search(term, self.engine_rank, API="Ranking")
            info += f"System: Fuzzy search replaced term '{term}' with '{new_term}' in the ranking tool.\n"
            self.term_rank = new_term
        else:
            self.term_rank = term
        
        # Fetch items from ranking API
        rank_indexes = self.fetch_rank_items(self.term_rank)

        # merge rec_indexes and rank_indexes and remove duplicates
        all_idx = rec_indexes + rank_indexes
        all_idx = list(set(all_idx))

        # Convert indexes to ids
        all_ids = self.item_corpus.convert_index_2_id(all_idx)
        
        # Update candidate bus with push method
        self.buffer.push("Fetch all items tool",all_ids)
        # update info
        info += f"Here are candidates id searched with the term: [{','.join(map(str, all_ids))}]."

        # Return message with ids
        return info