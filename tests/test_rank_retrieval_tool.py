# Test rank retrieval tool

# Set up the environment
import setup_env

# import relevant modules
from llm4crs.ranking.rank_retrieval_tool import RankRetrievalFeatureStoreTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.corpus import BaseGallery

# Set up environment variables that define data paths
from llm4crs.environ_variables import *

# Load corpus of items
item_corpus = BaseGallery(
    GAME_INFO_FILE,
    TABLE_COL_DESC_FILE,
    columns=USE_COLS,
    fuzzy_cols=["title"] + CATEGORICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
)

# Initialize the candidate buffer
candidate_buffer = CandidateBuffer(item_corpus)

# Initialize the rank retrieval tool
rank_retrieval_tool = RankRetrievalFeatureStoreTool('RankRetrievalFeatureStoreTool',
                                                    'Rank items from the feature store',
                                                    item_corpus, candidate_buffer,
                                                    terms=RANKING_SEARCH_TERMS_FILE)

# Fetch items from the feature store
term = "milk"

# run the rank retrieval tool
rank_retrieval_tool.run(term)

# print contents of the candidate buffer
print(candidate_buffer.basket_items)