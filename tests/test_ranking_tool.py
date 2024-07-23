# Tests ranking tool by fetching items from the feature store and ranking them.

# Set up the environment
import setup_env

# import relevant modules
from llm4crs.retrieval.fetch_tool import FetchFeatureStoreItemsTool
from llm4crs.ranking.rank_tool import RankFeatureStoreTool
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

# Initialize the fetchFeatureStore tool
fetch_tool = FetchFeatureStoreItemsTool('FeatureStoreItemTool',
                                        item_corpus, candidate_buffer,
                                        'Fetches items from the feature store')

# Initialize the rankFeatureStore tool
rank_tool = RankFeatureStoreTool('FeatureStoreRankTool',
                                 'Rank items from the feature store',
                                 item_corpus, candidate_buffer, fetch_tool)

# Fetch items from the feature store
term = "milk"

# run the fetch tool
fetch_tool.run(term)

# run the rank tool
rank_tool.run()

# print all information about recommended items
print(item_corpus.convert_id_2_info(candidate_buffer.memory))