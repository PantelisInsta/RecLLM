# Set up the environment
import setup_env

# import relevant modules
from llm4crs.retrieval.fetch_tool import FetchFeatureStoreItemsTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.corpus import BaseGallery

# Set up environment variables that define data paths
from llm4crs.environ_variables import *

# Variables
term = "milk"
content_type = ['substitute','complementary','theme']

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
                                        'Fetches items from the feature store',
                                        content_type=content_type)

# run the fetch tool
fetch_tool.run(term)

# check the candidate buffer
print(candidate_buffer.memory)

# print all information about these items from the corpus
print(item_corpus.convert_id_2_info(candidate_buffer.memory))