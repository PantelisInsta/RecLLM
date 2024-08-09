# Test OpenAI ranking tool

# Set up the environment
import setup_env

# import relevant modules
from llm4crs.retrieval.fetch_tool import FetchFeatureStoreItemsTool
from llm4crs.ranking.OpenAI_rank_tool import OpenAIRankingTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.corpus import BaseGallery
from llm4crs.utils import OpenAICall
import os

# Set up environment variables that define data paths
from llm4crs.environ_variables import *

# Set up the mode
mode = 'reco'

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

# Initialize the OpenAIRanking tool
rank_tool = OpenAIRankingTool('OpenAIRankingTool',
                                'Rank items using OpenAI API',
                                item_corpus, candidate_buffer, mode=mode)

# Initialize the OpenAI API
agent = OpenAICall(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_type=os.environ.get("OPENAI_API_TYPE", None),
            api_base=os.environ.get("OPENAI_API_BASE", None),
            api_version=os.environ.get("OPENAI_API_VERSION", None),
            temperature=0,
            model_type="chat",
            timeout=60,
            stop_words=None,
        )

# Fetch items from the feature store
term = "milk"

# run the fetch tool
fetch_tool.run(term)

# Run the OpenAI ranking tool
inputs = {"input": term, "agent": agent}
output = rank_tool.run(inputs)

print(output)