# Test basket compilation tool

# Set up the environment
import setup_env

# import relevant modules
from llm4crs.ranking.basket_compilation_tool import BasketTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.corpus import BaseGallery
from llm4crs.utils import OpenAICall
import os

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

# Initialize the basket compilation tool
basket_tool = BasketTool('BasketTool',
                         'Combine items from candidate buffer and call API to return final basket',
                         item_corpus, candidate_buffer)

# Initialize the OpenAI API
agent = OpenAICall(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_type=os.environ.get("OPENAI_API_TYPE", None),
            api_base=os.environ.get("OPENAI_API_BASE", None),
            temperature=0,
            model_type="chat",
            timeout=60,
            stop_words=None,
        )

# put items in basket
candidate_buffer.basket_items = {'milk': [2640, 4416, 5646, 2557, 5577],
                                 'cereal': [1078, 253, 336, 2705, 1138],
                                 'orange juice': [5369, 877, 63, 2282, 600],
                                 'bacon': [3039, 1584, 3424, 2187, 2774],
                                 'eggs': [1195, 4278, 296, 5657, 5501]}

# set up inputs to basket tool
inputs = {'agent': agent,
          'categories': ['milk', 'cereal', 'orange juice', 'bacon', 'eggs'],
          'budget': 20,
          }

# run the basket tool
output = basket_tool.run(inputs)

# print the output
print(output)