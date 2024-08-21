# Tool that combines items from candidate buffer and calls API to return final basket.

from loguru import logger
import time
import ast

from llm4crs.prompt import BASKET_COMPILATION_PROMPT

class BasketTool:
    """
    Defines a tool that combines items from candidate buffer and calls API to return final basket
    that meets budget constraints.
    
    Args:
        name (str): The name of the tool.
        desc (str): The description of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        agent (OpenAI): The OpenAI agent to call.
        prompt (str): The prompt to use for the API call.
        col_names (list): The names of the columns to include in the item information.
    """
    
    def __init__(self, name, desc, item_corpus, buffer, col_names=None):

        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.item_idx = []

        if col_names is None:
            col_names = ['title','index','price']
        self.col_names = col_names

    def get_item_info(self, item_ids):
        """
        Given item IDs, get item information from the item corpus. Then structure the information into a string
        that is appropriate to be fed to the OpenAI API for recommendation. Also save the indexes of the items
        for validation purposes.
        """
        # get item information
        info = self.item_corpus.convert_id_2_info(item_ids, self.col_names)

        # save item indexes in list
        self.item_idx.extend(info['index'])
        # organize item information into a string
        reco_info = []
        for i, id in enumerate(item_ids):
            # initialize string with index
            s = f"Item: "
            for k in info.keys():
                # include key and value in the string
                s += f"{k}: {info[k][i]}\n"
            reco_info.append(s)

        # join and add new line between each item information
        reco_str = '\n'.join(reco_info)

        return reco_str
    
    
    def split_output(self, output):
        """
        Split the output from the OpenAI API into a list of item IDs and detailed explanations.
        """
        # find first occurernce of '[' and ']'
        start = output.find('[')
        end = output.find(']')
        # extract item IDs
        item_ids = output[start:end+1]
        # extract explanations
        explanations = output[end+1:]
        
        return item_ids, explanations
    
    def check_id_validity(self, item_idx):
        """
        Check if the instacart product IDs returned by the OpenAI API are valid.
        """
        # convert string to list
        item_idx = item_idx.replace('[','').replace(']','').split(',')
        # convert to integers
        item_idx = [int(i) for i in item_idx]
        # check if all IDs are valid
        valid = all([i in self.item_idx for i in item_idx])

        # only keep valid IDs
        item_idx = [i for i in item_idx if i in self.item_idx]
        
        return valid, str(item_idx)
    
    
    def run(self, inputs):

        # categories of items
        categories = inputs['categories']
        # budget
        budget = inputs['budget']
        # agent
        agent = inputs['agent']

        # get items from candidate buffer
        items = self.buffer.basket_items

        item_info = ""
        # assimilate item information
        for category in categories:
            item_info += f"Items for category '{category}':\n"
            # get items in category
            item_ids = items[category]
            # get item information
            item_info += self.get_item_info(item_ids)
        
        # add item information to prompt
        prompt = BASKET_COMPILATION_PROMPT.format(categories=categories,budget=budget,
                                                  item_info=item_info)

        # call OpenAI API to get recommendations
        start = time.time()
        output = agent.call(prompt,max_tokens=1000)
        end = time.time()
        logger.debug(f"LLM basket call latency: {end - start:.2f} s.")

        # split output in list of item IDs and detailed explanations
        item_idx, explanations = self.split_output(output)

        # check if item IDs are valid
        valid, item_idx = self.check_id_validity(item_idx)
        if not valid:
            # log error
            logger.error("Warning: LLM ranker returned invalid item IDs.")

        # TODO check programmatically if budget constraint is met

        return explanations