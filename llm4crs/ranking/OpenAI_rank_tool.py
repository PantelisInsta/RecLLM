# Ranking tool using OpenAI API

from loguru import logger
import time
from llm4crs.prompt import LLM_RANKER_PROMPT_TEMPLATE, ITEM_QUERY_ATTRIBUTE_EXPLANATIONS


class OpenAIRankingTool:
    """
    Ranking tool that relies on the OpenAI API to generate recommendations. It receives the user query, and
    looks for items in the candidate bus that have attributes that best match the query.
    It first compiles a string of candidate items with all item informations, sends it to the OpenAI API,
    and returns a list of IDs of the top N recommendations ordered or the top M recommendations with explanations.

    Args:
        name (str): The name of the tool.
        desc (str): The description of the tool.
        item_corpus (BaseGallery): The corpus of items.
        buffer (CandidateBuffer): The candidate bus to store candidates.
        use (str): The usage of the tool. Either 'reco' for recommendation or 'eval' for evaluation.
        col_names (list): The names of the columns to include in the item information.
        rec_num (int): The number of recommendations to return.
        explain_top (int): The number of top recommendations to provide explanations for.
        api_key (str): The API key for the OpenAI API.
    """
    def __init__(self, name, desc, item_corpus, buffer, use='reco', col_names = None,
                  rec_num=20, explain_top=5, api_key=None):
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.use = use
        self.rec_num = rec_num
        self.explain_top = explain_top
        
        if col_names is None:
            col_names = ['title','index','category','attributes','description',
                         'details','price', 'popularity']
        
        self.col_names = col_names

        prompt_template = LLM_RANKER_PROMPT_TEMPLATE

        # replace known quantities in prompt
        self.prompt = prompt_template.format(rec_num=self.rec_num, explain_top=self.explain_top)


    def get_item_info(self, item_ids):
        """
        Given item IDs, get item information from the item corpus. Then structure the information into a string
        that is appropriate to be fed to the OpenAI API for recommendation. Also save the indexes of the items
        for validation purposes.
        """
        # get item information
        info = self.item_corpus.convert_id_2_info(item_ids, self.col_names)

        # save item indexes in list
        self.item_idx = info['index']
        # organize item information into a string
        reco_info = []
        for i, id in enumerate(item_ids):
            # initialize string with index
            s = f"Item: "
            for k in info.keys():
                # include key and value in the string
                s += f"{k}: {info[k][i]}\n"
            # if query-item information is available, include it
            try:
                qc_info = self.buffer.query_candidate_info[id]
                s += f"Query-item information: \n"
                for k in qc_info.keys():
                    s += f"{k}: {qc_info[k]}\n"
            except:
                pass
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
        
        return valid

    
    def run(self, inputs):

        # user query
        query = inputs["prompt"]
        # OpenAI agent
        agent = inputs["agent"]
        
        # get candidate items from the buffer
        item_ids = self.buffer.get()
        if len(item_ids) <= 0:
            return "There is no suitable items."
        
        # get item information
        reco_str = self.get_item_info(item_ids)

        # compile prompt without explicit calling
        prompt = self.prompt.format(query=query,reco_info=reco_str,qc_info=ITEM_QUERY_ATTRIBUTE_EXPLANATIONS)

        # call OpenAI API to get recommendations
        start = time.time()
        output = agent.call(prompt,max_tokens=1000)
        end = time.time()
        logger.debug(f"LLM ranker call latency: {end - start:.2f} s.")

        # split output in list of item IDs and detailed explanations
        item_idx, explanations = self.split_output(output)

        # check if item IDs are valid
        valid = self.check_id_validity(item_idx)
        if not valid:
            # log error
            logger.error("Warning: LLM ranker returned invalid item IDs.")

        if self.use == 'reco':
            return explanations
        elif self.use == 'eval':
            return item_idx
        else:
            return "Invalid use case."