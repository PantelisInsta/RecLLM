# Ranking tool using OpenAI API

from loguru import logger

TOOL_PROMPT_TEMPLATE = """You are an expert grocery item recommender. Your job is to \
recommend items to shoppers based on their query and candidate item information. Please try to \
understand user intent from the query and recommend items that best match what the shopper \
wants based on the provided item information. You should always first return a python list of integer item IDs in the \
order of recommendation. For example, if there are two recommendations with ids 481290 and 124259, \
you should return [481290, 124259]. If there are more than {rec_num} items, return the top {rec_num} \
recommendations. After the list, you should also provide a justification for the top {explain_top} picks. \
User query: {{query}} \n Candidate items: {{reco_info}} \n
"""

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

        prompt_template = TOOL_PROMPT_TEMPLATE

        # replace known quantities in prompt
        self.prompt = prompt_template.format(rec_num=self.rec_num, explain_top=self.explain_top)


    def get_item_info(self, item_ids):
        """
        Given item IDs, get item information from the item corpus. Then structure the information into a string
        that is appropriate to be fed to the OpenAI API for recommendation.
        """
        # get item information
        info = self.item_corpus.convert_id_2_info(item_ids, self.col_names)

        # organize item information into a string
        reco_info = []
        for i in range(len(item_ids)):
            # initialize string with index
            s = f"Item {i+1}: "
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
        prompt = self.prompt.format(query=query,reco_info=reco_str)

        # call OpenAI API to get recommendations
        output = agent.call(prompt,max_tokens=1000)

        # split output in list of item IDs and detailed explanations
        item_ids, explanations = self.split_output(output)

        if self.use == 'reco':
            return explanations
        elif self.use == 'eval':
            return item_ids
        else:
            return "Invalid use case."