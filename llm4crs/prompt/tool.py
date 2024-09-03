# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# need to be formatted: item, Item, ITEM


TOOL_NAMES = {
    "BufferStoreTool": "{Item} Candidates Storing Tool",
    "LookUpTool": "{Item} Information Look Up Tool",
    "HardFilterTool": "{Item} Properties Filtering Tool",
    "SoftFilterTool": "{Item} Similarity Filtering Tool",
    "RankingTool": "{Item} Candidates Ranking Tool",
    "BasketTool": "{Item} Basket Compilation Tool",
    "MapTool": "Mapping Tool",
}


CANDIDATE_STORE_TOOL_DESC = """
The tool is useful to save candidate {item} into buffer as the initial candidates, following tools would filter or ranking {item} from those candidates. \
For example, "Please select the most suitable {item} from these {item}". \
Don't use this tool when the user hasn't specified that they want to select from a specific set of {item}. \
The input of the tool should be a string of {item} names split by two ';', such as "{ITEM}1;; {ITEM}2;; {ITEM}3". 
"""


LOOK_UP_TOOL_DESC = """
The tool is used to look up {item}'s detailed information in a {item} information table (including statistical information), like number of {item}, description of {item}, price and so on. \

The input of the tools should be a SQL command (in one line) converted from the search query, which would be used to search information in {item} information table. \
You should try to select as less columns as you can to get the necessary information. \
Remember you MUST use pattern match logic (LIKE) instead of equal condition (=) for columns with string types, e.g. "title LIKE '%xxx%'". \

For example, if asking for "how many xxx {item}?", you should use "COUNT()" to get the correct number. If asking for "description of xxx", you should use "SELECT {item}_description FROM xxx WHERE xxx". \

The tool can NOT give recommendations. DO NOT SELECT id information!
"""


HARD_FILTER_TOOL_DESC = """
The tool is a hard-condition {item} filtering tool. The tool is useful when human want {item}s with some hard conditions on {item} properties. \
The input of the tool should be a one-line SQL SELECT command converted from hard conditions. Here are some rules: \
1. {item} titles can not be used as conditions in SQL;
2. always use pattern match logic for columns with string type;
3. only one {item} information table is allowed to appear in SQL command;
4. select all {item}s that meet the conditions, do not use the LIMIT keyword;
5. try to use OR instead of AND;
6. use given related values for categorical columns instead of human's description.
"""

FEATURE_STORE_FILTER_TOOL_DESC = """
The tool is a hard-condition {item} filtering tool. The tool is useful when the user wants {item} with some hard conditions on {item} properties. \
The input is a string query that is used to fetch the features of the {item} from the Instacart Feature Store. The query should NOT be a SQL command. \
You should use your logic to extract the user intent from the message. If the user does not specify details, the query should be simple, 
for example if the user says "I would like some grapes", the query should be "grapes". If the user asks for a specific attribute, for example if they \
say 'I am looking for gluten-free bread options', the query should be 'gluten-free bread'. If the user asks for an attribute that is not specific, \
for example 'I would like some healthy breakfast' you should interpret it based on the context, for example 'low-fat breakfast'.
"""
'''
You should make sure to extract the specific item the user wants, including any \
any primare attributes like 'organic', 'gluten-free', 'low-fat', etc. You should interpret which attributes are relevant for the item, for example 'healthy' 
could mean 'low-fat', 'organic' or 'cage-free', depending on the situation.
'''


SOFT_FILTER_TOOL_DESC = """
The tool is a soft condition {item} filtering tool to find similar {item}s for specific seed {item}s. \
Use this tool ONLY WHEN human explicitly want to find similar {item}s with seed {item}s. \
The tool can not recommend {item}s based on human's history. \
There is a similarity score threshold in the tool, only {item}s with similarity above the threshold would be kept. \
Besides, the tool could be used to calculate the similarity scores with seed {item}s for {item}s in candidate buffer for ranking tool to refine. \
The input of the tool should be a list of seed {item} titles/names, which should be a Python list of strings. \
Do not fake any {item} names.
"""


RANKING_TOOL_DESC = """
The tool is useful to refine {item}s order (for better experiences) or remove unwanted {item}s (when human tells the {item}s he does't want) in conversation. \
The input of the tool should be a json string, which may consist of three keys: "schema", "prefer" and "unwanted". \
"schema" represents ranking schema, optional choices: "popularity", "similarity" and "preference", indicating rank by {item} popularity, rank by similarity, rank by human preference ("prefer" {item}s). \
The "schema" depends on previous tool using and human preference. If "prefer" info here not empty, "preference" schema should be used. If similarity filtering tool is used before, prioritize using "similarity" except human want popular {item}s.
"prefer" represents {item} names that human has enjoyed or human has interacted with before (human's history), which should be an array of {item} titles. Keywords: "used to do", "I like", "prefer".
"unwanted" represents {item} names that human doesn't like or doesn't want to see in next conversations, which should be an array of {item} titles. Keywords: "don't like", "boring", "not interested in". 
"prefer" and "unwanted" {item}s should be extracted from human request and previous conversations. Only {item} names are allowed to appear in the input. \
The human's feedback for you recommendation in conversation history could be regard as "prefer" or "unwanted", like "I have tried those items you recommend" or "I don't like those".
Only when at least one of "prefer" and "unwanted" is not empty, the tool could be used. If no "prefer" info, {item}s would be ranked based on the popularity.\
Do not fake {item}s.
"""

FEATURE_STORE_RANK_TOOL_DESC = """
The tool is useful to filter {item} for recommendation and refine {item} order, i.e. rank them according to feature store information. The tools input should be 'null', and the tool \
should typically be used after the fetch tool.
"""


RANK_RETRIEVAL_TOOL_DESC = """
The tool performs both retrieval and ranking of {item} relevant to a query for recommendations, and stores the items in the candidate buffer. \
Therefore, there is no need to use a filtering tool before this tool. \
The input is a string query that is used to fetch the features of the {item} from the Instacart Feature Store. The query should NOT be a SQL command. \
If the tool is being used in combination with the basket tool, the tool should be called multiple times in a row with different queries \
to fetch and rank items from each category. \
"""

OPENAI_RANK_TOOL_DESC = """
Ranking tool that relies on the OpenAI API to generate recommendations. The tool's input should be 'null' and the tool should typically be used after the hard filtering tool. The tool can return a \
detailed explanation of the top recommendations, so no mapping tool is needed after that.
"""

MAP_TOOL_DESC = """
The tool is useful when you want to convert {item} id to {item} title before showing {item}s to human. \
The tool is able to get stored {item}s in the buffer and randomly select a specific number of {item}s from the buffer. \
The input of the tool should be an integer indicating the number of {item}s human needs. \
The default value is 5 if human doesn't give.
"""

BASKET_TOOL_DESC = """
The tool is useful to compile a basket of {item}s to recommend to the user. It has access to items options for each category \
stored in the candidate buffer, and it calls the OpenAI API to generate a list of {item} indexes to recommend, one item from each category. \
The input to the tool should be a Python dictionary that contains with two key-value pairs: \
one pair should have 'categories' as key and a list of items categories in string format as value (e.g. ['fruit', 'vegetable', 'meat']). \
The other pair should have 'budget' as key and an float as value, indicating the user's budget. \
"""


_TOOL_DESC = {
    "CANDIDATE_STORE_TOOL_DESC": CANDIDATE_STORE_TOOL_DESC,
    "LOOK_UP_TOOL_DESC": LOOK_UP_TOOL_DESC,
    "HARD_FILTER_TOOL_DESC": HARD_FILTER_TOOL_DESC,
    "FEATURE_STORE_FILTER_TOOL_DESC": FEATURE_STORE_FILTER_TOOL_DESC,
    "SOFT_FILTER_TOOL_DESC": SOFT_FILTER_TOOL_DESC,
    "RANKING_TOOL_DESC": RANKING_TOOL_DESC,
    "OPENAI_RANK_TOOL_DESC": OPENAI_RANK_TOOL_DESC,
    "FEATURE_STORE_RANK_TOOL_DESC": FEATURE_STORE_RANK_TOOL_DESC,
    "BASKET_TOOL_DESC": BASKET_TOOL_DESC,
    "MAP_TOOL_DESC": MAP_TOOL_DESC,
}


# It seems like this is not really used, so one can dynamically choose which ranking and retrieval tools to use
OVERALL_TOOL_DESC = """
There are several tools to use:
- {BufferStoreTool}: {CANDIDATE_STORE_TOOL_DESC}
- {LookUpTool}: {LOOK_UP_TOOL_DESC}
- {HardFilterTool}: {FEATURE_STORE_FILTER_TOOL_DESC}
- {SoftFilterTool}: {SOFT_FILTER_TOOL_DESC}
- {RankingTool}: {OPENAI_RANK_TOOL_DESC}
- {BasketTool}: {BASKET_TOOL_DESC}
- {MapTool}: {MAP_TOOL_DESC}
""".format(
    **TOOL_NAMES, **_TOOL_DESC
)

# """
# There are several tools to use:
# - {BufferStoreTool}: useful to store candidate {item} given by user into the buffer for further filtering and ranking. For example, user may say "I want the cheapest in the list" or "A and B, which is more suitable for me".
# - {LookUpTool}: used to look up some {item} information in a {item} information table (including statistical information), like number of {item}, description of {item} and so on.
# - {HardFilterTool}: useful when user expresses intentions about {item}s with some hard conditions on {item} properties. The input should be a SQL command. The tool would return {item} ids.
# - {SoftFilterTool}: useful when user gives specific {item} names (seed {item}s) and is looking for similar {item}s. The input could only be {item} titles. The tool would return {item} ids.
# - {RankingTool}: useful to refine {item}s order based on user profile (including user prefer {item}s, like play history or liked {item}s and unwanted {item}s, like disliked {item}s) expressed in conversation. The input is a json str with 3 keys: "schema", "prefer" and "unwanted". If the two informations are both empty, the tool should not be used.The tool would return {item} ids.
# - {MapTool}: useful to convert {item} id to {item} titles.
# """


TOOLBOX_DESC = """
This tool is a big tool box consisting of all tools metioned above. The tool box is used to execute tools when a plan is made. 
The input is a List of Dict, indicating the tool using plan and input to each tool. There are two keys "tool_name" and "input" in Dict. 
The format should be like: "[{'tool_name': TOOL-1, 'input': INPUT-1}, ..., {'tool_name': TOOL-N, 'input': INPUT-N} ]".
"""


LLM_RANKER_PROMPT_TEMPLATE = """You are an expert grocery item recommender. Your job is to \
recommend items to shoppers based on their query and candidate item information. Please try to \
understand user intent from the query and recommend items that best match what the shopper \
wants based on the provided item information. You should always first return a python list of {rec_num} \
integer item indexes in the order of recommendation. If there are more than {rec_num} items, return the \
top {rec_num} recommendations. If there are less than {rec_num} items, return all the items in the order \
of recomendation. For example, if there are two recommendations with indexes 481290 and 124259, \
you should return [481290, 124259]. Always return {rec_num} items, unless there are less than {rec_num} \
items available; in that case include all items in recommendation order. \n \
After the list, you should also provide a justification for the top {explain_top} picks, along with a relevance \
score for them in regards to the user query, ranging from 0 for not relevant to 1 for highly relevant. \n \n \
To help you decide, there is additional information about the relevance of each candidate items to the query, \
provided by an expert system. These are: \n {{qc_info}} \n \
{{user_info}} \n \
Now please make your recommendations, given the user query and candidate item information below. \n \
User query: {{query}} \n Candidate items: \n \n {{reco_info}} \n \

Do not fake any item indexes. Do not add placeholder indexes like 1,2,3 if there are less than {rec_num} items available. \
Do not discard items that don't seem relevant. Go! \
"""

ITEM_QUERY_ATTRIBUTE_EXPLANATIONS = """
Global and retailer click-through rate: Probability that the item was clicked (chosen/converted) in searches it appeared, for all retailers \
and the specific retailer respectively. \n \
Relevance score: A score indicating how relevant the item is to the user query. Ranges from 0 to 1. Please pay attention to this. \n \
Exact match, Strong substitute, Weak substitute, Close complement, Remote complement: Flags that describe the relationship between the item \
and the user query in more detail. Range from 0 to 1. For example, an exact match would have an exact match value of 1. \n \
"""

"""
EXTRA PROMPTS
4. If it's impossible to meet the budget when picking the cheapest items, you should drop the least \
crucial item categories until the budget is met. \n \

If you have dropped item categories, please only report the final list and \
description of items, and mention the categories that were dropped. \n \
"""

BASKET_COMPILATION_PROMPT = """
You are an expert grocery item recommender whose job is to compile a basket of items to recommend to the user. \
You have to choose one item from each category, so that the total cost is below a user-defined budget. The items \
for each category are provided by an expert ranking system, so the items higher up each list are preferred. When \
making recommendations, you should consider the following limitations, in order of preference: \n \
1. The total of the recommended items should be below the budget. \n \
2. We are a retailer, so our goal is to maximize profit. You should make sure to be as close to the budget as \
possible, while still below it. This often means preferring more expensive items which might be lower in the \
ranking, to come closer to the budget. \n \
3. If the above conditions are met, you should prioritize items with higher ranking. \n \

You should always first return a python list of integer item indexes of the items you picked. \
For example, if there are two recommendations with indexes 481290 and 124259, \
you should return [481290, 124259]. Do not return anything before the list. \n
After the list, you should provide more information about the items picked, including their name, \
category, price, index and position in their category (1st, 2nd etc). \n \

Here are the items categories: {categories} \n
Here is the budget: ${budget} \n

Here are the items: \n
{item_info}

Please try to use all your budget. DO NOT report the total cost of recommended items, or statements \
about whether the budget is met or not. Now make your recommendations. Go! \
"""