# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setup_env
import argparse
import os
import sys
import gradio as gr
from loguru import logger

from llm4crs.agent import CRSAgent
from llm4crs.agent_plan_first import CRSAgentPlanFirst
from llm4crs.agent_plan_first_openai import CRSAgentPlanFirstOpenAI
from llm4crs.buffer import CandidateBuffer
from llm4crs.corpus import BaseGallery
from llm4crs.critic import Critic
from llm4crs.environ_variables import *
from llm4crs.mapper import MapTool
from llm4crs.prompt import *
from llm4crs.ranking import RecModelTool, RankFeatureStoreTool, OpenAIRankingTool, RankRetrievalFeatureStoreTool, BasketTool
from llm4crs.retrieval import SimilarItemTool, SQLSearchTool, FetchFeatureStoreItemsTool, FetchAllTool
from llm4crs.query import QueryTool
from llm4crs.utils import FuncToolWrapper


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default="7891")
# parser.add_argument('--domain', type=str, default='game')
parser.add_argument(
    "--max_candidate_num",
    type=int,
    default=1000,
    help="Number of max candidate number of buffer",
)
parser.add_argument(
    "--similar_ratio",
    type=float,
    default=0.05,
    help="Ratio of returned similar items / total games",
)
parser.add_argument(
    "--rank_num", type=int, default=100, help="Number of games given by ranking tool"
)
parser.add_argument(
    "--max_output_tokens",
    type=int,
    default=1024,
    help="Max number of tokens in LLM output",
)
parser.add_argument(
    "--engine",
    type=str,
    default="gpt-4o",
    help="Engine of OpenAI API to use. The default is gpt-4o",
)
parser.add_argument(
    "--bot_type",
    type=str,
    default="chat",
    choices=["chat", "completion"],
    help="Type OpenAI models. The default is completion. Options [completion, chat]",
)

# chat history shortening
parser.add_argument(
    "--enable_shorten",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to enable shorten chat history with LLM",
)

# dynamic demonstrations
parser.add_argument(
    "--demo_mode",
    type=str,
    choices=["zero", "fixed", "dynamic"],
    default="dynamic",
    help="Directory path of demonstrations",
)
parser.add_argument(
    "--demo_dir_or_file",
    type=str,
    default="./demonstration/seed_demos_placeholder.jsonl",
    help="Directory or file path of demonstrations"
)
parser.add_argument(
    "--num_demos", type=int, default=5, help="number of demos for in-context learning"
)

# reflection mechanism
parser.add_argument(
    "--enable_reflection",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to enable reflection",
)
parser.add_argument(
    "--reflection_limits", type=int, default=3, help="limits of reflection times"
)

# plan first agent
parser.add_argument(
    "--plan_first",
    type=int,
    choices=[0, 1],
    default=1,
    help="Whether to use plan first agent",
)

# whether to use langchain in plan-first agent
parser.add_argument(
    "--langchain",
    type=int,
    choices=[0, 1],
    default=0,
    help="Whether to use langchain in plan-first agent",
)

# the reply style of the assistent
parser.add_argument(
    "--reply_style",
    type=str,
    choices=["concise", "detailed"],
    default="detailed",
    help="Reply style of the assistent. If detailed, details about the recommendation would be give. Otherwise, only item names would be given.",
)

# use feature store tools for hard filtering and ranking
parser.add_argument(
    "--feature_store_tools",
    action="store_true",
    help="Use feature store tools for hard filtering and ranking",
)

# use LLM ranker for ranking
parser.add_argument(
    "--LLM_ranker",
    action="store_true",
    help="Use LLM ranker for ranking",
)

# whether to update user profile or not
parser.add_argument(
    "--update_user_profile",
    action="store_true",
    help="Turns on user profile update",
)

# whether to load user profile from file
parser.add_argument(
    "--load_user_profile",
    type=str,
    default=None,
    help="Specify json file to load user profile from",
)

# whether to use recommendation API in the fetch all tool
parser.add_argument(
    "--use_reco_tool",
    action="store_true",
    help="Use recommendation API in the fetch all tool",
)

# affordability basket use-case
parser.add_argument(
    "--affordability_basket",
    action="store_true",
    help="Operate in affordability basket mode",
)

args = parser.parse_args()

# Define map that replaces {item} for domain specific item

domain = os.environ.get("DOMAIN", "groceries")
domain_map = {"item": domain, "Item": domain.capitalize(), "ITEM": domain.upper()}

default_chat_value = [
    None,
    "Hello, I'm a conversational {item} recommendation assistant. I'm here to help you discover your interested {item}s.".format(
        **domain_map
    ),
]

# Replace placeholders in tool names with domain specific item
tool_names = {k: v.format(**domain_map) for k, v in TOOL_NAMES.items()}

# Load the corpus (includes all items in the database)
item_corpus = BaseGallery(
    GAME_INFO_FILE,
    TABLE_COL_DESC_FILE,
    f"{domain}_information",
    columns=USE_COLS,
    fuzzy_cols=["title"] + CATEGORICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
)

# Initialize the candidate buffer
candidate_buffer = CandidateBuffer(item_corpus, num_limit=args.max_candidate_num)

# Define tools that are available to the LLM orchestrator. Which tools are available depends
# on the arguments passed to the script. Tool prompts are imported from prompt.py 
# The key of dict here is used to map to the prompt

# Pick which tools to use

map_tool = True
summarize = True
basket_tool = False
prompt_template = SYSTEM_PROMPT_PLAN_FIRST_RECO_ONLY

if args.feature_store_tools:
    hard_filter_tool = FetchFeatureStoreItemsTool(
        name=tool_names["HardFilterTool"],
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        desc=FEATURE_STORE_FILTER_TOOL_DESC.format(**domain_map),
        terms=SEARCH_TERMS_FILE,
    )
    ranking_tool = RankFeatureStoreTool(
        name=tool_names["RankingTool"],
        desc=FEATURE_STORE_RANK_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        fetch_tool=hard_filter_tool,
        terms=RANKING_SEARCH_TERMS_FILE,
    )
elif args.LLM_ranker:
    hard_filter_tool = FetchAllTool(
        name=tool_names["HardFilterTool"],
        desc=FEATURE_STORE_FILTER_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        terms_rec=SEARCH_TERMS_FILE,
        terms_rank=RANKING_SEARCH_TERMS_FILE,
        use_reco_tool=args.use_reco_tool,
    )
    ranking_tool = OpenAIRankingTool(
        name=tool_names["RankingTool"],
        desc=OPENAI_RANK_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        use="reco",
    )
    map_tool = False
    summarize = False

elif args.affordability_basket:
    hard_filter_tool = FetchAllTool(
        name=tool_names["HardFilterTool"],
        desc='Placeholder tool. Please do not use.',
        item_corpus=item_corpus,
        buffer=candidate_buffer,
    )
    ranking_tool = RankRetrievalFeatureStoreTool(
        name=tool_names["RankingTool"],
        desc=RANK_RETRIEVAL_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
    )
    map_tool = False
    summarize = False
    basket_tool = True
    prompt_template = SYSTEM_PROMPT_PLAN_FIRST_BASKET

else:
    hard_filter_tool = SQLSearchTool(
        name=tool_names["HardFilterTool"],
        desc=HARD_FILTER_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        max_candidates_num=args.max_candidate_num,
    )
    ranking_tool = RecModelTool(
        name=tool_names["RankingTool"],
        desc=RANKING_TOOL_DESC.format(**domain_map),
        model_fpath=MODEL_CKPT_FILE,
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        rec_num=args.rank_num,
    )

tools = {
    "BufferStoreTool": FuncToolWrapper(
        func=candidate_buffer.init_candidates,
        name=tool_names["BufferStoreTool"],
        desc=CANDIDATE_STORE_TOOL_DESC.format(**domain_map),
    ),
    "LookUpTool": QueryTool(
        name=tool_names["LookUpTool"],
        desc=LOOK_UP_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
    ),
    "HardFilterTool": hard_filter_tool,
    "SoftFilterTool": SimilarItemTool(
        name=tool_names["SoftFilterTool"],
        desc=SOFT_FILTER_TOOL_DESC.format(**domain_map),
        item_sim_path=ITEM_SIM_FILE,
        item_corpus=item_corpus,
        buffer=candidate_buffer,
        top_ratio=args.similar_ratio,
    ),
    "RankingTool": ranking_tool,
}

if map_tool:
    tools["MapTool"] = MapTool(
        name=tool_names["MapTool"],
        desc=MAP_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
    )

if basket_tool:
    tools["BasketTool"] = BasketTool(
        name=tool_names["BasketTool"],
        desc=BASKET_TOOL_DESC.format(**domain_map),
        item_corpus=item_corpus,
        buffer=candidate_buffer,
    )

# Reflection tool is only available for plan-first agent
if args.enable_reflection:
    critic = Critic(
        model="gpt-4" if "4" in args.engine else "gpt-3.5-turbo",
        engine=args.engine,
        buffer=candidate_buffer,
        domain=domain,
        bot_type=args.bot_type,
    )
else:
    critic = None

# Different agent types
if args.plan_first and args.langchain:
    AgentType = CRSAgentPlanFirst
elif args.plan_first and not args.langchain:
    AgentType = CRSAgentPlanFirstOpenAI
else:
    AgentType = CRSAgent

bot = AgentType(
    domain,
    tools,
    candidate_buffer,
    item_corpus,
    args.engine,
    args.bot_type,
    max_tokens=args.max_output_tokens,
    enable_shorten=args.enable_shorten,  # history shortening
    demo_mode=args.demo_mode,
    demo_dir_or_file=args.demo_dir_or_file,
    num_demos=args.num_demos,  # demonstration
    critic=critic,
    reflection_limits=args.reflection_limits,  # reflexion
    verbose=True,
    reply_style=args.reply_style,  # only supported for CRSAgentPlanFirstOpenAI
    user_profile_update=args.update_user_profile,
    planning_recording_file='plans/plan.json',  # planning recording file
    enable_summarize=summarize,
    prompt_template=prompt_template,
)

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level="DEBUG",
    format="<lc>[{time:YYYY-MM-DD HH:mm:ss} {level}] <b>{message}</b></lc>",
)

# Updates conversation history by adding user message. Messages have the convention
# [user_message, chatbot_reply]. When the user sends a message, the chatbot has not
# replied yet. The first message is [None,chatbot_welcome_message] to start the 
# conversation.

def user(user_message, history):
    return "", history + [[user_message, None]], history + [[user_message, None]]


# Chat conversation logic, where the user sends a message and the chatbot replies. Each time
# the chatbot receives a message, it runs a new cycle of one-step planning, tool calling and
# response generation (bot.run_gr function). The chatbot then returns the response to the user.

bot.init_agent()
bot.set_mode("accuracy")

# initialize user profile from file
if args.load_user_profile:
    bot.load_preferences(args.load_user_profile)

css = """
#chatbot .overflow-y-auto{height:600px}
#send {background-color: #FFE7CF}
"""
with gr.Blocks(css=css, elem_id="chatbot") as demo:
    with gr.Row(visible=True) as btn_raws:
        with gr.Column(scale=5):
            mode = gr.Radio(
                ["diversity", "accuracy"], value="accuracy", label="Recommendation Mode"
            )
        with gr.Column(scale=5):
            style = gr.Radio(
                ["concise", "detailed"], value=getattr(bot, 'reply_style', 'concise'), label="Reply Style"
            )
    chatbot = gr.Chatbot(
        elem_id="chatbot", label=f"{domain_map['Item']} RecAgent"
    )
    state = gr.State([])
    with gr.Row(visible=True) as input_raws:
        with gr.Column(scale=4):
            txt = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter", container=False
            )
        with gr.Column(scale=1, min_width=0):
            send = gr.Button(value="Send", elem_id="send", variant="primary")
        with gr.Column(scale=1, min_width=0):
            clear = gr.ClearButton(value="Clear")

    state.value = [default_chat_value]
    chatbot.value = [default_chat_value]

    txt.submit(user, [txt, state], [txt, state, chatbot]).then(
        bot.run_gr, [state], [chatbot, state]
    )
    txt.submit(candidate_buffer.clear, [], [])
    txt.submit(lambda: "", None, txt)

    send.click(user, [txt, state], [txt, state, chatbot]).then(
        bot.run_gr, [state], [chatbot, state]
    )
    send.click(candidate_buffer.clear, [], [])
    send.click(lambda: "", None, txt)
    mode.change(bot.set_mode, [mode], None)
    style.change(bot.set_style, [style], None)

    clear.click(bot.clear)
    clear.click(lambda: [default_chat_value], None, chatbot)
    clear.click(lambda: [default_chat_value], None, state)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
