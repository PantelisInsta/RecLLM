# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# setup
import setup_env

import os
import random
import re
import sys
import math
import json
import time
import openai
import argparse
import threading  
from typing import *
import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz
from datetime import datetime

from llm4crs.prompt import *
from llm4crs.utils import FuncToolWrapper
from llm4crs.corpus import BaseGallery
from llm4crs.agent import CRSAgent
from llm4crs.agent_plan_first import CRSAgentPlanFirst
from llm4crs.agent_plan_first_openai import CRSAgentPlanFirstOpenAI
from llm4crs.environ_variables import *
from llm4crs.critic import Critic
from llm4crs.mapper import MapTool
from llm4crs.query import QueryTool
from llm4crs.ranking import RecModelTool, RankFeatureStoreTool, OpenAIRankingTool
from llm4crs.buffer import CandidateBuffer
from llm4crs.retrieval import SQLSearchTool, SimilarItemTool, FetchFeatureStoreItemsTool



def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


def write_jsonl(obj, fpath: str):
    with open(fpath, 'w') as f:
        for entry in obj:
            json.dump(entry, f)
            f.write('\n')


  
class TimeoutError(Exception):  
    """Custom timeout error exception class"""  
    pass  
  
class TimeoutRunner:  
    def __init__(self, func, timeout_seconds):  
        self.func = func  # Function to be executed  
        self.timeout_seconds = timeout_seconds  # Timeout duration in seconds  
        self.result = None  # Placeholder for the function result  
        self.has_result = False  # Flag to indicate whether the function has completed  
        self.timer = None  # Placeholder for the timer object  
  
    def _run_func(self, *args, **kwargs):  
        """Execute the function and store the result."""  
        self.result = self.func(*args, **kwargs)  
        self.has_result = True  
  
    def _on_timeout(self):  
        """Raise a TimeoutError if the function has not completed."""  
        if not self.has_result:  
            raise TimeoutError("Function execution took too long")  
  
    def run(self, *args, **kwargs):  
        """Execute the function with a timeout.  
  
        If the function's execution time exceeds the specified timeout duration,  
        a TimeoutError exception is raised. If the function completes within the  
        timeout duration, the function's return value is returned.  
        """  
        # Create and start the timer  
        self.timer = threading.Timer(self.timeout_seconds, self._on_timeout)  
        self.timer.start()  
  
        # Execute the function in a separate thread  
        worker = threading.Thread(target=self._run_func, args=args, kwargs=kwargs)  
        worker.daemon = True
        worker.start()  
        worker.join(self.timeout_seconds)  # Wait for the function to complete or timeout  
  
        # Cancel the timer if the function has completed  
        self.timer.cancel()  
  
        # Raise a TimeoutError if the function has not completed  
        if not self.has_result:  
            raise TimeoutError("Function execution took too long")  
  
        # Return the function's result if it has completed  
        return self.result  



class Conversation:

    def __init__(self, user_prefix='User', agent_prefix='Assistent'):
        self.user_prefix = user_prefix
        self.agent_prefix = agent_prefix
        self.all_history = []
        self.history = []

    def add_user_msg(self, msg) -> None:
        self.history.append({'role': self.user_prefix, 'msg': msg})
        return
    
    def add_agent_msg(self, msg) -> None:
        self.history.append({'role': self.agent_prefix, 'msg': msg})
        return
        
    @property
    def total_history(self) -> str:
        res = ""
        for h in self.history:
            res += "{}: {}\n".format(h['role'], h['msg'])
        res = res[:-1]
        return res

    @property
    def turns(self) -> int:
        return math.ceil(len(self.history) / 2)

    def __len__(self) -> int:
        return len(self.history)

    def clear(self) -> None:
        if len(self.history) > 0:
            self.all_history.append({'id': len(self.all_history), 'conversation': self.history})
        self.history = []

    def dump(self, fpath: str):
        with open(fpath, 'w') as f:
            for entry in self.all_history:
                json.dump(entry, f)
                f.write('\n')


class RecBotWrapper:
    """
    Wrapper for RecAI, used for single-turn ranking.
    """
    def __init__(self, bot: CRSAgent, num_rec):
        self.bot = bot
        self.num_rec = num_rec

    def run(self, question: str):
        response = self.bot.run({"input": question + " The output must be a json string, where the key is the item and value is the rank. All candidates should be output."})
        return response

    def clear(self):
        self.bot.clear()


def hit_judge(msg: str, target: str, thres: float=80):
    """
    Compares agent recommendation to the target item using fuzzy matching,
    and determines whether it was a hit.
    """
    msg = re.sub(r"[^a-zA-Z0-9\s]", "", msg.lower())
    target = re.sub(r"[^a-zA-Z0-9\s]", "", target.lower())
    if fuzz.partial_ratio(msg, target) > thres:
        return True
    else:
        return False
    

def get_rank(pred: str, target: str):
    """
    Extracts a json string from the prediction and returns the rank of the target item.
    """
    pattern = r"\{.*?\}"
    json_str = re.findall(pattern, pred, re.DOTALL)
    if json_str:
        pred = json_str[0]
        try:
            pred = json.loads(pred)
            pred = {k.strip(): v for k, v in pred.items()}
        except:
            return -1
    else:
        return -1
    
    if target not in pred:
        for k, v in pred.items():
            k = str(k)
            if hit_judge(k, target, 90):
                try:
                    rank = int(v)
                    return rank
                except:
                    rank = -1
        return -1

    try:
        rank = int(pred[target])
    except:
        return -1
    
    return rank



def one_turn_conversation_eval(data: List[Dict], agent: RecBotWrapper, k: int):
    """
    Iterates through the data and evaluates the agent's performance in one-turn
    ranking conversations. Returns the NDCG@k metric and the conversation history.
    """
    conversation = []
    ndcg = []
    mr = []
    for i, d in enumerate(tqdm(data)):
        if hasattr(agent, "clear"):
            agent.clear()
        # candidates = neg_item_sample(corpus, d['context'], d['target'], k-1)
        agent_msg = agent.run(d['question'])
        if os.environ.get("DEBUG", '0') == '1':
            print(d['question'])
            print(agent_msg)
        rank = get_rank(agent_msg, d['target'])
        # rank = rank if rank > 0 else k
        if rank > 0:
            ndcg.append(1/math.log2(rank + 1))
        else:
            ndcg.append(0)

        tqdm.write(f"Sample {i}: Rank={rank}, NDCG@{k}={sum(ndcg)}/{len(ndcg)}={(sum(ndcg)/len(ndcg)):.4f}")
        conversation.append({'context': d['question'], 'target': d['target'], 'answer': agent_msg})
        
    # hit_turn = np.array(hit_turn)
    final_ndcg = sum(ndcg) / len(ndcg)
    return {f"NDCG@{k}": final_ndcg}, conversation
    
    
def main():

    # root dir directory
    ROOT_DIR = os.getcwd()
    
    parser = argparse.ArgumentParser("Evaluator")
    parser.add_argument("--data", type=str, default="data/ranking_data.jsonl")
    parser.add_argument("--max_turns", type=int, default=5, help="max turns limit for evaluation")
    parser.add_argument("--save", type=str, help='path to save conversation text')
    parser.add_argument('--engine', type=str, default='gpt-4',
                    help='Engine of OpenAI API to use as user simulator. The default is gpt-4') 
    parser.add_argument("--timeout", type=int, default=5, help="Timeout threshold when calling OAI. (seconds)")
    parser.add_argument("--k", type=int, default=20, help="NDCG cutoff")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")

    # parser.add_argument('--domain', type=str, default='game')
    parser.add_argument('--agent', default='recbot',type=str, help='agent type, "recbot" is our method and others are baselines')
    parser.add_argument('--num_rec', type=int, default=5,  help='number of items to be recommended')

    # recbot-agent
    parser.add_argument('--max_candidate_num', type=int, default=1000, help="Number of max candidate number of buffer")
    parser.add_argument('--similar_ratio', type=float, default=0.1, help="Ratio of returned similar items / total games")
    parser.add_argument('--rank_num', type=int, default=100, help="Number of games given by ranking tool")
    parser.add_argument('--max_output_tokens', type=int, default=512, help="Max number of tokens in LLM output")
    parser.add_argument('--bot_type', type=str, default='chat', choices=['chat', 'completion'],
                    help='Type OpenAI models. The default is chat. Options [completion, chat]') 

    # chat history shortening
    parser.add_argument('--enable_shorten', type=int, choices=[0,1], default=0, help="Whether to enable shorten chat history with LLM")

    # dynamic demonstrations
    parser.add_argument('--demo_mode', type=str, choices=["zero", "fixed", "dynamic"], default="zero", help="Directory path of demonstrations")
    parser.add_argument('--demo_dir_or_file', type=str, help="Directory or file path of demonstrations")
    parser.add_argument('--num_demos', type=int, default=3, help="number of demos for in-context learning")

    # reflection mechanism
    parser.add_argument('--enable_reflection', type=int, choices=[0,1], default=0, help="Whether to enable reflection")
    parser.add_argument('--reflection_limits', type=int, default=3, help="limits of reflection times")

    # plan first agent
    parser.add_argument('--plan_first', type=int, choices=[0,1], default=1, help="Whether to use plan first agent")
    parser.add_argument("--langchain", type=int, choices=[0, 1], default=0, help="Whether to use langchain in plan-first agent")

    args, _ = parser.parse_known_args()

    random.seed(args.seed)

    domain = os.environ.get("DOMAIN", 'groceries')

    domain_map = {'item': domain, 'Item': domain.capitalize(), 'ITEM': domain.upper()}

    # Combine root dir with data path
    args.data = os.path.join(ROOT_DIR, args.data) 
    print(args.data)
    eval_data = read_jsonl(args.data)[:10]

    conversation = Conversation()


    if args.agent == 'recbot':

        item_corpus = BaseGallery(GAME_INFO_FILE, TABLE_COL_DESC_FILE, f'{domain}_information',
                                    columns=USE_COLS, 
                                    fuzzy_cols=['title'] + CATEGORICAL_COLS, 
                                    categorical_cols=CATEGORICAL_COLS)
        tool_names = {k: v.format(**domain_map) for k,v in TOOL_NAMES.items()}

        candidate_buffer = CandidateBuffer(item_corpus, num_limit=args.max_candidate_num)

        # The key of dict here is used to map to the prompt
        tools = {
                "BufferStoreTool": FuncToolWrapper(func=candidate_buffer.init_candidates, name=tool_names['BufferStoreTool'], 
                                                desc=CANDIDATE_STORE_TOOL_DESC.format(**domain_map)),
                "LookUpTool": QueryTool(name=tool_names['LookUpTool'], desc=LOOK_UP_TOOL_DESC.format(**domain_map), item_corpus=item_corpus, buffer=candidate_buffer),
                "HardFilterTool": FetchFeatureStoreItemsTool(name=tool_names['HardFilterTool'], desc=FEATURE_STORE_FILTER_TOOL_DESC.format(**domain_map), item_corpus=item_corpus, 
                                                buffer=candidate_buffer, terms=SEARCH_TERMS_FILE),
                "SoftFilterTool": SimilarItemTool(name=tool_names['SoftFilterTool'], desc=SOFT_FILTER_TOOL_DESC.format(**domain_map), item_sim_path=ITEM_SIM_FILE, 
                                                item_corpus=item_corpus, buffer=candidate_buffer, top_ratio=args.similar_ratio),
                "RankingTool": OpenAIRankingTool(name=tool_names["RankingTool"], desc=OPENAI_RANK_TOOL_DESC.format(**domain_map),
                                                 item_corpus=item_corpus, buffer=candidate_buffer, use="eval"),
        }

        if args.enable_reflection:
            critic = Critic(
                model = 'gpt-4' if "4" in os.environ.get("AGENT_ENGINE", "") else 'gpt-3.5-turbo',
                engine = os.environ.get("AGENT_ENGINE", ""),
                buffer = candidate_buffer,
                domain = domain
            )
        else:
            critic = None


        if args.plan_first:
            if args.langchain:
                AgentType = CRSAgentPlanFirst
            else:
                AgentType = CRSAgentPlanFirstOpenAI
        else:
            AgentType = CRSAgent


        bot = AgentType(domain, tools, candidate_buffer, item_corpus, args.engine, 
                    args.bot_type, max_tokens=args.max_output_tokens, 
                    enable_shorten=args.enable_shorten,  # history shortening
                    demo_mode=args.demo_mode, demo_dir_or_file=args.demo_dir_or_file, num_demos=args.num_demos,    # demonstration
                    critic=critic, reflection_limits=args.reflection_limits, enable_summarize=False)   # reflexion
        
        bot.init_agent()
        bot = RecBotWrapper(bot, args.num_rec)

    else:
        raise ValueError("Not support for such agent.")


    metrics, conversation = one_turn_conversation_eval(eval_data, bot, k=args.k)
    
    # save conversation
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.save is not None:
        save_path = args.save
    else:
        save_path = os.path.join(os.path.dirname(args.data), f"saved_conversations_{args.agent}_{current_time}_{os.path.basename(args.data)}")
    write_jsonl(conversation, save_path)
    print("Conversation history saved in {}.".format(save_path))

    print(metrics)

if __name__ == "__main__":
    main()