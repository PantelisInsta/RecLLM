# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import re
import time
from copy import deepcopy
from ast import literal_eval
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple
from langchain.agents import Tool
from loguru import logger

from llm4crs.critic import Critic
from llm4crs.demo.base import DemoSelector
from llm4crs.prompt import SYSTEM_PROMPT_PLAN_FIRST, TOOLBOX_DESC, OVERALL_TOOL_DESC
from llm4crs.utils import OpenAICall, num_tokens_from_string, format_prompt
from llm4crs.utils.open_ai import get_openai_tokens
from llm4crs.memory.memory import UserProfileMemory


class ToolBox:
    '''
    Sets up and executes the tools in the toolbox. It also checks if the plan is correct
    '''
    def __init__(self, name: str, desc: str, tools: Dict[str, Callable[..., Any]]):
        self.name = name
        self.desc = desc
        self.tools = {}
        for k, v in tools.items():
            if "Map" not in k:
                self.tools[" ".join(k.split(" ")[1:])] = v
            else:
                self.tools[k] = v
        # Separate map and look-up tools
        for t_name in self.tools.keys():
            t_name_lower = t_name.lower()
            if "look up" in t_name_lower:
                self.look_up_tool_name = t_name
            if "map" in t_name_lower:
                self.map_tool_name = t_name

        self.failed_times = 0

    def _check_plan_legacy(self, plans: Dict) -> bool:
        """Check whether map tool is used correctly. If yes, return True.

        There are two cases:
        1. query tool and retrieval/ranking tool are used, but map tool is missed.
        2. retrieval/ranking tools are used, but map tool is missed.

        Args:
            plans(Dict): plans generated by LLM
        Returns:
            bool: whether map tool is used correctly. If True, the plan is legacy
        """
        query_tool_exist = any(["look up" in k.lower() for k in plans.keys()])
        map_tool_exist = any(["map" in k.lower() for k in plans.keys()])
        if not map_tool_exist:
            if (len(plans) > 1) and (query_tool_exist):  # case 1:
                return False
            if (len(plans) > 0) and (not query_tool_exist):  # case 2:
                return False
        return True

    def run(self, inputs: str, user_profile: UserProfileMemory=None):
        # the inputs should be a json string generated by LLM, containing the
        # tool execution plan. key is the tool name, value is the input string
        # of the tool

        # load user profile information
        if user_profile is not None:
            profile: dict = user_profile.get()
            profile["prefer"] = set(profile['history']+ profile['like'])
            profile["unwanted"] = set(profile["unwanted"])
            del profile["history"], profile["like"]
        else:
            profile = None
        success = True

        # parse json string of plans
        try:
            plans: list = json.loads(inputs)
            plans = {t["tool_name"]: t["input"] for t in plans}
        except Exception as e:
            try:
                plans = literal_eval(inputs)
                plans = {t["tool_name"]: t["input"] for t in plans}
            except Exception as e:
                success = False
                return (
                    success,
                    f"""An exception happens: {e}. The inputs should be a json string for tool using plan. The format should be like: "[{{'tool_name': TOOL-1, 'input': INPUT-1}}, ..., {{'tool_name': TOOL-N, 'input': INPUT-N}} ]".""",
                )

        # check if all tool names existing
        _plans = deepcopy(plans)
        plans = {}
        for k, v in _plans.items():
            if "Map" not in k:
                plans[k] = v
            else:
                plans[k] = v
        tool_not_exist = [k for k in plans.keys() if not any([x in k for x in self.tools.keys()])]
        if len(tool_not_exist) > 0:
            success = False
            return (
                success,
                f"These tools do not exist: {', '.join(tool_not_exist)}. Optional Tools: {', '.join(list(self.tools.keys()))}.",
            )

        if not self._check_plan_legacy(plans):  # map tool is missed
            # add map tool manually
            plans[self.map_tool_name] = "5"

        # Main code: iterate through the planned tools and execute them

        res = ""  # to store results
        for k, v in plans.items():
            try:
                if not isinstance(v, str):
                    v = json.dumps(v)
                # If the tool is ranking tool, merge the user profile information
                # This is where the user profile is used to update the ranking tool
                if "ranking" in k.lower():
                    # get ranking tool name as a string
                    ranking_tool_string = str(self.tools['Candidates Ranking Tool'])
                    # make sure profile info exists and ranking tool accepts it
                    if profile and 'reco' in ranking_tool_string:
                        try:
                            inputs = json.loads(v)
                            print(profile.items())
                            for _k, _v in profile.items():
                                _v.update(inputs.get(_k, []))
                                inputs[_k] = list(_v)
                        except json.decoder.JSONDecodeError as e:
                            inputs["schema"] = "preference"
                            inputs = {_k: list(_v) for _k,_v in profile.items()}
                        v = json.dumps(inputs)
                        print(v)
                # Find which tool matches the current plan step
                for x in self.tools.keys():
                    if x in k:
                        align_tool = x
                        break
                # Execute the tool
                output = self.tools[align_tool].run(v)
                # Store the output if it is a look up tool or map tool
                if ("look up" in k.lower()) or ("map" in k.lower()):
                    res += output
            except Exception as e:
                logger.debug(f"Error: {e}")
                self.failed_times += 1
                success = False
                return (
                    success,
                    f"The input to tool {k} does not meet the format specification.",
                )

        if len(res) == 0:
            self.failed_times += 1
            res = (
                "No output because tool plan is not correct. Only {lookup} and {map} would give response. "
                "If look up information, use {lookup}. If recommendation, use {map} in the final step. Also, remember to use ranking tool before map."
            )
            success = False
            res = res.format(lookup=self.look_up_tool_name, map=self.map_tool_name)
        return success, res


# Manages conversation history. Offers two conversation history shortening strategies:
# 1. cut: directly cut the earlier dialogues if the total tokens exceed a limit.
# 2. pure: use a LLM to shorten the dialogues.
class DialogueMemory:
    def __init__(
        self,
        memory_key: str = "chat_history",
        human_prefix: str = "Human",
        assistant_prefix: str = "Assistant",
        enable_shorten: bool = False,
        max_dialogue_tokens: int = 512,
        shortening_strategy: str = "cut",
        shortening_bot_engine: str = None,
        shortening_bot_type: str = "chat",
        shortening_bot_timeout: int = 60,
    ) -> None:
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.assistant_prefix = assistant_prefix
        self.max_dialogue_tokens = max_dialogue_tokens
        self.enable_shorten = enable_shorten
        if self.max_dialogue_tokens:
            if shortening_strategy not in {"cut", "pure"}:
                raise ValueError(
                    f"Except `shorterning_strategy` to be 'cut' or 'pure', while got {shortening_strategy}."
                )
            self.shortening_strategy = shortening_strategy
            if self.shortening_strategy == "pure":
                # Create the LLM to shorten the dialogues
                self._shortening_bot = OpenAICall(
                    model=shortening_bot_engine,
                    api_key=os.environ["OPENAI_API_KEY"],
                    api_type=os.environ.get("OPENAI_API_TYPE", None),
                    api_base=os.environ.get("OPENAI_API_BASE", None),
                    api_version=os.environ.get("OPENAI_API_VERSION", None),
                    temperature=0.0,
                    model_type=shortening_bot_type,
                    timeout=shortening_bot_timeout,
                )
            else:
                self._shortening_bot = None
        self._shortened_dialogue_history: str = ""  # Store summarized earlier parts of the conversation
        self._shortened_dialogue_turn: int = 0  # Store the turn number of the last summarized dialogue
        self.memory = []  # Store raw (unshortened) conversation history

    # Clear all conversation history (i.e. when restarting dialogues)
    def clear(self) -> None:
        self._shortened_dialogue_history: str = ""
        self._shortened_dialogue_turn: int = 0
        self.memory = []

    # Get the conversation history by combining the raw conversation history and 
    # the summarized dialogues
    def get(self) -> str:
        info = ""
        for m in self.memory[self._shortened_dialogue_turn :]:
            info += f"{m['role']}: {m['content']} \n"
        return self._shortened_dialogue_history + "\n" + info

    # Append new message to the history
    def append(self, role: str, message: str) -> None:
        assert role in {
            "human",
            "assistant",
        }, "role must be 'human' or 'assistant'"
        role = self.human_prefix if role == "human" else self.assistant_prefix
        self.memory.append({"role": role, "content": message})
        if self.enable_shorten:
            self.shorten()

    # Shorten the conversation history if the total tokens exceed the limit
    def shorten(self) -> None:
        total_dialogue = self.get()
        total_tokens = num_tokens_from_string(total_dialogue)
        if (self.max_dialogue_tokens is not None) and (
            total_tokens > self.max_dialogue_tokens
        ):
            if self._shortening_bot:
                # use LLM to shorten the dialogues
                sys_prompt = f"You are a helpful assistant to summarize conversation history and make it shorter. The output should be like:\n{self.human_prefix}: xxxx\n{self.assistant_prefix}: xxx. "
                user_prompt = f"Please help me to shorten the conversational history below. \n{total_dialogue}"
                output = self._shortening_bot.call(
                    user_prompt=user_prompt,
                    sys_prompt=sys_prompt,
                    max_tokens=self.max_dialogue_tokens
                )
            else:
                # directly cut earlier dialogues
                dialogues = [f"{m['role']}: {m['content']}" for m in self.memory]
                _dialogue_str = "\n".join(dialogues)
                while num_tokens_from_string(_dialogue_str) > self.max_dialogue_tokens:
                    dialogues = dialogues[2:]
                    _dialogue_str = "\n".join(dialogues)
                output = _dialogue_str

            self._shortened_dialogue_history = output
            self._shortened_dialogue_turn = len(self.memory)


class CRSAgentPlanFirstOpenAI:
    '''
    Main LLM orchestrator class. It receives the input from the user, generates the plan
    for the tools to be executed, and then executes the tools. It also manages the
    conversation history and the user profile. It uses OpenAI API to generate the
    responses to the user. It also uses a critic to evaluate the responses and decide
    whether to rechain the conversation.
    '''
    def __init__(
        self,
        domain: str,
        tools: Dict[str, Callable],
        candidate_buffer,
        item_corpus,
        engine: str = "text-davinci-003",
        bot_type: str = "completion",
        enable_shorten: bool = False,
        shorten_strategy: str = "pure",
        demo_mode: str = "zero",
        demo_dir_or_file: str = None,
        num_demos: int = 3,
        critic: Critic = None,
        reflection_limits: int = 3,
        verbose: bool = True,
        timeout: int = 60,
        reply_style: str = "detailed",
        user_profile_update: int = -1,
        planning_recording_file: str = None,
        enable_summarize: int = 1,
        **kwargs,
    ):
        # Redo all the domain-specific renaming here
        self.domain = domain
        self._domain_map = {
            "item": self.domain,
            "Item": self.domain.capitalize(),
            "ITEM": self.domain.upper(),
        }
        self._tool_names = {k: v.name for k, v in tools.items()}
        self._tools = list(tools.values())
        self.toolbox = ToolBox(
            "ToolExecutor", TOOLBOX_DESC, {tool.name: tool for tool in self._tools}
        )
        # Receives both candidate buffer and item corpus. In the initial version, the
        # candidate buffer was initialized with the corpus, but we can do things differently
        self.candidate_buffer = candidate_buffer
        self.item_corpus = item_corpus
        self.engine = engine
        assert bot_type in {
            "chat",
            "completion",
        }, f"`bot_type` should be `chat` or `completion`, while got {bot_type}"
        self.bot_type = bot_type
        self.mode = "accuracy"
        self.enable_shorten = enable_shorten
        assert shorten_strategy in {
            "cut",
            "pure",
        }, f"`shorten_strategy` should be `cut` or `pure`, while got {shorten_strategy}"
        self.shorten_strategy = shorten_strategy
        self.kwargs = kwargs
        # Whether to include in-context examples (demonstrations) or not
        if demo_mode == "zero":
            self.selector = None
        else:
            if demo_dir_or_file:
                self.selector = DemoSelector(
                    demo_mode, demo_dir_or_file, k=num_demos, domain=domain
                )
            else:
                self.selector = None

        self.critic = critic

        # reflection
        self.reflection_limits = reflection_limits
        self._reflection_cnt = 0

        self.verbose = verbose
        self.timeout = timeout

        assert reply_style in {
            "concise",
            "detailed",
        }, f"`reply_style` should be `concise` or `detailed`, while got {reply_style}"
        self.reply_style = reply_style
        self.memory = None
        self.agent = None
        self.prompt = None
        self.user_profile_update = user_profile_update
        self._k_turn = 0
        self.user_profile = None
        self.planning_recording_file = planning_recording_file
        self._check_file(planning_recording_file)
        self._record_planning = (self.planning_recording_file is not None)
        self._plan_record_cache = {"traj": [], "conv": [], "reward": 0}

        self.enable_summarize = enable_summarize

    # Ensure that path exists for the planning recording file
    def _check_file(self, fpath: str):
        if fpath:
            dirname = os.path.dirname(fpath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)


    # Initialize openAI API agents for shortening and orchestration
    def init_agent(self, temperature: float = 0.0):
        self.memory = DialogueMemory(
            enable_shorten=self.enable_shorten,
            shortening_strategy=self.shorten_strategy,  # summarize or cut
            shortening_bot_engine=self.engine,
            shortening_bot_type=self.bot_type,
            shortening_bot_timeout=self.timeout,
        )
        self.prompt = self.setup_prompts(self._tools)
        stopwords = ["Obsersation", "observation", "Observation:", "observation:"]
        self.agent = OpenAICall(
            model=self.engine,
            api_key=os.environ["OPENAI_API_KEY"],
            api_type=os.environ.get("OPENAI_API_TYPE", None),
            api_base=os.environ.get("OPENAI_API_BASE", None),
            api_version=os.environ.get("OPENAI_API_VERSION", None),
            temperature=temperature,
            model_type=self.bot_type,
            timeout=self.timeout,
            stop_words=stopwords,
        )
        if self.user_profile_update > 0:
            self.user_profile = UserProfileMemory(llm_engine=self.agent)

    # Customize system prompt to contain all information about what tools are 
    # available and what they do
    def setup_prompts(self, tools: List[Tool]):
        # Dictionary of tool name and description fed to the system prompt
        tools_desc = "\n".join([f"{tool.name}: {tool.desc}" for tool in self._tools])
        tool_names = "[" + ", ".join([f"{tool.name}" for tool in self._tools]) + "]"
        template = SYSTEM_PROMPT_PLAN_FIRST.format(
            tools_desc=tools_desc,
            tool_exe_name=self.toolbox.name,
            tool_names=tool_names,
            **self._tool_names,
            **self._domain_map,
        )
        return template

    # Sets conversation mode from the interactive interface
    def set_mode(self, mode: str):
        assert mode in {
            "diversity",
            "accuracy",
        }, "`mode` should be `diversity` or `accuracy`"
        self.mode = mode
        for tool in self._tools:
            if hasattr(tool, "mode"):
                tool.mode = mode
        logger.debug(f"Mode changed to {mode}.")

    # Sets conversation style
    def set_style(self, style: str):
        if style not in {"concise", "detailed"}:
            raise ValueError(
                f"`reply_style` shoud be `concise` or `detailed`, while got {style}."
            )
        self.reply_style = style
        logger.debug(f"Reply style changed to {style}.")

    # Report how many times tools invocation was unsuccessful
    @property
    def failed_times(self):
        return self.toolbox.failed_times
    
    # Record the plan along with the reward for potential future training
    def save_plan(self, reward: int):
        self._plan_record_cache['reward'] = reward
        with open(self.planning_recording_file, 'a') as f:
            f.write(json.dumps(self._plan_record_cache))
            f.write("\n")
        logger.info(f"Append the plan to the {self.planning_recording_file}.")

    # Clear all history including conversation history, user profile and planning record
    def clear(self):
        self.memory.clear()
        if self._record_planning:
            self._plan_record_cache = {"traj": [], "conv": [], "reward": 0}
        logger.debug("History Cleared!")
        logger.debug("Memory Cleared!")
        if self.user_profile:
            self.user_profile.clear()

    # Main function to run the agent. It receives the input from the user, and invokes the plan
    # execution tool. It offers options for single-turn and multi-turn conversations, and manages
    # the user profile. It invokes the critic for plan improvement through reflection.
    def run(
        self, inputs: Dict[str, str], chat_history: str = None, reflection: str = None
    ):
        prompt_map = {
            "agent_scratchpad": "",
            "tools": f"{self.toolbox.name}: {self.toolbox.desc} \n",
            "examples": "" if not self.selector else self.selector(inputs["input"]),
            "history": "",  # chat history
            "input": inputs["input"],
            "reflection": "" if not reflection else reflection,
            "table_info": self.item_corpus.info(query=inputs["input"])
        }

        # Reset candidate buffer and failed times
        self.toolbox.failed_times = 0
        self.candidate_buffer.clear_tracks()
        self.candidate_buffer.clear()

        if chat_history is not None:
            # for one-turn evaluation, just obtain chat history
            # we could use this initially if we are not interested in follow-up prompts
            prompt_map["history"] = chat_history
        else:
            # for multi-turn conversation, get summarized chat history from DialogueMemory class
            prompt_map["history"] = self.memory.get()

        # Record input to the planning
        if self._record_planning:
            self._plan_record_cache['conv'].append(
                {"role": "user", "content": prompt_map["input"]}
            )

        # Generate response using the plan-first strategy, which invokes the tools
        response = self.plan_and_exe(self.prompt, prompt_map)

        rechain = False

        # Invokes critic to evaluate the response and decide whether to rechain
        # To rechain, invokes the run() function recursively. Rechains up to the
        # maximum reflection limit, or until the response is satisfactory
        # Cool to see how this can be used to improve the response quality
        if self.critic is not None:
            rechain, reflection = self.critic(
                inputs["input"],
                response,
                prompt_map["history"],
                self.candidate_buffer.track_info,
            )
            if rechain:
                logger.debug(
                    f"Reflection: Need Reflection! Reflection information: {reflection}"
                )
                if self._reflection_cnt < self.reflection_limits:
                    self._reflection_cnt += 1
                    # do rechain
                    response = self.run(inputs, reflection=reflection)
                    self._reflection_cnt = 0
                else:
                    rechain = False
                    logger.debug(
                        "Reflection exceeds max times, adopt current response."
                    )
            else:
                logger.debug("No rechain needed, good response.")

        # If not rechain is required, append the response to the memory and
        # update the planning record
        if not rechain:
            self.memory.append("human", inputs["input"])
            self.memory.append("assistant", response)
            if self._record_planning:
                self._plan_record_cache['conv'].append(
                    {"role": "assistant", "content": response}
                )
            logger.debug(f"Response:\n {response}\n")
        
        self._k_turn += 1

        # Update user profile with current memory
        if self._k_turn > 0 and (self.user_profile_update > 0) and (self._k_turn % self.user_profile_update == 0):
            self.user_profile.update(self.memory.get())
            self.memory.clear()
            logger.debug("User Profile Updated!")

        return response


    # Function for plan first strategy. It generates the plan for the tools to be executed,
    # executes the tools and returns the response to the user. Also records the planning
    def plan_and_exe(self, prompt: str, prompt_map: Dict) -> str:
        
        # Format prompt
        prompt = prompt.format(**prompt_map)
        # time the openai call
        start = time.time()
        # Call the agent to generate the plan
        llm_output = self.agent.call(user_prompt=prompt)
        end = time.time()
        logger.debug(f"OpenAI API call latency: {end - start} s.")
        # Parse the llm output to interpret it
        finish, info = self._parse_llm_output(llm_output)

        # Record the plan
        if self._record_planning:
            self._plan_record_cache['traj'].append(
                {'role': "prompt", "content": prompt}
            )
            self._plan_record_cache['traj'].append(
                {"role": "plan", "content": llm_output}
            )

            # use save_plan method to save plan to file
            self.save_plan(reward=0)

        if finish:
            # if the llm provided a clear response, return response
            resp = info
        else:
            # if the llm returned a plan to run, run to plan by invoking
            # the toolbox
            logger.debug(f"Plan: {info}")
            success, result = self.toolbox.run(info, self.user_profile)
            logger.debug(f"Execution Result: \n {result}")
            if success:
                # include plan to the prompt map
                prompt_map['plan'] = info
                
                start = time.time()
                if self.enable_summarize:
                    # Summarize the response to keep context window concise
                    resp = self._summarize_recommendation(result, prompt_map)
                else:
                    resp = result
                # total_token_usage.update(token_usage)
                end = time.time()
                logger.debug(f"Result summarization latency: {end - start} s.")
            else:
                resp = "Something went wrong, please retry."

        return resp
    
    # Parse the output from the LLM to interpret it
    def _parse_llm_output(self, llm_output: str) -> Tuple[bool, str]:
        
        # Preprocessing
        llm_output = llm_output.strip()
        llm_output = llm_output.replace("###", "")  # in case of special token
        
        # Looks for indications of an action plan
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            if "Final Answer:" in llm_output:
                # If no indications and "final answer", return final answer of the LLM
                return (True, llm_output.split("Final Answer:")[-1].strip())
            # Otherwise, return the LLM's output that is not matched to any of the presets
            logger.debug(f"LLM's output not matched: {llm_output}")
            return (True, llm_output)

        # If a plan is found, extract and return action plan
        action = match.group(1).strip()
        action_input = match.group(2)
        return (False, action_input)
    

    # Invoke the summarization agent to generate a concise response
    def _summarize_recommendation(
        self, tool_result: str, prompt_map: dict
    ) -> Tuple[str, Dict]:
        sys_prompt = (
            "You are a conversational recommender assistant. "
            "You are good at give comprehensive response to recommend some items to human. "
            f"Items are retrieved by some useful tools: {OVERALL_TOOL_DESC.format( **self._domain_map)} \n"
            "You need to give some explainations for the recommendation according to the chat history with human."
        )

        user_prompt = (
            f"Previous Chat History: \n{prompt_map['history']}.\n\n"
            f"Human's Input:\n {prompt_map['input']}.\n\n"
            f"Tool Execution Track:\n {self.candidate_buffer.track_info}.\n\n"
            f"Execution Result: \n {tool_result}.\n\n"
            "Please use those information to generate flexible and comprehensive response to human. Never tell the tool names to human.\n"
        )

        """
        Invokes the API once more to summarize the final results. Functionality like this might be leveraged to
        do some more involved reasoning in the future, for example for the affordability tool to pick the final
        items that are within the user's budget.
        """
        if self.reply_style == "concise":
            user_prompt += "Do not give details about items and make the response concise."

        resp = self.agent.call(user_prompt=user_prompt, sys_prompt=sys_prompt, temperature=1.0)
        return resp
    
    # Interfaces with Gradio to bring the input from gradio format to the one accepted
    # by agent.run(). It also returns the output in the format expected by Gradio
    # We could sidestep this part if we dont need the interactive interface
    def run_gr(self, state):
        
        # Extract user input from last state (first element in list)
        text = state[-1][0]
        logger.debug(
            f"\nProcessing run_text, Input text: {text}\nCurrent state: {state}\n"
            f"Current Memory: \n{self.memory.get()}"
        )
        try:
            # Track both time and tokens used
            tic = time.time()
            with get_openai_tokens() as cb:
                # Run the agent and generate the response
                response = self.run({"input": text.strip()})
                logger.debug(cb.get())
            toc = time.time()
            logger.debug(f"Time collapsed: {toc - tic} s.")

        except Exception as e:
            # Return error if something went wrong with invoking agent
            logger.debug(f"Here is the exception: {e}")
            response = "Something went wrong, please try again."
        
        # Update last state with agent response (second element in list)
        state[-1][1] = response
        
        return state, state


if __name__ == "__main__":
    pass
