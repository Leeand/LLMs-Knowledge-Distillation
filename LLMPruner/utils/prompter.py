"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union
import pdb

alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:" 
}

boolq_template = {
    "description": "Template used by boolq.",
    "prompt": "### Instruction:\nRead the input passage and answer the question: {question}? Your answer should be “Yes'or“No”\n\n### Input:\n{passage}\n\n### Response:\n",
    "response_split": "### Response:"
}



class Prompter(object):
    __slots__ = ("template_name","template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template_name = template_name
        if not template_name or 'alpaca' in template_name:
            self.template = alpaca_template
        else:
            self.template = boolq_template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        # pdb.set_trace()
        if not self.template_name or 'alpaca' in self.template_name:
            if input:
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_no_input"].format(
                    instruction=instruction
                )
        else:
            res = self.template["prompt"].format(
                    question=instruction, passage=input[:800]
                )
        # pdb.set_trace()
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class ZeroPrompter(object):
    __slots__ = ("_verbose")

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        
        if self._verbose:
            print(
                f"Without using prompt template!"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if instruction[-1] == '.':
            instruction = instruction[:-1] + ':'
        if instruction[-1] not in ['.', ':', '?', '!']:
            instruction = instruction + ':'
        instruction += ' '

        if input:
            if input[-1] not in ['.', ':', '?', '!']:
                input = input + '.'
            res = instruction + input
        else:
            res = instruction
        if label:
            res = f"{res} {label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.strip()
