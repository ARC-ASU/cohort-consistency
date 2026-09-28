"""
Retriever header. This file is prepended (via ``exec``) to every generated program, both during
RL training (verl/workers/reward_manager/cohort.py), so it must stay self-contained.

``retrieve(question, answer_type)`` sends an atomic fact-lookup question to the retriever LLM.
A few-shot rejection prompt makes the retriever reply "idk" to non-atomic (multi-step or
subjective) queries; this is raised as ``ValueError("idk")`` and counted as a rejection.

Endpoint (OpenAI-compatible, e.g. ``vllm serve``), configured through environment variables:
    COHORT_LLM_BASE_URL          default http://0.0.0.0:5500/v1
    COHORT_LLM_MODEL             served model name (paper: Qwen2.5-{3B,7B}-Instruct)
    COHORT_LLM_API_KEY           default "PROGRAM"
    COHORT_RETRIEVER_MAX_TOKENS  default 1024
"""
import os
import re
import json
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI


def call_text_model(messages, port=5500, max_tokens=None, model_name=None, temperature=0.7):
    client = OpenAI(
        base_url=os.environ.get("COHORT_LLM_BASE_URL") or f"http://0.0.0.0:{port}/v1",
        api_key=os.environ.get("COHORT_LLM_API_KEY", "PROGRAM"),
    )
    response = client.chat.completions.create(
        model=model_name or os.environ.get("COHORT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        messages=messages,
        max_tokens=max_tokens or int(os.environ.get("COHORT_RETRIEVER_MAX_TOKENS", "1024")),
        temperature=temperature,
        n=1
    )
    return [choice.message.content for choice in response.choices]


def extract_json_from_text(text):
    """Parse the JSON object enclosed in ```json ... ```."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return None


def retrieve(prompt: str, return_type: type) -> any:
    messages = [
      {"role": "system", "content": "You are a fact-lookup assistant. For each user query, first decide if it's a simple, single-step fact lookup without solving it and then return a JSON object with exactly one key, \"answer\", wrapped in ```json ...```. Match the type specified in parentheses (int, str, list, bool). If a query requires more than a straightforward fact check or true/false lookup—for example, multi-step reasoning or subjective judgment—reply with \"idk\"."},
      {"role": "user", "content": "Who finished immediately after the winner at the 1992 Olympic 100m final? (str)"},
      {"role": "assistant", "content": "[Explanation] You must identify the winner, then determine who came second, so this isn't a single-step fact lookup.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "How many planets are in the solar system? (int)"},
      {"role": "assistant", "content": "[Explanation] This is a simple fact check.```json{\"answer\": 8}```"},
      {"role": "user", "content": "What is the profession of Michael Jackson? (str)"},
      {"role": "assistant", "content": "[Explanation] Asks for a single well-known profession of a public figure. It is a simple fact check.```json{\"answer\": \"singer\"}```"},
      {"role": "user", "content": "Who has more than one Nobel Prize? (list)"},
      {"role": "assistant", "content": "[Explanation] Requests a factual list of individuals with multiple Nobel Prizes.```json{\"answer\": [\"John Bardeen\", \"Frederick Sanger\", \"Linus Pauling\", \"Marie Curie\"]}```"},
      {"role": "user", "content": "Is the CEO of Tesla older than the current President of France? (bool)"},
      {"role": "assistant", "content": "[Explanation] This requires retrieving two birthdates (Tesla's CEO and France's President) and then comparing their ages.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "Is the Eiffel Tower located in Paris, France? (bool)"},
      {"role": "assistant", "content": "[Explanation] A single-step factual question about a landmark's location.```json{\"answer\": false}```"},
      {"role": "user", "content": "Did England win any Olympic gold medals in 1800? (bool)"},
      {"role": "assistant", "content": "[Explanation] To answer this you first must check when the modern Olympics began, then verify England's medal record—this is multi-step historical reasoning.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "What is the population of the largest country entirely south of the equator? (int)"},
      {"role": "assistant", "content": "[Explanation] Must identify the country fully south of the equator then look up its population.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "List the U.S. states admitted to the Union between the first and the last of the original 13 colonies. (list)"},
      {"role": "assistant", "content": "[Explanation] Requires ordering states by admission date and filtering between two anchors.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "Can food be cooked in the cosmic microwave background? (bool)"},
      {"role": "assistant", "content": "[Explanation] You must first look up the CMB temperature (~2.7 K), then compare that to the physics of cooking—this is multi-step scientific reasoning.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "Are Waris Hussein and Mathieu Kassovitz both actors? (bool)"},
      {"role": "assistant", "content": "[Explanation] You must look up each person's profession separately and then compare—this is multi-step reasoning.```json{\"answer\": \"idk\"}```"},
      {"role": "user", "content": "{} ({})".format(prompt, return_type.__name__)}
    ]
    responses = call_text_model(messages)
    results = []
    for i in range(len(responses)):
        try:
            out = extract_json_from_text(responses[i])
            if out is not None:
                results.append(out["answer"])
        except:
            continue
    if any("idk" in str(x) for x in results):
        raise ValueError("idk")
    
    return return_type(results[0])
