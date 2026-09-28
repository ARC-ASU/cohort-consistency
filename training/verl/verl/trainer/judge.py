"""
Frozen judge used for the critique-based rewards.

For each program the judge sees the masked question, the cohort (question, gold, prediction),
the cohort-level reasoning path and the program, and returns a JSON object with
  "score":   factor-complete decomposition score s_fc in [1, 10]   -> R_fc
  "program": an improved program p+; AST similarity(p, p+) in [0, 1] -> R_sa
"""

import os
from typing import Dict, List, Tuple, Optional
import re
import json
import ast
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# The judge is served by the same OpenAI-compatible endpoint (vLLM) as the retriever.
#   COHORT_LLM_BASE_URL  default http://0.0.0.0:5500/v1
#   COHORT_LLM_MODEL     served model name (the path/name passed to `vllm serve`)
#   COHORT_LLM_API_KEY   default "PROGRAM"
def _client(port=None):
    base_url = os.environ.get("COHORT_LLM_BASE_URL") or f"http://0.0.0.0:{port or 5500}/v1"
    return OpenAI(base_url=base_url, api_key=os.environ.get("COHORT_LLM_API_KEY", "PROGRAM"))


def _model_name(model_name=None):
    return model_name or os.environ.get("COHORT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def call_text_model(messages, port=5500, max_tokens=4096, model_name=None, temperature=0.0):
    """Query the judge endpoint and return the list of generated texts."""
    response = _client(port).chat.completions.create(
        model=_model_name(model_name),
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1
    )
    return [choice.message.content for choice in response.choices]


def extract_json_from_text(text):
    """
    Extracts JSON content from a string enclosed in triple backticks and returns the parsed JSON object.

    :param text: The input string containing JSON within triple backticks.
    :return: Parsed JSON object if found, otherwise None.
    """
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_content = match.group(1)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return None
    return None

class NormalizeNames(ast.NodeTransformer):
    """AST transformer to normalize variable and constant names for comparison."""
    
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)
    
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value="CONST"), node)

def normalize_ast(code: str) -> str:
    """
    Normalize an AST by replacing all variable names and constants.
    
    Args:
        code: Python code string
        
    Returns:
        Normalized AST dump
    """
    try:
        tree = ast.parse(code)
        norm_tree = NormalizeNames().visit(tree)
        return ast.dump(norm_tree, annotate_fields=True, include_attributes=False)
    except:
        return ""

def structural_similarity(code1: str, code2: str) -> float:
    """
    Calculate structural similarity between two pieces of code.
    
    Args:
        code1: First code string
        code2: Second code string
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        norm1 = normalize_ast(code1)
        norm2 = normalize_ast(code2)
        if norm1 and norm2:
            return difflib.SequenceMatcher(None, norm1, norm2).ratio()
    except:
        pass
    # Fall back to simple text similarity if AST parsing fails
    return difflib.SequenceMatcher(None, code1, code2).ratio()

def prepare_judge_messages(masked_question: str, similar_questions: List[Tuple], 
                          reasoning: str, program: str, choices: list) -> List[Dict]:
    """
    Prepare messages for a single judge request.
    
    Args:
        masked_question: The masked template question
        similar_questions: List of (question, gold_answer, program_prediction) tuples
        reasoning: The reasoning path
        program: The original program to evaluate and improve
        choices: The answer choices
        
    Returns:
        List of message dictionaries for the API call
    """
    
    sys_prompt = """
You are given a group of questions derived from the same masked template. Your job is (1) to evaluate the PREVIOUS program and assign a single integer score, and (2) to REGENERATE one improved Python function that solves ALL questions in the group. Think step-by-step. Then output a valid JSON with exactly two keys: {"score": <int>, "program": "<code>"}.

------------------------------------------------------------
Your tasks (output JSON with "score" and "program"):

A) Evaluate the PREVIOUS program and produce ONE integer "score" (1-10) based on these 3 dimensions:
   1) Alignment with important reasoning factors from the Reasoning Path (e.g., accounting for alternative eating methods, typical utensil requirements, cultural variability, and defaulting conservatively).
   2) Absence of forbidden shortcuts (NO hard-coded mappings/dictionaries; NO direct string checks for specific entities; NO pattern-matching the literal question text; NO label leakage).
   3) Proper decomposition: breaks the task into simple, orthogonal subquestions via retrieve() or equivalent evidence checks that generalize to unseen items (NOT relying on the specific items in the examples).

Scoring rubric guidance (for your internal use):
- 9-10: Strong factor coverage, no shortcuts, clear multi-step decomposition with robust fallbacks.
- 7-8: Good factor coverage, minor gaps, mostly clean decomposition.
- 5-6: Partial factor coverage and/or weak decomposition.
- 3-4: Major gaps; some shortcut-like behavior or brittle logic.
- 1-2: Fails most dimensions; relies on prohibited shortcuts or ignores factors.
NOTE: If the program used any shortcuts, you do not need to consider other factors and should give it a score <= 2. And set the shortcut field in the output JSON to true.


B) Generate a NEW program as a single Python function:
   - Signature: def answer(..) -> int
   - Return: 0 or 1.
   - MUST use decomposition into general, masked sub-queries (e.g., via a provided retrieve() helper) that DO NOT include any concrete items or terms from the questions. Only use placeholders derived from the inputs (FoodItem, Utensil) and generic concepts (e.g., "requires utensil by norm?", "hand-eatable?", "availability of alternatives?").
   - MUST NOT define or rely on any hard-coded mapping/dictionary/list of specific foods or utensils.
   - MUST NOT read or reference the literal question text.
   - SHOULD reflect Premise and the Inference.
   - Keep the program logic self-contained except for a black-box retrieve(prompt: str, type_hint: type) -> Any helper that you may call with neutral, generalized subquestions (no concrete food names baked into code). You may assume retrieve exists.
   - Keep the code concise, readable, and deterministic given retrieve's booleans/scores; include brief comments.

OUTPUT FORMAT:
First, do some analysis and think step by step. 
Then generate a JSON object with EXACTLY three keys:
```json{"shortcut": <bool>, "score": <integer 1-10>,"program": "<the improved Python code as a single string>"}```
"""
    
    # Format the questions and results
    questions_text = ""
    for i, (question, gold_answer, program_prediction) in enumerate(similar_questions):
        questions_text += f"- Q{i}: \"{question}\" | Gold: {gold_answer} | Program: {program_prediction}\n"
    
    user_prompt = f"""
Masked template
- Masked question: "{masked_question}"
- Choices: {choices}
- Options: ["A", "B"]

Questions and results of the previous program
{questions_text.strip()}

Reasoning Path (reference facts you should respect)
{reasoning}

Previous candidate program (to be evaluated and then improved)

{program}
NOTE: Detect whether it uses any shortcuts first. If the program used any shortcuts, you do not need to consider other factors and should give it a score <= 2. Remember to put the score within the JSON output. First answer do you think the current program is valid or not? And remember to set shortcut to be true if it used any shortcuts.
"""
    example1 = """
Masked template
- Masked question: "What is the main advantage of making an Object PropertyX?"
- Choices: ["to keep it simple to build", "to make it easy to carry"]
- Options: ["A", "B"]

Questions and results of the previous program
- Q0: "What is the main advantage of making a smartphone compact in size?" | Gold: B | Program: B
- Q1: "What is the main advantage of making a bicycle light in weight?"     | Gold: B | Program: B
- Q2: "What is the main advantage of making a book small in size?"    | Gold: B | Program: B
- Q3: "What is the main advantage of making a backpack waterproof?" | Gold: B | Program: B
- Q4: "What is the main advantage of making a tablet portable?" | Gold: B | Program: B
- Q5: "What is the main advantage of making a laptop computer light in weight?" | Gold: B | Program: B

Reasoning Path (reference facts you should respect)
Look at the property mentioned in the question.

Ask first: Does this property make the object easier to build or manufacture?

Example: if the property is "few parts" or "simple design," the advantage would be A (simple to build).

If not, then ask: Does this property make the object easier to carry or move around?

Properties like compact size, lightweight, small, portable → advantage is B (easy to carry).

If the property doesn't clearly fit either, choose the one that is the more reasonable default (usually portability).


Previous candidate program (to be evaluated and then improved)

from typing import List, Dict

def answer(Object: str, PropertyX: str) -> int:
    # Check if the property is a physical characteristic
    is_physical = retrieve(
        f"Is {PropertyX} a physical characteristic of {Object}?",
        bool
    )
    if is_physical:
        return 0  # 'to keep it simple to build'

    # Check if the property is a functional feature
    is_functional = retrieve(
        f"Is {PropertyX} a functional feature of {Object}?",
        bool
    )
    if is_functional:
        return 0  # 'to keep it simple to build'

    # If it's neither physical nor functional, the main advantage is simplicity
    return 0  # 'to keep it simple to build'

"""
    answer1 = """
    Let\'s analyze the previous program step by step:\n\n1. **Alignment with Important Reasoning Factors**:\n   - The program checks if the property is a physical characteristic or a functional feature.\n   - If the property is a physical characteristic, it returns "to keep it simple to build".\n   - If the property is a functional feature, it also returns "to keep it simple to build".\n   - If the property is neither, it defaults to "to keep it simple to build".\n\n2. **Absence of Forbidden Shortcuts**:\n   - The program does not use hard-coded mappings or dictionaries.\n   - It does not directly check the literal question text.\n   - However, it assumes that any property that is not a physical characteristic must be a functional feature, which is a shortcut.\n\n3. **Proper Decomposition**:\n   - The program decomposes the task into checking if the property is a physical characteristic or a functional feature.\n   - However, the decomposition is flawed because it incorrectly assumes that any non-physical property must be functional, which is not always true.\n\nGiven the analysis, the program uses a shortcut by assuming that any non-physical property must be functional, leading to incorrect answers in many cases. Therefore, the score should be <= 2. ```json
"""
    code1 = """
from typing import Any

def answer(Object: str, PropertyX: str) -> int:
       # 1) Manufacturing/build simplicity (neutral, generalized subquestion)
    simplifies_build = retrieve(
        (
            "For an object like '" + Object + "', would making it '" + PropertyX +
            "' mainly reduce part count/design complexity or assembly steps (i.e., simplify manufacturing)?"
        ),
        bool,
    )

    # 2) Portability evidence (generalized, not item-specific)
    aids_portability = retrieve(
        (
            "Does having the property '" + PropertyX + "' for an object like '" + Object +
            "' primarily improve portability (easier to carry due to lower weight, smaller size, compactness, or being portable)?"
        ),
        bool,
    )

    # Decision per path
    if simplifies_build and not aids_portability:
        return 0  # A: simple to build
    if aids_portability and not simplifies_build:
        return 1  # B: easy to carry

    # Tie/unclear -> conservative default per guidance: portability
    return 1  # B: easy to carry
"""
    # print("code", code)
    txt = f'''{{"shortcut": true, "score": 2, "program": {json.dumps(code1)} }}'''
    answer1 = answer1 + txt + "\n```"
    example2 = """
Masked template
- Masked question: "Are both founders of CompanyX still involved in the company?"
- Choices: ["No", "Yes"]
- Options: ["A", "B"]

Questions and results of the previous program
- Q0: "Are both founders of Google still involved in the company?" | Gold: B | Program: A
- Q1: "Are both founders of Facebook still involved in the company?" | Gold: B | Program: B
- Q2: "Are both founders of Apple still involved in the company?" | Gold: B | Program: B
- Q3: "Are both founders of Nvidia still involved in the company?" | Gold: B | Program: A
- Q4: "Are both founders of Microsoft still involved in the company?" | Gold: B | Program: A
- Q5: "Are both founders of Twitter still involved in the company?" | Gold: B | Program: B

Reasoning Path (reference facts you should respect)
Enumerate founders of CompanyX (include all co-founders).

For each founder, check whether they are currently involved in an official or ongoing capacity (executive role, board seat, formal advisory role, or sustained operational responsibility).

Answer Yes (B) only if all founders are still involved; otherwise No (A).

Edge cases:

If CompanyX has merged/been acquired/renamed, assess involvement with the current successor entity.

Honorary titles without real duties do not count as involvement unless there is clear, ongoing governance/operational responsibility.

If evidence is incomplete or conflicting, default conservatively: unless all founders clearly remain involved, answer No (A).

Previous candidate program (to be evaluated and then improved)

from typing import List, Dict

def answer(CompanyX: str) -> int:
    # Retrieve the names of the founders of CompanyX
    founders = retrieve(
        f"Who are the founders of {CompanyX}?",
        list
    )

    # Check if each founder is still involved with the company
    still_involved = []
    for founder in founders:
        involved = retrieve(
            f"Is {founder} still involved with {CompanyX}?",
            bool
        )
        still_involved.append(involved)

    # If all founders are still involved, return 'Yes'
    if all(still_involved):
        return 1
    else:
        return 0
NOTE: If the program used any shortcuts, you should give it a score <= 3.

"""
    answer2 = """
Let\'s analyze the previous program step by step:\n\n1. **Alignment with Important Reasoning Factors**:\n   - The program correctly retrieves the names of the founders of the company.\n   - It then checks if each founder is still involved in the company.\n   - It correctly returns `1` (Yes) if all founders are still involved and `0` (No) otherwise.\n\n2. **Absence of Forbidden Shortcuts**:\n   - The program does not use hard-coded mappings or dictionaries.\n   - It does not directly check the literal question text.\n   - It does not rely on specific companies or founders.\n\n3. **Proper Decomposition**:\n   - The program decomposes the task into retrieving the founders and checking their involvement.\n   - This decomposition is general and can handle different companies and founders.\n\nGiven the analysis, the program does not use any shortcuts and follows the reasoning path correctly. ```json
"""
    code2 = """
from typing import List


def answer(CompanyX: str) -> int:
    # Get all original founders (fallback to empty list if unavailable)
    founders =  retrieve(
        f"Who are the founders of {CompanyX}?",
        list
    )
    if not founders:
        return 0  # Conservative default: cannot verify → No

    for founder in founders:
        # Ask if this founder has an ongoing, substantive role
        involved = retrieve(
            f"Does {founder} currently hold an executive, board, or formal advisory role with {CompanyX}?",
            bool
        )
        honorary = retrieve(
            f"Is {founder}'s association with {CompanyX} only honorary/ceremonial without real duties?",
            bool
        )
        unclear = retrieve(
            f"Is information about {founder}'s current involvement with {CompanyX} unclear or conflicting?",
            bool
        )

        # If any founder is not clearly substantively involved, return No
        if not involved or honorary or unclear:
            return 0

    # All founders are substantively involved
    return 1
"""
    # print("code", code)
    # txt = f'''{{"score": 8, "program": {json.dumps(code2)} }}'''
    txt = f'''{{"shortcut": false, "score": 8, "program": {json.dumps(code2)} }}'''
    answer2 = answer2 + txt + "\n```"
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": example1},
        {"role": "assistant", "content": answer1},
        {"role": "user", "content": example2},
        {"role": "assistant", "content": answer2},
        {"role": "user", "content": user_prompt},
    ]

def judge(masked_question: str, similar_questions: List[Tuple], reasoning: str, 
          program: str, choices: list, port=5500, model_name=None) -> Tuple[int, float, Optional[str]]:
    """
    Single judge function that evaluates a program and returns a score and similarity measure.
    
    Args:
        masked_question: The masked template question
        similar_questions: List of (question, gold_answer, program_prediction) tuples
        reasoning: The reasoning path
        program: The original program to evaluate and improve
        choices: The answer choices
        port: Port for the API server
        model_name: Model name for the API
        
    Returns:
        Tuple[int, float, Optional[str]]: (score, structural_similarity, new_program)
    """
    
    messages = prepare_judge_messages(
        masked_question=masked_question,
        similar_questions=similar_questions,
        reasoning=reasoning,
        program=program,
        choices=choices
    )
    
    responses = call_text_model(messages, port=port, model_name=model_name, temperature=0)
    
    for response in responses:
        try:
            output = extract_json_from_text(response)
            if output is not None and "score" in output and "program" in output:
                score = int(output["score"])
                new_program = output["program"]
                
                # Calculate structural similarity between original and new program
                similarity = structural_similarity(program, new_program)
                
                return score, similarity, new_program
        except Exception as e:
            continue
    
    # If no valid response found, return default values
    return 0, 0.0, None

def batch_judge(batch_inputs: List[Dict], port=5500, max_tokens=4096, 
                model_name=None, temperature=0.0, batch_size=20) -> List[Tuple[int, float, Optional[str]]]:
    """
    Batch version of the judge function that processes multiple inputs at once.
    
    Args:
        batch_inputs: List of dictionaries containing judge inputs
            Each dict should have keys: masked_question, similar_questions, reasoning, program, choices
        port: Port for the API server
        max_tokens: Maximum tokens for generation
        model_name: Model name for the API
        temperature: Temperature for generation
        batch_size: Number of requests to process in parallel
        
    Returns:
        List of tuples containing (score, similarity, new_program) for each input
    """
    
    client = _client(port)
    model_name = _model_name(model_name)

    results = []
    
    # Process in batches
    for batch_start in range(0, len(batch_inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_inputs))
        current_batch = batch_inputs[batch_start:batch_end]
        
        # Prepare all messages for this batch
        batch_messages = []
        for input_dict in current_batch:
            messages = prepare_judge_messages(
                masked_question=input_dict["masked_question"],
                similar_questions=input_dict["similar_questions"],
                reasoning=input_dict["reasoning"],
                program=input_dict["program"],
                choices=input_dict["choices"]
            )
            batch_messages.append(messages)
        
        # Make parallel API calls for the current batch
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create futures with their original indices
            future_to_index = {}
            for idx, messages in enumerate(batch_messages):
                future = executor.submit(
                    client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                future_to_index[future] = idx
            
            # Initialize responses list with correct size
            batch_responses = [None] * len(batch_messages)
            
            # Collect responses and place them in correct positions
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    response = future.result(timeout=60)  # 30 second timeout per request
                    batch_responses[idx] = response
                except Exception as e:
                    # print(f"Error in API call for index {idx}: {e}")
                    batch_responses[idx] = None
        
        # Process responses
        for i, response in enumerate(batch_responses):
            if response is None:
                results.append((0, 0.0, None))
                continue
                
            try:
                response_text = response.choices[0].message.content
                output = extract_json_from_text(response_text)
                # print("Judge response:", response_text)
                # print("Extracted output:", output)
                
                if output is not None and "score" in output and "program" in output:
                    # print("Output JSON:", output)
                    if "shortcut" in output and output["shortcut"] == True:
                        score = 1
                    else:
                        if "shortcut" in output and output["shortcut"] == False:
                            score = min(4+int(output["score"]), 10)
                        else:
                            score = int(output["score"])
                    new_program = output["program"]
                    
                    # Calculate structural similarity
                    original_program = current_batch[i]["program"]
                    # print(f"Original Program:\n{original_program}\nNew Program:\n{new_program}\n")
                    similarity = structural_similarity(original_program, new_program)
                    
                    results.append((score, similarity, new_program))
                else:
                    results.append((0, 0.0, None))
            except Exception as e:
                # print(f"Error processing response: {e}")
                results.append((0, 0.0, None))
    
    return results

# For backward compatibility - export the single judge function as well
__all__ = ['judge', 'batch_judge', 'structural_similarity', 'call_text_model']