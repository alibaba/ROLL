# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
Qwen2_5_7B_Instruct_RLVR_Prompt = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
"""

LONGCOT_QWEN_2_5_SYSTEM = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
# template name -> default system
MATH_DEFAULT_SYSTEMS = {
    "qwen2_5": "Please reason step by step, and put your final answer within \\boxed{}.",
}
CODE_DEFAULT_SYSTEMS = {
    "qwen2_5": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}

BASE_CHAT_FORMAT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    " The assistant first thinks about the reasoning process in the mind and then provides the user "
    "with the answer, ensuring that the final result in the answer is enclosed in \\boxed{{}}. The "
    "reasoning process and answer are enclosed within '<think>' '</think>' and '<answer>' '</answer>' "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {{content}} Assistant:"
)

prompt_maps = {"Qwen2.5-7B-Instruct-RLVR-prompt": Qwen2_5_7B_Instruct_RLVR_Prompt}
