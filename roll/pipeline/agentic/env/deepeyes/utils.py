"""
reference: https://github.com/Visual-Agent/DeepEyes/blob/main/verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py
"""


import numpy as np
from typing import Dict, Any
import re
import json
from math import ceil, floor


class PROMPT:
    SYSTEM_PROMPT_V1 = """You are a helpful assistant.
    # Tools
    You may call one or more functions to assist with the user query.
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to zoom in."},"bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["image_path","bbox"]}}}
    {"type":"function","function":{"name":"image_rotate_tool","description":"Rotate an image by a specified angle (clockwise or counterclockwise).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to be rotated."},"angle":{"type":"integer","description":"Rotation angle in degrees (e.g., 90, 180, 270). Positive values for clockwise, negative for counterclockwise."}},"required":["image_path","angle"]}}}
    </tools>
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>"""
    # user v1 failed, model do not output toolcall
    USER_PROMPT_V1 = "\nReason in your mind and then give the final answer. Output strictly following the format <think>[your inner thoughts]</think><answer>[your final answer]</answer>."
    # v2: no image_path
    #     SYSTEM_PROMPT_V2 = """You are a helpful assistant.
    # # Tools
    # You may call one or more functions to assist with the user query.
    # You are provided with function signatures within <tools></tools> XML tags:
    # <tools>
    # {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["bbox"]}}}
    # </tools>
    # For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    # <tool_call>
    # {"name": <function-name>, "arguments": <args-json-object>}
    # </tool_call>"""
    SYSTEM_PROMPT_V2 = """You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
    USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
    SYSTEM_PROMPT_V3 = ""
    USER_PROMPT_V3 = """\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. 
Otherwise generate a new grouding in JSON format:
```json\n{\n  "function": "zoom_in",\n  "bbox_2d": [x1, y1, x2, y2],\n  "label": "object_name"\n}\n``` 
The zoomed-in image of your grounding will be provided in next turn.
"""
    SYSTEM_PROMPT_V4 = ""
    USER_PROMPT_V4 = """\nIf the current images are insufficient to answer the question, request a zoom-in by providing this tool_call object within tags:
<tool_call>
{"function": "zoom_in", "bbox_2d": [x1, y1, x2, y2], "label": "object_name"}
</tool_call>
The zoomed image will be provided in the next turn. Otherwise, provide your answer within <answer> </answer> tags.
"""
    SYSTEM_PROMPT_V5 = """You are a helpful assistant.
# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
You may call **one or more** functions to assist with the user query.
**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [8, 40, 50, 150], "label": "the person under the tree"}}  
</tool_call>"""
    # USER_PROMPT_V5 = "\nThink first, call **image_zoom_in_tool** one or more times if needed, i.e., <think>...</think>  <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    # # 看第一轮的rollout，这个会有一些问题，导致模型最后没回答，只是说了一句信息完备，不用调工具了。后续观察score上涨很快，应该自己学会了！
    # TURN_PROMPT_V5 = "\nAbove are the tool responses after calling {}. Think first, continue to call **image_zoom_in_tool** if needed. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed)."
    #     TURN_PROMPT_V5_PLUS = """Think in your mind first, <think> Analyze the problem thoroughly. Determine if available information suffices or if tools are needed. Decide whether to call tools one or more times or provide final answer.</think>
    # Then execute one action: <tool_call> tools </tool_call> OR <answer> complete response </answer>
    # """
    TURN_PROMPT_V5 = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    USER_PROMPT_V5 = TURN_PROMPT_V5



class VisualToolBoxV2(object):
    name = "visual_toolbox_v2"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."
    user_prompt = PROMPT.USER_PROMPT_V2
    metrics_agg_mode = {
        "extract_answer": "sum",
        "extract_none": "sum",
        "invalid_tool_call": "sum",
        "success_tool_call": "sum",
        "failed_tool_call": "sum",
        "tool_call": "sum",
    }

    def __init__(self):
        self.multi_modal_data = None  # To store the current image being processed

    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r"<answer>(.*?)</answer>", action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        Args:
            action_string: The string containing the tool call in XML tags.
        Returns:
            A dictionary with the tool name and arguments.
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r"<tool_call>(.*?)</tool_call>", action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        Args:
            action_string: The string containing the tool call in XML tags.
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        exe_info = {
            "extract_answer": 0,
            "extract_none": 0,
            "invalid_tool_call": 0,
            "success_tool_call": 0,
            "failed_tool_call": 0,
            "tool_call": 0,
        }
        answer = self.extract_answer(action_string)
        if answer:
            exe_info["extract_answer"] = 1
            return "", 0.0, True, exe_info
        action = self.extract_action(action_string)
        if not action:
            exe_info["extract_none"] = 1
            return "", 0.0, True, exe_info
        exe_info["tool_call"] = 1
        try:
            tool_call = json.loads(action.strip())
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = f"Error: {str(error_msg)}"
            exe_info["invalid_tool_call"] = 1
            return obs, 0.0, False, exe_info
        try:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            if tool_name == "image_zoom_in_tool":
                # Zoom in by cropping the image
                # image_path = args["image_path"]
                bbox = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = self.multi_modal_data["image"][0]
                cropped_img = img.crop(bbox)
                current_image = cropped_img
            elif tool_name == "image_rotate_tool":
                # Rotate the image
                # image_path = args["image_path"]
                angle = args["angle"]
                # img = Image.open(image_path)
                img = self.multi_modal_data["image"][0]
                rotated_img = img.rotate(angle)
                current_image = rotated_img
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            obs = {
                "prompt": "<tool_response>" + "<image>" + self.user_prompt + "</tool_response>",
                "image": [current_image],
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            print(f"[DEBUG] SUCCESS ACTION {action_string=}")
            exe_info["success_tool_call"] = 1
            return obs, reward, done, exe_info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f"[DEBUG] Execute WRONG - {str(e)} {action_string=}")
            obs = f"Error: {str(e)}"
            reward = 0.0  # No reward for failed execution
            done = False
            exe_info["failed_tool_call"] = 1
            return obs, reward, done, exe_info

    def reset(self, image):
        self.multi_modal_data = {"image": image}
        self.height = self.multi_modal_data["image"][0].height
        self.width = self.multi_modal_data["image"][0].width

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f"invalid shape for {left=}, {top=}, {right=}, {bottom=}"
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, (
                f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            )
            assert min(height, width) > 30, f"{height=}, {width=} is too small"
            assert max(height, width) >= 56 and min(height, width) >= 14, (
                "images shape error, input image shape is too small"
            )
            return True
        except Exception as err:
            print(f" [ERROR vl_agent #2] {err=}")
            return False

    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None
        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa
    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa
    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa
    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa
    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa
    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa
    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa
    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.
# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.
Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.
# TONE #
Professional, scientific.
# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]
# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.
**Question**:
{query}
**Reference Answer**
{gold_ans}
## Student Final Answer
{pred_ans}"""
MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.
# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.
Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.
# TONE #
Professional, scientific.
# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]
# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.
**Question**:
{query}
**Reference Answer**
{gold_ans}
## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"
    return full_prompt
