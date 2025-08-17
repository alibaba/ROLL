# Copyright LiveCodeBench @ 2024
import re

def extract_code_generation(model_output: str):
    if "<|begin_of_solution|>" in model_output:
        model_output = model_output.split("<|begin_of_solution|>")[-1].strip()
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1].strip()
    if "```" not in model_output:
        return model_output.strip()
    else:
        code_pattern = r"```(cpp|python|java|python3)\s*\n*(.*?)```"
        code = re.findall(code_pattern, model_output, re.DOTALL)
        if code and len(code) > 0:
            return code[-1][1]
        else:
            solutions = re.findall(r"```(.*?)```", model_output, re.DOTALL)
            if len(solutions) == 0:
                return ""
            def_solutions = [sol for sol in solutions if "def" in sol]
            if def_solutions:
                return def_solutions[-1]
            return solutions[-1]
