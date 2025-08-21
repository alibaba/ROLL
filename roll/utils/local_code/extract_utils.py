# Copyright LiveCodeBench @ 2024
import re

def extract_code_generation(model_output: str):
    """
    Extract code from model output with various formats.
    """
    if "<|begin_of_solution|>" in model_output:
        model_output = model_output.split("<|begin_of_solution|>")[-1].strip()
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1].strip()
    
    if "```" not in model_output:
        return model_output.strip()
    
    code_pattern = r"```([\w\+\#\-\.]*)\s*\n*(.*?)```"
    code = re.findall(code_pattern, model_output, re.DOTALL)
    
    if code and len(code) > 0:
        def_solutions = [sol[1] for sol in code if "def " in sol[1]]
        if def_solutions:
            return def_solutions[0]
        return code[0][1]
    else:
        solutions = re.findall(r"```(.*?)```", model_output, re.DOTALL)
        if not solutions:
            return ""
        def_solutions = [sol for sol in solutions if "def " in sol]
        if def_solutions:
            return def_solutions[0]
        
        return solutions[0]
