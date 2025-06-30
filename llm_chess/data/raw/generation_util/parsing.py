import ast

def pqt_extract_ground_truth(answer, task_type):
    if task_type == "predictmove":
        return ast.literal_eval(answer)
    elif task_type == "bestmove" or task_type == "worstmove":
        return ast.literal_eval(answer)
    elif task_type == "legalmoves":
        return ast.literal_eval(answer)
    else:
        raise ValueError(f"Task type: {task_type} is undefined.")