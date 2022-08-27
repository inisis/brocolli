import re

def get_function_name(node_target):
    function_name = re.findall(r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node_target))[0]

    return function_name