import re

def preprocess_code(code: str) -> str:
    # Remove comments
    code = re.sub(r'(?m)^ *#.*\n?', '', code)
    code = re.sub(r'(?m)^ *//.*\n?', '', code)
    # Remove string literals
    code = re.sub(r'".*?"', '', code)
    code = re.sub(r"'.*?'", '', code)
    # Tokenize
    tokens = re.findall(r'\b\w+\b', code)
    return ' '.join(tokens)