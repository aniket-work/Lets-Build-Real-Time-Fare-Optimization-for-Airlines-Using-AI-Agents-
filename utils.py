import re


def extract_python_code(text):
    pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else "No Python code found."


def extract_final_output(text):
    pattern = re.compile(r'Pricing recommendation: (.*)')
    match = pattern.search(text)
    return match.group(1).strip() if match else "No pricing recommendation found."