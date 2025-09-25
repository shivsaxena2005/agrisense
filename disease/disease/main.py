# caller.py
from test import run_pipeline   # replace with the filename where run_pipeline is defined

# Example 1: Get output in English
result_english = run_pipeline("brown rust", language=1)
print("\n=== English Output ===")
print(result_english)

