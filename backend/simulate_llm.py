from llm import call_llm
import sys
try:
    resp = call_llm("What is 2+2?")
    print("LLM Success:", resp)
except Exception as e:
    print("LLM Error:", repr(e))
    import traceback
    traceback.print_exc()
