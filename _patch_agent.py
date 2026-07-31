import re
path = r"d:\meau\main\agent.py"
with open(path, "r", encoding="utf-8") as f:
    t = f.read()

old_block = """        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
            "timeout": LLM_REQUEST_TIMEOUT,
            "max_retries": 1,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url"""

new_block = """        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
            "timeout": LLM_REQUEST_TIMEOUT,
            "max_retries": 1,
            "max_tokens": 4096,
        }
        # DeepSeek 模型禁用 thinking 模式（仅需工具调用决策）
        if "deepseek" in model.lower():
            llm_kwargs["model_kwargs"] = {"thinking": {"type": "disabled"}}
        if base_url:
            llm_kwargs["base_url"] = base_url"""

assert old_block in t, "old_block not found!"
t = t.replace(old_block, new_block, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(t)
print("OK")
