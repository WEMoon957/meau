import os; from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
c = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'])

# 测试1：高 max_tokens（让推理+回复都有空间）
print("=== 测试1: max_tokens=2000 ===")
r = c.chat.completions.create(model="deepseek-v4-flash", messages=[{"role":"user","content":"你是谁？一句话"}], max_tokens=2000)
print(f"content={r.choices[0].message.content}")
print(f"usage={r.usage}")

# 测试2：禁用思考（普通输出）
print("\n=== 测试2: 禁用 thinking ===")
r2 = c.chat.completions.create(
    model="deepseek-v4-flash", messages=[{"role":"user","content":"你是谁？一句话"}],
    max_tokens=100, extra_body={"thinking": {"type": "disabled"}}
)
print(f"content={r2.choices[0].message.content}")
print(f"usage={r2.usage}")