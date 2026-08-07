"""点餐智能体 - 主程序入口

启动交互式对话，实现：用户提问 -> AI理解 -> 推荐菜品 -> 辅助下单
"""

import asyncio
import os
import sys
import io

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Windows终端兼容：强制UTF-8输出
if sys.platform == "win32":
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


from agent import OrderingAgent


BANNER = """
========================================
          小味点餐智能体
========================================
  我是您的专属点餐助手，可以帮您：
  * 查询菜品信息（辣度、过敏原、适合人群）
  * 智能推荐菜品（按口味、人数、健康需求）
  * 加入购物车并完成下单
  * 生成服务员话术（迎宾、推荐、客诉应对等）

  输入[菜单]查看全部菜品
  输入[话术]查看话术功能说明
  输入[退出]结束对话
========================================
"""


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("未检测到 OPENAI_API_KEY 环境变量")
        api_key = input("请输入您的 OpenAI API Key: ").strip()
        if not api_key:
            print("API Key 不能为空，程序退出。")
            return

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    base_url = os.environ.get("OPENAI_BASE_URL", "")

    agent = OrderingAgent(api_key=api_key, model=model, base_url=base_url)

    print(BANNER)

    while True:
        try:
            # CLI 单用户场景，input 阻塞事件循环可接受
            user_input = input("\n您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n感谢使用，再见！")
            break

        if not user_input:
            continue

        if user_input in ("退出", "exit", "quit", "q"):
            print("感谢使用小味点餐智能体，再见！")
            break

        if user_input in ("清空对话", "重置", "reset"):
            agent.reset()
            print("对话已重置。")
            continue

        if user_input in ("菜单", "menu"):
            from tools import list_menu
            print(f"\n小味: {await list_menu.ainvoke({'category': ''})}")
            continue

        if user_input == "话术":
            print("""
========================================
        服务员话术功能说明
========================================
  话术向量库已构建，支持以下功能：

  1. 场景话术检索
     直接描述场景，如：
     - "顾客带小孩来怎么推荐"
     - "顾客嫌菜太辣怎么应对"
     - "四个人聚餐怎么推荐"
     - "怎么推荐招牌菜"

  2. 话术类型说明
     - 菜品卖点(selling_point): 每道菜的推荐卖点
     - 场景应对(scene): 迎宾、推荐、结账等标准话术
     - 搭配推荐(pairing): 菜品组合推荐话术
     - 异常处理(exception): 客诉、过敏、缺菜应对

  3. 添加自定义话术
     如："添加话术：夏季推荐冰镇酸梅汤，解暑解腻"
========================================
""")
            continue

        try:
            reply = await agent.achat(user_input)
            print(f"\n小味: {reply}")
        except Exception as e:
            print(f"\n处理请求时出错: {e}")
            print("请稍后重试。")


if __name__ == "__main__":
    asyncio.run(main())
