"""服务员话术数据生成模块

基于菜品数据自动生成各类话术，包括：
1. 菜品卖点话术 - 每道菜的推荐卖点
2. 场景应对话术 - 不同场景下的标准应对
3. 搭配推荐话术 - 菜品组合推荐话术
4. 异常处理话术 - 客诉、过敏、缺菜等应对

同时支持手动补充自定义话术（通过 add_custom_script）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from menu_data import Dish, get_all_dishes


# ======================== 话术类型 ========================
SCRIPT_TYPES = {
    "selling_point": "菜品卖点",
    "scene": "场景应对",
    "pairing": "搭配推荐",
    "exception": "异常处理",
    "custom": "自定义话术",
}


def generate_selling_point_scripts(dishes: list[Dish]) -> list[dict]:
    """生成菜品卖点话术"""
    scripts = []
    for d in dishes:
        # 招牌菜话术
        if d.is_signature:
            sig_tag = "招牌菜"
        else:
            sig_tag = "特色菜"

        # 构建卖点
        selling_points = []
        if d.spicy_level != "不辣":
            selling_points.append(f"辣度{d.spicy_level}，爱吃辣的顾客首选")
        if "高蛋白" in d.dietary_tags:
            selling_points.append("高蛋白营养健康")
        if "低脂" in d.dietary_tags:
            selling_points.append("低脂轻食，适合健身人群")
        if "素食" in d.dietary_tags:
            selling_points.append("素食友好")
        if d.allergens:
            allergen_text = "、".join(d.allergens)
            selling_points.append(f"含{allergen_text}，需提醒过敏顾客")
        else:
            selling_points.append("无常见过敏原，大众安全选择")

        # 人群适合
        if d.suitable_for:
            suitable_text = "、".join(d.suitable_for)
            selling_points.append(f"适合{suitable_text}")

        content = f"{d.name}（{d.category}，￥{d.price}）{sig_tag}推荐话术：{d.description}。{'；'.join(selling_points)}。"

        scripts.append({
            "id": f"sp_{d.id}",
            "type": "selling_point",
            "dish_name": d.name,
            "content": content,
            "metadata": {
                "type": "selling_point",
                "dish_name": d.name,
                "category": d.category,
                "price": d.price,
                "is_signature": d.is_signature,
            }
        })
    return scripts


def generate_scene_scripts() -> list[dict]:
    """生成场景应对话术"""
    scripts = [
        {
            "id": "scene_greet",
            "type": "scene",
            "dish_name": "",
            "content": "迎宾话术：您好，欢迎光临！请问几位用餐？这边请，我先为您倒杯水。我们今天有新鲜的水煮鱼和清蒸鲈鱼，都是非常受欢迎的招牌菜，您可以先看看菜单。",
            "metadata": {"type": "scene", "scene": "迎宾", "dish_name": ""}
        },
        {
            "id": "scene_recommend_signature",
            "type": "scene",
            "dish_name": "",
            "content": "推荐招牌菜话术：给您推荐几道我们的招牌菜——水煮鱼是特辣口味，鲜嫩鱼片浸在红油辣椒中，麻辣鲜香；红烧排骨酥烂入味，酱香浓郁，是下饭神器；清蒸鲈鱼新鲜清蒸，肉质细嫩，原汁原味。这三道菜是我们点单率最高的。",
            "metadata": {"type": "scene", "scene": "推荐招牌", "dish_name": ""}
        },
        {
            "id": "scene_spicy",
            "type": "scene",
            "dish_name": "",
            "content": "顾客要吃辣话术：如果您爱吃辣，强烈推荐水煮鱼，特辣口味，麻辣鲜香；宫保鸡丁微辣，鸡丁滑嫩配花生米，甜辣适口；麻婆豆腐中辣，嫩豆腐配牛肉末，麻辣鲜香。怕辣的话可以搭配酸梅汤解辣。",
            "metadata": {"type": "scene", "scene": "推荐辣菜", "dish_name": ""}
        },
        {
            "id": "scene_kids",
            "type": "scene",
            "dish_name": "",
            "content": "带小孩用餐话术：小朋友可以点番茄炒蛋，酸甜可口不辣，营养丰富；扬州炒饭料足味美，孩子都爱吃；再配一个紫菜蛋花汤，清淡鲜美。注意不要给孩子点水煮鱼和麻婆豆腐，太辣了。",
            "metadata": {"type": "scene", "scene": "儿童用餐", "dish_name": ""}
        },
        {
            "id": "scene_health",
            "type": "scene",
            "dish_name": "",
            "content": "健康饮食推荐话术：注重健康的话，凉拌木耳低脂素食，清脆爽口；蒜蓉西兰花清爽营养；清蒸鲈鱼高蛋白低脂，原汁原味。可以搭配番茄蛋花汤，营养均衡又不会太油腻。",
            "metadata": {"type": "scene", "scene": "健康推荐", "dish_name": ""}
        },
        {
            "id": "scene_cold_dish",
            "type": "scene",
            "dish_name": "",
            "content": "天冷推荐话术：天冷的话推荐红烧排骨，酥烂入味暖身子；酸萝卜老鸭汤，开胃滋补；再来份小炒黄牛肉，中辣暖胃。最后来碗热豆浆暖暖的。",
            "metadata": {"type": "scene", "scene": "天冷推荐", "dish_name": ""}
        },
        {
            "id": "scene_hot_dish",
            "type": "scene",
            "dish_name": "",
            "content": "天热推荐话术：天热的话先来个口水鸡开胃，凉拌木耳清脆爽口；主食来份蛋炒饭清淡不腻；配一杯酸梅汤或者柠檬蜂蜜水，酸甜解暑。",
            "metadata": {"type": "scene", "scene": "天热推荐", "dish_name": ""}
        },
        {
            "id": "scene_party",
            "type": "scene",
            "dish_name": "",
            "content": "聚餐推荐话术：聚餐的话建议荤素搭配——凉菜来口水鸡和蒜泥白肉，热菜水煮鱼、红烧排骨、小炒黄牛肉，再加个蒜蓉西兰花解腻，汤品来酸萝卜老鸭汤，最后扬州炒饭管饱。人均六七十左右。",
            "metadata": {"type": "scene", "scene": "聚餐推荐", "dish_name": ""}
        },
        {
            "id": "scene_solo",
            "type": "scene",
            "dish_name": "",
            "content": "一人食推荐话术：一个人吃的话，来份宫保鸡丁配蛋炒饭，有荤有素刚好；或者担担面配番茄蛋花汤，简单又满足。不用点太多，不够再加。",
            "metadata": {"type": "scene", "scene": "一人食", "dish_name": ""}
        },
        {
            "id": "scene_upsell",
            "type": "scene",
            "dish_name": "",
            "content": "追加推荐话术：您点的菜差不多够了，要不要再来个汤？酸萝卜老鸭汤是我们的招牌汤品，开胃滋补。或者来份甜点，芒果布丁和红豆双皮奶都很受欢迎。",
            "metadata": {"type": "scene", "scene": "追加上单", "dish_name": ""}
        },
    ]
    return scripts


def generate_pairing_scripts(dishes: list[Dish]) -> list[dict]:
    """生成搭配推荐话术"""
    scripts = []
    pairings = [
        {
            "id": "pair_spicy_cool",
            "content": "辣菜解辣搭配：点了水煮鱼或麻婆豆腐这类辣菜，建议搭配酸梅汤或柠檬蜂蜜水，酸甜解辣又解腻。口感上先辣后甜，层次丰富。",
            "dishes": "水煮鱼,麻婆豆腐,酸梅汤,柠檬蜂蜜水",
        },
        {
            "id": "pair_meat_veg",
            "content": "荤素搭配推荐：红烧排骨配蒜蓉西兰花，一荤一素营养均衡；宫保鸡丁配凉拌木耳，口感丰富不油腻。中式餐饮讲究荤素搭配，这样吃最舒服。",
            "dishes": "红烧排骨,蒜蓉西兰花,宫保鸡丁,凉拌木耳",
        },
        {
            "id": "pair_rice_soup",
            "content": "主食配汤推荐：扬州炒饭配酸萝卜老鸭汤，炒饭料足味美，老鸭汤开胃滋补，干稀搭配最养胃。蛋炒饭配番茄蛋花汤也是经典组合，简单温馨。",
            "dishes": "扬州炒饭,酸萝卜老鸭汤,蛋炒饭,番茄蛋花汤",
        },
        {
            "id": "pair_party_set",
            "content": "四人聚餐黄金搭配：口水鸡+蒜泥白肉（凉菜开胃）、水煮鱼+红烧排骨+清蒸鲈鱼（热菜硬菜）、酸萝卜老鸭汤（暖胃汤品）、扬州炒饭（主食收尾）、酸梅汤（解腻饮品）。荤素凉热齐全，人均约70元。",
            "dishes": "口水鸡,蒜泥白肉,水煮鱼,红烧排骨,清蒸鲈鱼,酸萝卜老鸭汤,扬州炒饭,酸梅汤",
        },
        {
            "id": "pair_date_set",
            "content": "情侣约会搭配：糖醋里脊（酸甜浪漫）+清蒸鲈鱼（鲜嫩精致）+桂花糕（甜蜜收尾）。两菜一甜点，浪漫又不浪费。",
            "dishes": "糖醋里脊,清蒸鲈鱼,桂花糕",
        },
    ]
    for p in pairings:
        scripts.append({
            "id": p["id"],
            "type": "pairing",
            "dish_name": p["dishes"],
            "content": p["content"],
            "metadata": {"type": "pairing", "dish_name": p["dishes"]}
        })
    return scripts


def generate_exception_scripts() -> list[dict]:
    """生成异常处理话术"""
    scripts = [
        {
            "id": "exc_allergy",
            "type": "exception",
            "dish_name": "",
            "content": "过敏应对话术：请问您对什么食物过敏？我们的口水鸡和老醋花生含有花生，蒜泥白肉也含有花生碎。如果您对花生过敏，我为您推荐不含花生的菜品，比如凉拌木耳、番茄炒蛋、清蒸鲈鱼等，这些都没有常见过敏原。",
            "metadata": {"type": "exception", "scene": "过敏应对", "dish_name": ""}
        },
        {
            "id": "exc_too_spicy",
            "type": "exception",
            "dish_name": "",
            "content": "嫌辣应对话术：不好意思这道菜确实比较辣，我给您倒杯酸梅汤解解辣。下次可以点微辣的宫保鸡丁或者不辣的番茄炒蛋。要不要我帮您加一份米饭，配着吃会好一些。",
            "metadata": {"type": "exception", "scene": "嫌辣应对", "dish_name": ""}
        },
        {
            "id": "exc_not_available",
            "type": "exception",
            "dish_name": "",
            "content": "缺菜应对话术：非常抱歉，这道菜今天的食材刚好卖完了。我给您推荐一道类似的——（根据缺的菜品推荐同类菜品）。这道菜也很受欢迎，您要不要试试？",
            "metadata": {"type": "exception", "scene": "缺菜应对", "dish_name": ""}
        },
        {
            "id": "exc_wait_long",
            "type": "exception",
            "dish_name": "",
            "content": "等久应对话术：实在抱歉让您久等了，您的菜马上就好。先给您续杯水，这道菜需要现做保证口感，马上就端上来。感谢您的耐心等待。",
            "metadata": {"type": "exception", "scene": "等久应对", "dish_name": ""}
        },
        {
            "id": "exc_complaint",
            "type": "exception",
            "dish_name": "",
            "content": "客诉应对话术：非常抱歉给您带来不好的体验，您反映的问题我记下了。这道菜我帮您退掉/换一份，请您稍等。感谢您的反馈，我们会改进。请问还有什么需要我做的吗？",
            "metadata": {"type": "exception", "scene": "客诉应对", "dish_name": ""}
        },
        {
            "id": "exc_bill",
            "type": "exception",
            "dish_name": "",
            "content": "结账话术：您好，这是您的账单，一共X道菜，合计￥XX。请问您是现金、扫码还是刷卡？感谢您的光临，欢迎下次再来！",
            "metadata": {"type": "exception", "scene": "结账", "dish_name": ""}
        },
    ]
    return scripts


def generate_all_scripts(dishes: list[Dish] = None) -> list[dict]:
    """生成全部话术数据"""
    if dishes is None:
        dishes = get_all_dishes()

    scripts = []
    scripts.extend(generate_selling_point_scripts(dishes))
    scripts.extend(generate_scene_scripts())
    scripts.extend(generate_pairing_scripts(dishes))
    scripts.extend(generate_exception_scripts())
    return scripts


def add_custom_script(content: str, script_type: str = "custom",
                      dish_name: str = "", scene: str = "") -> dict:
    """添加自定义话术（预留接口，供后续手动补充）

    Args:
        content: 话术内容
        script_type: 话术类型
        dish_name: 关联菜品名（可选）
        scene: 场景描述（可选）

    Returns:
        话术字典，可用于添加到向量库
    """
    import hashlib
    script_id = "custom_" + hashlib.md5(content.encode()).hexdigest()[:8]
    metadata = {"type": script_type, "dish_name": dish_name}
    if scene:
        metadata["scene"] = scene
    return {
        "id": script_id,
        "type": script_type,
        "dish_name": dish_name,
        "content": content,
        "metadata": metadata,
    }


if __name__ == "__main__":
    # 测试：生成并打印话术
    all_scripts = generate_all_scripts()
    print(f"共生成 {len(all_scripts)} 条话术")
    print(f"  - 菜品卖点: {sum(1 for s in all_scripts if s['type'] == 'selling_point')} 条")
    print(f"  - 场景应对: {sum(1 for s in all_scripts if s['type'] == 'scene')} 条")
    print(f"  - 搭配推荐: {sum(1 for s in all_scripts if s['type'] == 'pairing')} 条")
    print(f"  - 异常处理: {sum(1 for s in all_scripts if s['type'] == 'exception')} 条")
    print()
    print("示例话术：")
    for s in all_scripts[:3]:
        print(f"\n[{s['id']}] {s['content'][:80]}...")
