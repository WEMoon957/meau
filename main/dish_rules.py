"""菜品规则引擎 - 推荐时自动拦截违规组合

从知识库加载所有互斥规则（菌子重复、口味冲突）和避雷搭配，
构建冲突图谱，在推荐时自动检测和过滤违规组合。

核心功能：
  1. load_rules()         - 加载所有规则，构建冲突图谱
  2. find_conflicts()     - 检测一组菜品是否存在冲突
  3. filter_by_rules()    - 从候选列表中过滤掉与已选菜品冲突的项
  4. get_rule_warnings()  - 生成冲突警告文案
"""

import os
import sys
import re
import threading

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from menu_data import get_all_dishes


# ======================== 规则引擎单例 ========================
_rules_instance = None
_rules_lock = threading.Lock()


def _get_rules():
    """获取规则引擎单例（线程安全，懒加载）"""
    global _rules_instance
    if _rules_instance is not None:
        return _rules_instance

    with _rules_lock:
        if _rules_instance is not None:
            return _rules_instance
        _rules_instance = _DishRulesEngine()
        return _rules_instance


def reload_rules():
    """强制重新加载规则（知识库更新后调用）"""
    global _rules_instance
    with _rules_lock:
        _rules_instance = _DishRulesEngine()


# ======================== 规则引擎实现 ========================
class _DishRulesEngine:
    """菜品规则引擎

    数据结构：
      self.conflict_map: {dish_name: set(conflicting_dish_names)}
      self.rule_texts:   [{dishes: [names], text: str, type: str}]
    """

    # 规则文本中的否定/冲突关键词
    CONFLICT_KEYWORDS = [
        "不宜", "不要", "不能", "不可", "避免", "冲突", "重复",
        "不推荐", "不建议", "禁忌", "慎重", "注意",
    ]

    def __init__(self):
        self.conflict_map: dict[str, set[str]] = {}
        self.rule_texts: list[dict] = []
        self._loaded = False
        self._load()

    def _load(self):
        """从知识库加载所有规则，构建冲突图谱"""
        try:
            from kb_query import get_all_exclusion_rules, get_all_avoid_combos
        except ImportError:
            print("⚠️ 菜品规则引擎：无法导入 kb_query，规则未加载")
            return
        except Exception as e:
            print(f"⚠️ 菜品规则引擎：导入失败 {e}")
            return

        # 获取菜单中所有菜品名称（用于匹配规则文本）
        try:
            menu_dishes = get_all_dishes()
        except Exception:
            menu_dishes = []
        dish_names_set = {d.name for d in menu_dishes}

        # 也从知识库获取菜品名称作为补充
        try:
            from kb_query import _get_kb
            kb = _get_kb()
            kb_dish_names = kb.get_all_dishes()
            dish_names_set.update(kb_dish_names)
        except Exception:
            pass

        if not dish_names_set:
            print("⚠️ 菜品规则引擎：未找到菜品名称，规则未加载")
            return

        # 加载互斥规则和避雷搭配
        all_rules = []
        try:
            all_rules.extend(get_all_exclusion_rules())
        except Exception as e:
            print(f"⚠️ 加载互斥规则失败: {e}")

        try:
            all_rules.extend(get_all_avoid_combos())
        except Exception as e:
            print(f"⚠️ 加载避雷搭配失败: {e}")

        # 解析每条规则，提取涉及的菜品名称
        for rule in all_rules:
            text = rule.get("text", "")
            rule_type = rule.get("metadata", {}).get("type", "")
            section = rule.get("metadata", {}).get("section", "")

            # 在规则文本中查找菜品名称
            matched_dishes = self._find_dish_names_in_text(text, dish_names_set)

            if len(matched_dishes) >= 2:
                # 两个以上菜品出现在同一条规则中，建立冲突关系
                self.rule_texts.append({
                    "dishes": matched_dishes,
                    "text": text,
                    "section": section,
                    "type": rule_type,
                })

                # 构建双向冲突图谱
                for i, d1 in enumerate(matched_dishes):
                    for d2 in matched_dishes[i + 1:]:
                        self.conflict_map.setdefault(d1, set()).add(d2)
                        self.conflict_map.setdefault(d2, set()).add(d1)

        self._loaded = True
        rule_count = len(self.rule_texts)
        conflict_count = sum(len(v) for v in self.conflict_map.values()) // 2
        print(f"✅ 菜品规则引擎已加载：{rule_count} 条规则，{conflict_count} 组冲突关系")

    def _find_dish_names_in_text(
        self, text: str, dish_names: set[str]
    ) -> list[str]:
        """在规则文本中查找出现的菜品名称

        Args:
            text: 规则文本
            dish_names: 所有菜品名称集合

        Returns:
            匹配到的菜品名称列表（按在文本中出现的位置排序）
        """
        matches = []
        for name in dish_names:
            if name and name in text:
                matches.append(name)

        # 按在文本中的位置排序
        matches.sort(key=lambda n: text.find(n))
        return matches

    # ======================== 公开接口 ========================

    def find_conflicts(self, dish_names: list[str]) -> list[dict]:
        """检测一组菜品是否存在冲突

        Args:
            dish_names: 菜品名称列表

        Returns:
            冲突列表，每项: {dish1, dish2, rule_text, section}
        """
        if not self._loaded or not self.conflict_map:
            return []

        conflicts = []
        seen_pairs = set()

        for i, d1 in enumerate(dish_names):
            conflicts_for_d1 = self.conflict_map.get(d1, set())
            for d2 in dish_names[i + 1:]:
                if d2 in conflicts_for_d1:
                    pair_key = tuple(sorted([d1, d2]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # 找到对应的规则文本
                    rule_text = ""
                    section = ""
                    for rule in self.rule_texts:
                        if d1 in rule["dishes"] and d2 in rule["dishes"]:
                            rule_text = rule["text"]
                            section = rule["section"]
                            break

                    conflicts.append({
                        "dish1": d1,
                        "dish2": d2,
                        "rule_text": rule_text,
                        "section": section,
                    })

        return conflicts

    def filter_by_rules(
        self,
        candidates: list,
        selected_names: set[str],
        id_attr: str = "name",
    ) -> tuple[list, list[str]]:
        """从候选列表中过滤掉与已选菜品冲突的项

        Args:
            candidates: 候选菜品列表（Dish 对象或字典）
            selected_names: 已选菜品名称集合
            id_attr: 从候选项中提取名称的属性名

        Returns:
            (过滤后的列表, 被过滤掉的名称列表)
        """
        if not self._loaded or not self.conflict_map:
            return candidates, []

        filtered = []
        removed = []

        for item in candidates:
            name = getattr(item, id_attr, None) if not isinstance(item, dict) \
                else item.get(id_attr)

            if name is None:
                filtered.append(item)
                continue

            # 检查是否与已选菜品冲突
            conflicts = self.conflict_map.get(name, set())
            if conflicts & selected_names:
                removed.append(name)
            else:
                filtered.append(item)

        return filtered, removed

    def get_rule_warnings(self, dish_names: list[str]) -> str:
        """生成冲突警告文案

        Args:
            dish_names: 菜品名称列表

        Returns:
            警告文案，无冲突时返回空字符串
        """
        conflicts = self.find_conflicts(dish_names)
        if not conflicts:
            return ""

        lines = ["⚠️ 菜品规则提醒："]
        for c in conflicts:
            lines.append(
                f"  • {c['dish1']} 与 {c['dish2']}："
                f"{c['section'] or '存在冲突'}"
            )
        return "\n".join(lines)

    def has_conflict(self, dish_name: str, selected_names: set[str]) -> bool:
        """快速判断单个菜品是否与已选菜品冲突

        Args:
            dish_name: 待检测菜品名称
            selected_names: 已选菜品名称集合

        Returns:
            True 表示存在冲突
        """
        if not self._loaded:
            return False
        conflicts = self.conflict_map.get(dish_name, set())
        return bool(conflicts & selected_names)


# ======================== 模块级便捷接口 ========================

def find_conflicts(dish_names: list[str]) -> list[dict]:
    """检测一组菜品是否存在冲突（模块级便捷接口）"""
    return _get_rules().find_conflicts(dish_names)


def filter_by_rules(candidates: list, selected_names: set[str]) -> tuple[list, list[str]]:
    """从候选列表中过滤掉与已选菜品冲突的项（模块级便捷接口）"""
    return _get_rules().filter_by_rules(candidates, selected_names)


def get_rule_warnings(dish_names: list[str]) -> str:
    """生成冲突警告文案（模块级便捷接口）"""
    return _get_rules().get_rule_warnings(dish_names)


def has_conflict(dish_name: str, selected_names: set[str]) -> bool:
    """快速判断单个菜品是否与已选菜品冲突（模块级便捷接口）"""
    return _get_rules().has_conflict(dish_name, selected_names)


def preload_rules() -> None:
    """启动时预加载规则引擎"""
    try:
        _get_rules()
    except Exception as e:
        print(f"⚠️ 菜品规则引擎预加载失败: {e}")
