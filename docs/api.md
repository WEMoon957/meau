# 小菌点餐智能体 API 文档

**Base URL**: `http://<host>:3000`

---

## 1. 对话接口

用于向点餐智能体发送消息，获取 AI 推荐回复。

### POST /api/ai/chat

**请求体** (JSON)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 会话标识，最大 64 字符，同一会话的多次请求需保持一致 |
| `membership_level` | string | 否 | 会员等级，默认 `"普通会员"` |
| `user_message` | string | 是 | 用户消息内容，1~2000 字符 |

**membership_level 可选值**

| 值 | 说明 |
|------|------|
| `普通会员` | 默认，标准推荐 |
| `银卡会员` | 偏好多推招牌菜 |
| `金卡会员` | 优先特色菜、放宽品类限制 |
| `钻石会员` | 顶级菜品优先，品质体验为主 |

**请求示例**

```json
{
  "session_id": "abc123-def456",
  "membership_level": "金卡会员",
  "user_message": "2个人吃辣，推荐一下"
}
```

**响应** (200)

```json
{
  "code": 200,
  "msg": "success",
  "aimessage": "为您推荐以下菜品：\n\n--- 菌汤锅底 ---\n  1. 菌汤生态鸡子母锅  ￥68\n  ...\n合计：￥256\n\n推荐理由：...",
  "session_id": "abc123-def456"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 状态码，200 表示成功 |
| `msg` | string | 状态描述 |
| `aimessage` | string | AI 回复内容（含菜品名称、价格、辣度、合计、推荐理由） |
| `session_id` | string | 会话标识 |

**错误码**

| HTTP | code | 说明 |
|------|------|------|
| 429 | — | 请求频率过高（会话级或 IP 级限流） |
| 500 | — | 服务内部错误 |
| 502 | — | 模型服务不可用 |
| 503 | — | 服务繁忙/限流服务不可用 |

---

## 2. 重置会话

清除指定会话的历史上下文。

### POST /api/ai/reset

**请求体** (JSON)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 否 | 会话标识，默认 `"default"` |

**请求示例**

```json
{
  "session_id": "abc123-def456"
}
```

**响应** (200)

```json
{
  "code": 200,
  "msg": "success",
  "session_id": "abc123-def456"
}
```

---

## 3. 健康检查

供负载均衡探活，含数据库、Redis、知识库状态。

### GET /api/health

**响应** (200 / 503)

```json
{
  "code": 200,
  "msg": "ok",
  "data": {
    "dish_count": 120,
    "active_sessions": 5,
    "max_sessions": 500,
    "session_ttl_seconds": 1800,
    "max_concurrent_chats": 20,
    "concurrent_at_capacity": false,
    "rate_limits": {
      "per_session": "10/60s",
      "per_ip": "30/60s"
    },
    "dependencies": {
      "db": "ok",
      "redis": "ok",
      "kb": "ok"
    }
  }
}
```

当任一核心依赖（db/redis）不可用时返回 503，`msg` 为 `"degraded"`。

---

## 4. 服务信息

### GET /api/ai/info

**响应** (200)

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "name": "小味点餐智能体",
    "model": "qwen3.7-max",
    "tools": ["query_dish", "list_menu", "recommend_dishes", "search_dish_knowledge", "get_pairing_plan", "get_exclusion_rules", "get_fruit_allergen_info"],
    "capabilities": ["菜品问答", "智能推荐", "菜品知识库查询", "搭配方案推荐", "互斥规则提示", "水果过敏原查询"],
    "limits": {
      "session_ttl_seconds": 1800,
      "max_sessions": 500,
      "max_concurrent_chats": 20,
      "chat_rate_per_session": 10,
      "chat_rate_per_ip": 30,
      "chat_rate_window_seconds": 60
    }
  }
}
```

---

## 调用流程建议

```
1. 前端生成或获取 session_id
2. POST /api/ai/chat 发送用户消息
3. 展示响应的 aimessage 给用户
4. 用户继续对话时，使用同一个 session_id 再次调用 chat
5. 如需重新开始，调用 POST /api/ai/reset
```
