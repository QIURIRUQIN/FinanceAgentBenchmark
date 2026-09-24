# 金融QA轨迹分析 — 方法论文档

本文档记录对金融QA评测（SEC EDGAR 财报问答）中不同模型轨迹进行深度分析的方法论。涵盖数据格式、分析步骤、KPI定义、错误模式分类、输出要点及常见陷阱。

---

## 1. 数据源与字段含义

分析涉及三个数据源：

### 1.1 `selected_tasks.csv` — 题目定义

| 列 | 含义 |
|---|---|
| `Question` | 题目文本 |
| `Answer` | 标准答案（gold answer） |
| `Question Type` | 题目类别，共6类：`Adjustments`, `Beat or Miss`, `Financial Modeling / Projections`, `Numerical Reasoning`, `Simple retrieval - Qualitative`, `Simple retrieval - Quantitative` |
| `Rubric` | JSON 格式的评分标准，每项含 `operator` 和 `criteria` |

### 1.2 `result.json`（每道题的轨迹文件）— 模型运行结果

```
finance-agent-main/logs/finance/<model>/<run>/<qNNN>/result.json
```

| 顶层字段 | 类型 | 含义 |
|---|---|---|
| `final_answer` | string | 模型提交的最终答案 |
| `success` | bool | Agent 是否正常结束（不代表答案正确） |
| `total_turns` | int | 交互轮次（LLM 调用次数） |
| `tool_calls_count` | int | 工具调用总次数 |
| `error_count` | int | 工具调用失败次数 |
| `final_duration_seconds` | float | 总运行时间（秒） |
| `tool_usage` | dict | 各工具调用次数统计，如 `{"web_search": 3, "edgar_search": 5, ...}` |
| `turns` | list | 每轮的详情数组 |

**`turns[i]` 结构：**

| 字段 | 含义 |
|---|---|
| `tool_calls` | 本轮工具调用列表 |
| `duration_seconds` | 本轮耗时 |
| `metadata` | token/cost 信息 |

**`tool_calls[j]` 结构：**

| 字段 | 含义 |
|---|---|
| `tool_name` | 工具名：`web_search`, `edgar_search`, `parse_html_page`, `retrieve_information`, `submit_final_result` |
| `output_length` | 工具返回内容的字符数。≤2 表示空结果 |
| `success` | 工具调用是否成功 |
| `error` | 错误详情（如有） |

### 1.3 系统提示词

```python
# finance_agent/prompt.py 第5行
You should answer all questions as if the current date is April 07, 2025.
```

**⚠️ 这是影响最大的配置项**：提示词中的固定日期直接决定了模型是否会"拒绝回答"那些在真实评测数据中已经存在的文件。

---

## 2. 分析流程（5 步）

### Step 1: 解析题目 & 构建轨迹索引

```python
questions = parse_csv("selected_tasks.csv")   # → [{qid, question, answer, q_type, rubric}]
qmap = build_result_map("<model_log_dir>")     # → {q001: "path/to/result.json", ...}
```

**注意**：不是每道题都有轨迹。缺失的题目（`is_missing=True`）必须纳入 KPI 计算的分母中（见第 3 节陷阱）。

### Step 2: 轨迹摘要提取

对每道题的 `result.json` 调用 `summarize_trajectory()`，从 turns/tool_calls 中提取：

| 指标 | 来源 | 用途 |
|---|---|---|
| `search_count` | 计数 `web_search` 调用 | 判断搜索密度 |
| `edgar_count` | 计数 `edgar_search` 调用 | 判断是否直接查SEC |
| `parse_count` | 计数 `parse_html_page` 调用 | 判断是否深度阅读文件 |
| `retrieve_count` | 计数 `retrieve_information` 调用 | 判断是否精确提取信息 |
| `empty_results` | `output_length <= 2` 的搜索 | 诊断搜索策略僵化 |
| `tool_failures` | `success == False` 的工具调用 | 诊断工具可靠性问题 |
| `submit_turn` | 首次 `submit_final_result` 的轮次 | 判断是否过早提交 |

### Step 3: Rubric 评分

**核心方法**：将模型的 `final_answer` 与 `rubric` 中的每条 `criteria` 做关键词匹配。

```python
def score_rubric(model_answer, rubric_str):
    rubric = json.loads(rubric_str)          # 解析 JSON
    for item in rubric:
        keys = _extract_keys(item['criteria'])  # 从评分标准中提取关键词
        matches = count_matches(keys, model_answer.lower())  # 在答案中搜索
        # 按 operator 判定通过/失败
```

- 每个 rubric item 是一个独立的评分项（1分）
- `score = passed / total` → 百分比
- `contradiction` 类型的 item 不参与评分

**不是精确匹配**，是关键词命中 + 阈值判定。

### Step 4: 错误模式分类

对每道题未通过的 rubric 项，归类到已知错误模式：

| 模式 | 触发条件 |
|---|---|
| **WrongAnswer** | `equals`/`contains` 类 operator 且 matches=0 |
| **IncompleteSet** | `contains`/`AND` 类 operator 且部分匹配 |
| **MissingSpecificValue** | 框架正确但缺少关键数字/名称 |
| **ExtraWrongInfo** | `not_contains` 类 operator 且 matches>0 |
| **WrongSourceFiling** | 搜索量大(>5)但答案仍错 → 检索了错误的文件 |
| **ConceptualMisunderstanding** | 搜索量极小(≤3)但答案全错 → 根本性理解偏差 |
| **WrongFormat** | 输出格式不符合题目要求 |

**轨迹回溯增强**：结合 `tool_failures`、`empty_results`、`submit_turn` 等轨迹特征来辅助判定根因。

### Step 5: 生成 HTML 报告

包含 6 个部分（见第 4 节输出要点）。

---

## 3. ⚠️ 常见陷阱（务必注意）

### 陷阱 1: 平均 Rubric 的分母必须包含失败题和缺失题

**错误做法**：
```python
# 仅统计"成功"的题目 → 分母偏小，得分虚高
scored = [d for d in data if d['success']]
avg = sum(d['rubric_pct'] for d in scored) / len(scored)
```

**正确做法**：
```python
# 所有有 rubric_max > 0 的题目都应纳入分母
# 包括 status=success/fail 以及 rubric_pct=0 的题目
scored = [d for d in all_data if d['rubric_max'] > 0]
avg = sum(d['rubric_pct'] for d in scored) / len(scored)
```

> 实际案例：qwen3.6-27b 报告的 rubric 平均 "4.5%"，但正确值应为 **43.2%**（差了近10倍）。
> 12 道 rubric=0% 的失败题被漏掉了。

### 陷阱 2: 成功率的分母也需包含缺失轨迹的题目

```python
# 缺失轨迹的题目（is_missing=True）也应纳入分母
success_rate = successful / total_all_questions * 100
#                                      ↑ 包含 is_missing 的题目
```

### 陷阱 3: 满分率的分子必须是 rubric_pct ≥ 99%（不是100%）

由于关键词匹配不是精确比对，实际满分可能出现 99.x% 的情况。

```python
full_scores = sum(1 for d in scored if d['rubric_pct'] >= 99)
```

### 陷阱 4: 系统提示词中的日期是硬编码的

分析报告中必须标注模型的系统提示日期。如果该日期早于评测数据的时间范围，会出现大量"时效性误判"（模型拒绝回答"未来"问题），这是提示词问题而非模型能力问题。

---

## 4. 输出报告结构

HTML 报告应包含以下 6 个部分，顺序如下：

### 4.1 KPI 卡片行

8 个核心指标，单行排列：

> 成功率 | 平均Rubric得分率 | 满分率 | 平均运行时间(含中位数) | 平均交互轮次 | 平均工具调用 | ⭐零失误题目数 | ⭐有扣分题目数

⭐ 标注的是除基本指标外建议补充的维度。

### 4.2 图表区（4 图）

| 图表 | 类型 | 内容 |
|---|---|---|
| c1 | 分组柱状图 | 按题型：成功率 + Rubric 得分率 |
| c2 | 双Y轴柱状图 | 按题型：平均轮次 + 运行时间 |
| c3 | 环形图 | 错误模式分布 |
| c4 | 散点图 | Rubric 得分 vs 轮次（按题型着色） |

### 4.3 按问题类别统计表

| 类别 | 题数 | 成功率 | Rubric得分率 | 满分率 | 耗时 | 轮次 | 工具 | 常见失分模式 |
|---|---|---|---|---|---|---|---|

### 4.4 ⭐ 错误模式归因分析（核心新增）

**不做简单的"失败模式分布"（如 ToolFailureCascade 2次/EmptyRetrievalLoop 2次），
而是对"成功但未满分"(rubric<100%)的题目做深度归因分析。**

分析方法：
1. 阅读所有非满分题目的 `final_answer` 和预期答案的差异
2. 结合轨迹数据（搜索次数、空结果、轮次、工具失败）找根因
3. 将题目分组到有意义的错误模式中
4. 每种模式：描述错误表现 → 分析根因 → 举例说明

**推荐的错误模式分类（非 operator 层面，而是行为层面）：**

| 模式 | 核心特征 | 影响度 |
|---|---|---|
| **时效性误判** | 模型认为日期在未来，拒绝回答 | 最大（~15题, Rubric中位≈15%） |
| **时态错位** | 使用了错误年份/季度的数据 | 大（~8题, Rubric中位≈40%） |
| **搜索策略僵化** | 大量空结果但不调整查询 | 中（~10题, Rubric中位≈50%） |
| **Adjustments 列举不全** | 现金流调整项多列/漏列 | 集中（4题全受影响） |
| **信息存在性误判** | 声称指标"不存在"但实际有 | 少但严重（~3题） |
| **多步骤推理链断裂** | 财务建模中间步骤失败 | 严重（Rubric中位≈20%） |

### 4.5 改善建议 / 关键洞察

基于分析结果，给出 3-5 条操作性强的改进建议。
优先指出**系统性问题**（如提示词日期）而非模型能力问题。

### 4.6 逐题分析卡片

每道题一张卡片，包含：
- 题目ID、类型、Rubric得分、成功/失败状态
- 模型回答 vs 预期答案（可折叠）
- 轨迹诊断（如有扣分）：未通过的 rubric 项 + 根因分析

---

## 5. 设计编码规范

| 要素 | 规范 |
|---|---|
| 配色 | 暗色主题 `#0b1120` 背景，`#1e293b` 卡片，`#334155` 边框 |
| 状态色 | 🟢 `#22c55e` 满分/成功，🟡 `#eab308` 中等，🔴 `#ef4444` 失败 |
| 图表库 | Chart.js 4.x CDN |
| 字体 | `PingFang SC`, `Microsoft YaHei`, system sans-serif |
| 宽度 | `max-width: 1500px`，支持响应式 `grid-template-columns: repeat(auto-fit, ...)` |
| 语言 | 中文为主，技术术语保留英文 |

---

## 6. 脚本设计模式

参考 `analyze_glm.py` 的结构：

```
1. 解析 CSV 题目定义
2. 扫描轨迹目录，构建 qid → result.json 的映射
3. 逐题处理：
   a. summarize_trajectory()     → 轨迹摘要
   b. score_rubric()             → Rubric 评分
   c. classify_failed_criteria() → 错误模式归类
   d. trajectory_root_cause()    → 轨迹回溯诊断
4. 聚合统计（全局 + 按类别）
5. 生成 HTML（内联 CSS + Chart.js）
```

---

## 7. 扩展：多模型对比分析

当需要对多个模型进行对比时，额外注意：

1. **确保分析脚本一致** — 同一套 rubric 评分逻辑
2. **检查系统提示差异** — 日期、工具描述、指令是否一致
3. **统一分母** — 所有模型都使用相同的题目集作为分母
4. **对比维度**：
   - 成功率 vs Rubric得分率（反映"能做对"和"能拿满"的差距）
   - 错误模式分布对比（暴露各模型的弱项）
   - 耗时/轮次效率对比
   - 工具使用策略对比（是否偏好 web_search vs edgar_search）
