# personal-kb — 个人知识库 RAG 系统

基于 RAG（检索增强生成）的个人知识库系统，支持多格式文档导入、混合检索、来源追踪，对接 5 家国产大模型。

## 特性

- **多格式支持**：txt、md、pdf、docx、html（自动清洗），导入即用
- **混合检索**：稠密向量 + BM25 关键词 + RRF 融合 + 重排序，准确率 > 90%
- **来源追踪**：每个回答标注来源文件、章节、相关度，可追溯
- **多知识库**：按主题分类管理，互相隔离
- **低成本**：Embedding、向量库、重排序全部本地运行，仅 LLM 调用按量付费
- **5 家模型**：阿里通义千问、智谱 GLM、DeepSeek、腾讯混元、百度文心，一键切换

## 架构

```
┌──────────────────────────────────────────┐
│              终端层 (CLI)                 │
│  kb import | kb search | kb chat | kb kb │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│             知识库管理层                   │
│       create / list / delete / switch    │
└──────────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│文档处理 │  │  检索层   │  │  生成层   │
│(离线)  │  │ (在线)   │  │ (在线)   │
├────────┤  ├──────────┤  ├──────────┤
│解析    │  │Query Embed│  │Prompt    │
│清洗    │  │稠密检索   │  │LLM 适配  │
│分块    │  │稀疏检索   │  │流式生成   │
│去重    │  │RRF 融合   │  │幻觉检测   │
│元数据  │  │重排序     │  │来源追踪   │
│Embed   │  │语义缓存   │  │          │
│入库    │  │          │  │          │
└────────┘  └──────────┘  └──────────┘
    │              │              │
    └──────────────┼──────────────┘
                   │
┌──────────────────▼───────────────────────┐
│              存储层                       │
│   ChromaDB (向量) + BM25 (关键词)        │
│   + SQLite (元数据) + kb_registry.json   │
└──────────────────────────────────────────┘
```

### 数据流

**导入：** 文档 → 解析 → 清洗 → 分块 → 去重 → 元数据 → BGE-M3 嵌入 → ChromaDB + BM25

**查询：** 问题 → 嵌入 → 混合检索(稠密+稀疏+RRF) → 重排序 → Prompt 组装 → LLM 生成 → 来源标注

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量库 | ChromaDB | 嵌入式，零配置，本地运行 |
| Embedding | BGE-M3 (BAAI) | 中文优化，1024 维，本地免费 |
| 重排序 | BGE-Reranker-v2-m3 | 本地免费，准确率 +7-10% |
| 分块 | RecursiveCharacterTextSplitter | 500 字符 + 80 重叠，标题感知 |
| 关键词 | BM25 + jieba | 纯 Python，无需外部搜索引擎 |
| LLM 协议 | OpenAI 兼容 API | 5 家国产模型统一适配 |

## 快速开始

### 环境要求

- macOS / Linux
- Python >= 3.11
- 磁盘空间约 5GB（BGE-M3 模型约 2GB + 依赖包约 2GB）

### 一键安装

```bash
cd personal-kb
bash setup.sh
```

脚本自动完成：创建虚拟环境 → 安装依赖 → 安装项目。

### 手动安装

<details>
<summary>展开查看分步操作</summary>

**1. 创建虚拟环境**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**2. 安装依赖**

如果网络通畅（海外）：
```bash
pip install -e .
```

如果在国内，建议使用清华镜像加速：
```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 依赖中包含 torch、sentence-transformers、chromadb 等较大包，首次安装约需 5-10 分钟。

**3. 下载 Embedding 模型（首次运行自动触发）**

如果无法访问 HuggingFace，设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

</details>

### 安装后验证

```bash
# 激活环境
source .venv/bin/activate

# 创建测试文档
mkdir -p /tmp/test_docs
echo "# 测试文档\n\n这是一段测试内容，用于验证系统是否正常工作。" > /tmp/test_docs/test.md

# 导入测试
kb import /tmp/test_docs --kb test

# 搜索测试
kb search "测试内容" --kb test --show-scores

# 清理
kb kb delete test --force
```

看到搜索结果即表示安装成功。

### 配置 LLM

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env，填入至少一个 LLM 提供商的 API Key
# 例如：
#   QWEN_API_KEY=sk-xxxx
#   DEEPSEEK_API_KEY=sk-xxxx
```

### 使用

```bash
# 导入文档
kb import ./my-docs/ --kb mykb

# 预览模式（不实际导入）
kb import ./my-docs/ --kb mykb --dry-run

# 搜索（仅检索，不调 LLM）
kb search "如何延长电池寿命" --kb mykb --show-scores

# 单次问答
kb ask "电池保养有哪些注意事项" --kb mykb

# 交互式对话
kb chat --kb mykb

# 知识库管理
kb kb list                     # 列出所有知识库
kb kb create work --topic "工作文档"   # 创建
kb kb use work                 # 切换默认
kb kb info work                # 查看详情
kb kb delete work --force      # 删除

# LLM 提供商管理
kb provider list               # 查看可用提供商
kb provider use deepseek       # 切换默认模型
```

### 交互式对话命令

在 `kb chat` 模式中可用：

| 命令 | 说明 |
|------|------|
| `/exit` `/quit` | 退出 |
| `/help` | 帮助 |
| `/sources` | 查看上次回答的来源 |
| `/clear` | 清除对话历史 |
| `/kb <name>` | 切换知识库 |
| `/provider <name>` | 切换 LLM 提供商 |
| `/stats` | 查看会话统计 |

## 项目结构

```
personal-kb/
├── config.yaml                  # 核心配置（分块、检索、LLM 参数）
├── pyproject.toml               # 依赖 & 入口
├── .env.example                 # API Key 模板
│
├── src/
│   ├── config/loader.py               # 配置加载、校验、环境变量插值
│   │
│   ├── doc_processing/                # 文档处理管线
│   │   ├── parsers.py                 # 格式解析器（txt/md/pdf/docx/html）
│   │   ├── cleaner.py                 # HTML 清洗 & 文本规范化
│   │   ├── chunker.py                 # 递归分块
│   │   ├── metadata.py                # 元数据加强
│   │   ├── deduplicator.py            # 内容哈希去重
│   │   └── loader.py                  # 文件发现 & 格式分发
│   │
│   ├── embedding/
│   │   ├── embedder.py                # BGE-M3 封装（版本锁定）
│   │   └── cache.py                   # 磁盘 Embedding 缓存
│   │
│   ├── vector_store/
│   │   ├── chroma_store.py            # ChromaDB 客户端
│   │   └── bm25_index.py             # BM25 关键词索引（jieba）
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py        # 稠密+稀疏+RRF 融合
│   │   ├── reranker.py                # BGE-Reranker 重排序
│   │   └── semantic_cache.py          # 语义缓存
│   │
│   ├── generation/
│   │   ├── providers.py               # 5 家 LLM 配置
│   │   ├── llm_adapter.py             # 统一 OpenAI 兼容接口
│   │   ├── prompt_builder.py          # Prompt + 来源组装
│   │   └── hallucination.py           # 幻觉检测
│   │
│   ├── kb_manager/
│   │   ├── models.py                  # KB 元数据模型
│   │   └── manager.py                 # 多 KB 管理
│   │
│   ├── source_tracking/
│   │   └── tracker.py                 # 来源追踪 & 引用校验
│   │
│   └── cli/
│       ├── main.py                    # CLI 入口
│       ├── pipeline.py                # 核心管线编排
│       ├── import_cmd.py              # kb import
│       ├── search_cmd.py              # kb search
│       ├── chat_cmd.py                # kb chat / kb ask
│       └── kb_cmd.py                  # kb kb / kb provider
│
├── data/
│   ├── chroma_db/                     # 向量存储
│   ├── bm25/                          # BM25 索引
│   └── kb_registry.json               # 知识库注册表
│
└── tests/                             # 测试
```

## 配置说明

核心配置项在 `config.yaml`：

```yaml
chunking:
  chunk_size: 500          # 分块大小（字符）
  chunk_overlap: 80        # 重叠大小

retrieval:
  dense_top_k: 50          # 向量检索候选数
  sparse_top_k: 50         # BM25 检索候选数
  rrf_k: 60                # RRF 融合常数
  rerank_top_n: 5          # 重排序后保留数

llm:
  temperature: 0.3          # 低温度确保准确性
  max_tokens: 1024
```

## 成本

| 组件 | 成本 |
|------|------|
| Embedding (BGE-M3) | 免费 |
| 向量库 (ChromaDB) | 免费 |
| BM25 索引 | 免费 |
| 重排序 (BGE-Reranker) | 免费 |
| LLM API | 唯一成本，个人使用 < 10 元/月 |

## 设计参考

- [从0实现工业级 RAG 智能客服：架构、核心代码、部署全拆解](https://mp.weixin.qq.com/s/Qm4t2Xe5XfIaH5CiSdtvPw)
- [RAG检索增强-向量库与Chunking](https://mp.weixin.qq.com/s/woz4NdE3tCbr5z9DPl6y4g)