#!/usr/bin/env bash
set -e

# =========================================================
#  personal-kb 一键安装脚本
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  personal-kb 个人知识库 RAG 系统 - 安装"
echo "=========================================="
echo ""

# 1. 检查 Python 版本
echo "[1/4] 检查 Python 环境..."

PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            echo "  找到 Python: $version ($PYTHON)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ❌ 需要 Python >= 3.11，未找到合适版本"
    echo "  请安装 Python 3.11+: brew install python@3.11"
    exit 1
fi

# 2. 创建虚拟环境
echo ""
echo "[2/4] 创建虚拟环境..."

if [ -d ".venv" ]; then
    echo "  虚拟环境已存在，跳过"
else
    $PYTHON -m venv .venv
    echo "  虚拟环境创建完成"
fi

source .venv/bin/activate

# 3. 安装依赖
echo ""
echo "[3/4] 安装依赖..."

# 检测网络，国内优先用清华镜像
MIRROR=""
if curl -s --connect-timeout 3 https://pypi.tuna.tsinghua.edu.cn/simple &>/dev/null; then
    echo "  检测到清华镜像可用，使用国内源加速"
    MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

pip install --upgrade pip -q 2>/dev/null
pip install -e . $MIRROR

echo "  依赖安装完成"

# 4. 创建 .env（如果不存在）
echo ""
echo "[4/4] 初始化配置..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  已创建 .env 文件，请编辑填入 API Key"
    echo ""
    echo "  支持的 LLM 提供商:"
    echo "    - QWEN_API_KEY      (阿里通义千问)"
    echo "    - GLM_API_KEY       (智谱 GLM)"
    echo "    - DEEPSEEK_API_KEY  (DeepSeek)"
    echo "    - HUNYUAN_API_KEY   (腾讯混元)"
    echo "    - ERNIE_API_KEY     (百度文心一言)"
    echo ""
    echo "  至少配置一个即可使用。"
else
    echo "  .env 文件已存在，跳过"
fi

echo ""
echo "=========================================="
echo "  ✅ 安装完成！"
echo "=========================================="
echo ""
echo "  使用方式:"
echo "    source .venv/bin/activate"
echo "    kb import ./your-docs/ --kb mykb"
echo "    kb chat --kb mykb"
echo ""
echo "  如果无法访问 HuggingFace 下载模型，设置:"
echo "    export HF_ENDPOINT=https://hf-mirror.com"
echo ""