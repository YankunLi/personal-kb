"""kb chat and kb ask commands: interactive chat and single-shot Q&A."""

import asyncio
import logging

import click

from src.cli.pipeline import get_pipeline
from src.source_tracking.tracker import format_sources_output

logger = logging.getLogger(__name__)


@click.command("chat")
@click.option("--kb", "kb_name", default="default", help="知识库名称")
@click.option("--provider", "provider_name", default=None, help="LLM 提供商")
@click.option("--no-stream", is_flag=True, help="禁用流式输出")
def chat_cmd(kb_name: str, provider_name: str, no_stream: bool):
    """交互式对话模式，基于知识库问答。"""
    pipeline = get_pipeline()

    try:
        kb_info = pipeline.kb_manager.get(kb_name)
    except ValueError as e:
        click.echo(f"❌ {e}")
        return

    prov_name = provider_name or pipeline.config.defaults.provider
    prov_info = pipeline.config.llm.providers.get(prov_name)

    click.echo(f"使用知识库: {kb_name} ({kb_info.chunk_count} chunks, {kb_info.file_count} files)")
    if prov_info:
        click.echo(f"LLM: {prov_info.name} ({prov_info.model})")
    click.echo("输入 /exit 退出, /help 查看帮助, /sources 查看来源, /clear 清除历史\n")

    chat_history: list[dict[str, str]] = []
    last_sources: list = []

    while True:
        try:
            query = click.prompt("你", prompt_suffix=": ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\n再见！")
            break

        query = query.strip()
        if not query:
            continue

        # Handle special commands
        if query.startswith("/"):
            cmd = query.lower()
            if cmd in ("/exit", "/quit"):
                click.echo("再见！")
                break
            elif cmd == "/help":
                click.echo("""
可用命令:
  /exit, /quit    退出
  /help           显示帮助
  /sources        显示上次回答的来源
  /clear          清除对话历史
  /kb <name>      切换知识库
  /provider <name> 切换 LLM 提供商
  /stats          显示会话统计
""")
                continue
            elif cmd == "/sources":
                if last_sources:
                    click.echo(format_sources_output(last_sources))
                else:
                    click.echo("暂无来源（请先提问）")
                continue
            elif cmd == "/clear":
                chat_history.clear()
                last_sources.clear()
                click.echo("对话历史和来源已清除")
                continue
            elif cmd.startswith("/kb "):
                new_kb = query[4:].strip()  # use original query, not lowercased
                if pipeline.kb_manager.exists(new_kb):
                    kb_name = new_kb
                    chat_history.clear()
                    last_sources.clear()
                    kb_info = pipeline.kb_manager.get(kb_name)
                    click.echo(f"已切换到知识库: {kb_name} ({kb_info.chunk_count} chunks)")
                else:
                    click.echo(f"知识库 '{new_kb}' 不存在")
                continue
            elif cmd.startswith("/provider "):
                new_prov = query[10:].strip().lower()  # provider keys are lowercase
                if new_prov in pipeline.config.llm.providers:
                    provider_name = new_prov
                    prov_info = pipeline.config.llm.providers[new_prov]
                    click.echo(f"已切换到: {prov_info.name} ({prov_info.model})")
                else:
                    click.echo(f"提供商 '{new_prov}' 不可用")
                continue
            elif cmd == "/stats":
                try:
                    kb_info = pipeline.kb_manager.get(kb_name)
                    click.echo(f"知识库: {kb_name} | Chunks: {kb_info.chunk_count} | 历史: {len(chat_history)} 条")
                except ValueError:
                    click.echo(f"⚠️  知识库 '{kb_name}' 已不存在，请用 /kb <name> 切换")
                continue

        # Normal query
        click.echo()  # blank line
        asyncio.run(_do_chat(pipeline, query, kb_name, provider_name, chat_history, last_sources, no_stream))
        click.echo()


async def _do_chat(pipeline, query, kb_name, provider_name, chat_history, last_sources, no_stream):
    """Execute a chat turn with streaming output."""
    try:
        if no_stream:
            response = await pipeline.chat(
                query, kb_name=kb_name, provider_name=provider_name,
                chat_history=chat_history, stream=False,
            )
            click.echo(f"🤖 {response.answer}")
            if response.sources:
                last_sources.clear()
                last_sources.extend(response.sources)
                click.echo(format_sources_output(response.sources))
            if response.hallucination_risk != "low":
                click.echo(f"⚠️  幻觉风险: {response.hallucination_risk}")

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response.answer})
        else:
            click.echo("🤖 ", nl=False)
            full_answer = ""
            sources = None

            async for chunk in pipeline.chat_stream(
                query, kb_name=kb_name, provider_name=provider_name,
                chat_history=chat_history,
            ):
                if chunk["type"] == "token":
                    click.echo(chunk["content"], nl=False)
                    full_answer += chunk["content"]
                elif chunk["type"] == "answer":
                    click.echo(chunk["content"], nl=False)
                    full_answer = chunk["content"]
                elif chunk["type"] == "sources":
                    sources = chunk["sources"]
                    last_sources.clear()
                    last_sources.extend(chunk["sources"])
                elif chunk["type"] == "done":
                    click.echo()
                    if sources:
                        click.echo(format_sources_output(sources))
                    if chunk.get("hallucination_risk", "low") != "low":
                        click.echo(f"⚠️  幻觉风险: {chunk['hallucination_risk']}")

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": full_answer})

        # Cap chat history to last 20 turns (40 messages) to prevent unbounded
        # memory growth in long-running interactive sessions.
        if len(chat_history) > 40:
            chat_history[:] = chat_history[-40:]
    except KeyboardInterrupt:
        click.echo("\n⏸️  已取消")
    except Exception as e:
        logger.warning("Chat error: %s", e, exc_info=True)
        click.echo(f"\n❌ 对话出错: {e}")
        # Don't add failed query to history


@click.command("ask")
@click.argument("query")
@click.option("--kb", "kb_name", default="default", help="知识库名称")
@click.option("--provider", "provider_name", default=None, help="LLM 提供商")
@click.option("--no-stream", is_flag=True, help="禁用流式输出")
def ask_cmd(query: str, kb_name: str, provider_name: str, no_stream: bool):
    """单次问答，不进入交互模式。

    QUERY: 问题字符串。
    """
    pipeline = get_pipeline()

    click.echo(f"🔍 在知识库 '{kb_name}' 中查询...\n")
    asyncio.run(_do_chat(pipeline, query, kb_name, provider_name, [], [], no_stream))