"""kb search command: search knowledge base without LLM generation."""

import click

from src.cli.pipeline import get_pipeline


@click.command("search")
@click.argument("query")
@click.option("--kb", "kb_name", default=None, help="知识库名称")
@click.option("--top-k", default=5, show_default=True, help="返回结果数量")
@click.option("--show-scores", is_flag=True, help="显示相关度分数")
def search_cmd(query: str, kb_name: str, top_k: int, show_scores: bool):
    """搜索知识库（仅检索，不调用 LLM）。

    QUERY: 搜索查询字符串。
    """
    pipeline = get_pipeline()
    if kb_name is None:
        kb_name = pipeline.config.defaults.kb

    click.echo(f"🔍 在知识库 '{kb_name}' 中搜索: {query}\n")

    try:
        results = pipeline.search(query, kb_name=kb_name, top_k=top_k)
    except ValueError as e:
        click.echo(f"❌ {e}")
        return
    except KeyboardInterrupt:
        click.echo("\n⏸️  已取消")
        return
    except Exception as e:
        click.echo(f"❌ 搜索失败: {e}")
        return

    if not results:
        click.echo("未找到相关结果。如果知识库为空，请先使用 'kb import' 导入文档。")
        return

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        source_file = metadata.get("source_file_basename", "未知")
        section = metadata.get("section", "")
        score = result.get("rerank_score")
        if score is None:
            score = result.get("score", 0)
        if score is None:
            score = 0.0
        content = result.get("content", "")

        click.echo(f"── 结果 {i} ──")
        if show_scores:
            click.echo(f"📊 相关度: {score:.4f}")
        click.echo(f"📄 来源: {source_file}")
        if section:
            click.echo(f"📑 章节: {section}")
        click.echo(f"📝 内容: {content[:200]}{'...' if len(content) > 200 else ''}")
        click.echo()