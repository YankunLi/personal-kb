"""kb import command: import documents into a knowledge base."""

import click

from src.cli.pipeline import get_pipeline


@click.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--kb", "kb_name", default=None, help="目标知识库名称")
@click.option("--recursive/--no-recursive", default=True, help="是否递归导入子目录")
@click.option("--dry-run", is_flag=True, help="预览模式，不实际导入")
def import_cmd(path: str, kb_name: str, recursive: bool, dry_run: bool):
    """导入文档到知识库。

    PATH: 文档文件或目录路径。
    """
    pipeline = get_pipeline()
    if kb_name is None:
        kb_name = pipeline.config.defaults.kb

    click.echo(f"📂 导入文档到知识库 '{kb_name}'...")
    if dry_run:
        click.echo("🔍 预览模式（不会实际导入）")

    # Progress callback
    last_status = [""]

    def progress(step: str, file_path: str, count: int):
        if step == "parse":
            msg = f"  📄 解析: {file_path}"
        elif step == "embed":
            msg = "  🧠 正在生成向量嵌入..."
        elif step == "index":
            msg = "  📊 正在建立索引..."
        elif step == "rollback":
            msg = "  ⚠️  BM25 写入失败，正在回滚..."
        else:
            msg = f"  {step}: {file_path}"
        if msg != last_status[0]:
            click.echo(msg)
            last_status[0] = msg

    try:
        result = pipeline.import_documents(
            path=path,
            kb_name=kb_name,
            recursive=recursive,
            dry_run=dry_run,
            progress_callback=progress,
        )
    except KeyboardInterrupt:
        click.echo("\n\n⏸️  已取消")
        return
    except Exception as e:
        _handle_import_error(e)
        return

    if dry_run:
        click.echo(f"\n✅ 预览完成: {result['files']} 个文件将被导入")
    else:
        click.echo(f"\n✅ 导入完成!")
        click.echo(f"   文件: {result['files']} 个")
        click.echo(f"   分块: {result['chunks']} 个")
        if result['duplicates'] > 0:
            click.echo(f"   去重: {result['duplicates']} 个重复块已跳过")
        if result.get('failed', 0) > 0:
            click.echo(f"   ⚠️  {result['failed']} 个文件解析失败")


def _handle_import_error(e: Exception):
    """Provide user-friendly error messages for common import failures."""
    msg = str(e).lower()
    if "no such file" in msg or "not found" in msg or "找不到" in msg or "不存在" in msg:
        click.echo(f"❌ 文件未找到: {e}")
    elif "permission" in msg or "权限" in msg or "拒绝" in msg:
        click.echo(f"❌ 权限不足: {e}")
    elif "memory" in msg or "oom" in msg or "内存" in msg:
        click.echo(f"❌ 内存不足，请尝试导入较小的文件: {e}")
    elif ("model" in msg and ("load" in msg or "download" in msg)) or "模型" in msg:
        click.echo(f"❌ 模型加载失败，请检查网络和 HuggingFace 镜像设置: {e}")
    elif "disk" in msg or "磁盘" in msg or "空间" in msg:
        click.echo(f"❌ 磁盘空间不足: {e}")
    elif "network" in msg or "网络" in msg or "timeout" in msg or "超时" in msg:
        click.echo(f"❌ 网络错误，请检查网络连接: {e}")
    else:
        click.echo(f"❌ 导入失败: {e}")