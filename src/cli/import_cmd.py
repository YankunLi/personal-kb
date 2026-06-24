"""kb import command: import documents into a knowledge base."""

import click

from src.cli.pipeline import get_pipeline


@click.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--kb", "kb_name", default="default", help="目标知识库名称")
@click.option("--recursive/--no-recursive", default=True, help="是否递归导入子目录")
@click.option("--dry-run", is_flag=True, help="预览模式，不实际导入")
def import_cmd(path: str, kb_name: str, recursive: bool, dry_run: bool):
    """导入文档到知识库。

    PATH: 文档文件或目录路径。
    """
    pipeline = get_pipeline()

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
        else:
            msg = f"  {step}: {file_path}"
        if msg != last_status[0]:
            click.echo(msg)
            last_status[0] = msg

    result = pipeline.import_documents(
        path=path,
        kb_name=kb_name,
        recursive=recursive,
        dry_run=dry_run,
        progress_callback=progress,
    )

    if dry_run:
        click.echo(f"\n✅ 预览完成: {result['files']} 个文件将被导入")
    else:
        click.echo(f"\n✅ 导入完成!")
        click.echo(f"   文件: {result['files']} 个")
        click.echo(f"   分块: {result['chunks']} 个")
        if result['duplicates'] > 0:
            click.echo(f"   去重: {result['duplicates']} 个重复块已跳过")