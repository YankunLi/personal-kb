"""kb kb command: knowledge base management (create, list, delete, use, info)."""

import click

from src.cli.pipeline import get_pipeline
from src.config.loader import load_config


@click.group("kb")
def kb_cmd():
    """知识库管理命令。

    创建、查看、删除、切换知识库。
    """
    pass


@kb_cmd.command("create")
@click.argument("name")
@click.option("--topic", default="", help="知识库主题描述")
def kb_create(name: str, topic: str):
    """创建新知识库。

    NAME: 知识库名称。
    """
    pipeline = get_pipeline()
    try:
        info = pipeline.kb_manager.create(name, topic=topic)
        click.echo(f"✅ 知识库 '{name}' 创建成功")
        if topic:
            click.echo(f"   主题: {topic}")
    except ValueError as e:
        click.echo(f"❌ {e}")


@kb_cmd.command("list")
def kb_list():
    """列出所有知识库。"""
    pipeline = get_pipeline()
    config = load_config()
    default_kb = config.defaults.kb

    kbs = pipeline.kb_manager.list()
    if not kbs:
        click.echo("暂无知识库。使用 'kb kb create <name>' 创建。")
        return

    click.echo(f"{'名称':<20} {'主题':<20} {'Chunks':<10} {'文件':<10}")
    click.echo("-" * 60)
    for kb in kbs:
        marker = " *" if kb.name == default_kb else ""
        click.echo(f"{kb.name + marker:<20} {kb.topic:<20} {kb.chunk_count:<10} {kb.file_count:<10}")


@kb_cmd.command("delete")
@click.argument("name")
@click.option("--force", is_flag=True, help="跳过确认")
def kb_delete(name: str, force: bool):
    """删除知识库。

    NAME: 要删除的知识库名称。
    """
    if not force:
        if not click.confirm(f"确定要删除知识库 '{name}' 吗？此操作不可恢复！"):
            click.echo("已取消")
            return

    pipeline = get_pipeline()
    try:
        pipeline.kb_manager.delete(name, force=True)
        click.echo(f"✅ 知识库 '{name}' 已删除")
    except ValueError as e:
        click.echo(f"❌ {e}")


@kb_cmd.command("use")
@click.argument("name")
def kb_use(name: str):
    """设置默认知识库。

    NAME: 知识库名称。
    """
    import re
    import tempfile
    import shutil
    from pathlib import Path

    from src.config.loader import get_project_root

    pipeline = get_pipeline()
    if not pipeline.kb_manager.exists(name):
        click.echo(f"❌ 知识库 '{name}' 不存在")
        return

    # Update config.yaml preserving comments and formatting
    config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        content = f.read()

    content = re.sub(r"(\n\s*kb:\s*)(?:\"|\')?(\S+)(?:\"|\')?", rf"\1{name}", content)
    with tempfile.NamedTemporaryFile("w", dir=config_path.parent, delete=False, encoding="utf-8") as tf:
        tf.write(content)
    shutil.move(tf.name, config_path)

    click.echo(f"✅ 默认知识库已切换为 '{name}'")


@kb_cmd.command("info")
@click.argument("name", default="default")
def kb_info(name: str):
    """查看知识库详细信息。

    NAME: 知识库名称（默认: default）。
    """
    pipeline = get_pipeline()
    try:
        info = pipeline.kb_manager.get(name)
        click.echo(f"名称: {info.name}")
        click.echo(f"主题: {info.topic or '无'}")
        click.echo(f"创建时间: {info.created_at}")
        click.echo(f"Chunk 数量: {info.chunk_count}")
        click.echo(f"文件数量: {info.file_count}")
    except ValueError as e:
        click.echo(f"❌ {e}")


@kb_cmd.command("rename")
@click.argument("old_name")
@click.argument("new_name")
def kb_rename(old_name: str, new_name: str):
    """重命名知识库。

    OLD_NAME: 当前名称。
    NEW_NAME: 新名称。
    """
    pipeline = get_pipeline()
    try:
        pipeline.kb_manager.rename(old_name, new_name)
        click.echo(f"✅ 知识库 '{old_name}' 已重命名为 '{new_name}'")
    except ValueError as e:
        click.echo(f"❌ {e}")


# Provider commands
@click.group("provider")
def provider_cmd():
    """LLM 提供商管理。"""
    pass


@provider_cmd.command("list")
def provider_list():
    """列出所有可用的 LLM 提供商。"""
    config = load_config()
    default_provider = config.defaults.provider

    click.echo(f"{'名称':<15} {'提供商':<20} {'模型':<25}")
    click.echo("-" * 60)
    for key, prov in config.llm.providers.items():
        marker = " *" if key == default_provider else ""
        has_key = "✅" if prov.api_key and not prov.api_key.startswith("${") and "xxx" not in prov.api_key.lower() else "❌"
        click.echo(f"{key + marker:<15} {prov.name:<20} {prov.model:<25} {has_key}")


@provider_cmd.command("use")
@click.argument("name")
def provider_use(name: str):
    """切换默认 LLM 提供商。

    NAME: 提供商名称 (qwen/glm/deepseek/hunyuan/ernie)。
    """
    import re
    import tempfile
    import shutil
    from pathlib import Path

    from src.config.loader import get_project_root

    config = load_config()
    if name not in config.llm.providers:
        click.echo(f"❌ 未知提供商 '{name}'。可用: {list(config.llm.providers.keys())}")
        return

    config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        content = f.read()

    content = re.sub(r"(\n\s*provider:\s*)(?:\"|\')?(\S+)(?:\"|\')?", rf"\1{name}", content)
    with tempfile.NamedTemporaryFile("w", dir=config_path.parent, delete=False, encoding="utf-8") as tf:
        tf.write(content)
    shutil.move(tf.name, config_path)

    click.echo(f"✅ 默认 LLM 已切换为 '{name}' ({config.llm.providers[name].name})")