"""CLI entry point for the personal knowledge base system."""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="kb")
def cli():
    """个人知识库 RAG 系统 - Personal Knowledge Base with RAG.

    管理你的个人文档，通过 AI 智能检索和问答。
    """
    pass


# Import subcommands
from src.cli.import_cmd import import_cmd
from src.cli.search_cmd import search_cmd
from src.cli.chat_cmd import chat_cmd, ask_cmd
from src.cli.kb_cmd import kb_cmd, provider_cmd

cli.add_command(import_cmd)
cli.add_command(search_cmd)
cli.add_command(chat_cmd)
cli.add_command(ask_cmd)
cli.add_command(kb_cmd)
cli.add_command(provider_cmd)


if __name__ == "__main__":
    cli()