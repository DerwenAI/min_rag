#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal example using DSPy to implement a question/answer chat bot
based on _retrieval-augmented generation_ (RAG).
"""

import pathlib
import tomllib
import typing

from icecream import ic
import bs4
import markdown

from min_rag import RAG, TextChunk, VectorStore


def get_chunks (
    *,
    md_dir: pathlib.Path = pathlib.Path("data/talks"),
    chunk_size: int = 1000,
    ) -> typing.Iterator[ list[ str ] ]:
    """Generate chunked lists of Markdown paragraphs."""
    for md_path in md_dir.rglob("*.md"):
        ic(md_path.name)

        # use `BeautifulSoup` to extract paragraphs from Markdown files
        soup: bs4.BeautifulSoup = bs4.BeautifulSoup(
            markdown.markdown(
                md_path.read_text(),
                output_format = "html5",
            ),
            "html.parser",
        )

        # group the paragraph texts into chunk-sized lists
        sum_chars: int = 0
        para_list: list[ str ] = []

        for para in soup.find_all("p"):
            num_chars: int = len(para.text)

            if num_chars > 0:
                if (sum_chars + num_chars) < chunk_size:
                    para_list.append(para.text)
                    sum_chars += num_chars
                else:
                    # emit prev collection of paragraph texts
                    yield para_list
                    para_list = [ para.text ]
                    sum_chars = num_chars

        # emit last collection of paragraph texts
        yield para_list


if __name__ == "__main__":
    # configuration
    config_path: pathlib.Path = pathlib.Path("config.toml")

    with config_path.open("rb") as fp:
        config: dict = tomllib.load(fp)

    # instantiate a vector store
    vec_store: VectorStore = VectorStore(config)
    vec_store.open_vector_tables(create = True)

    # form text chunks: join the list of paragraph texts with
    # a double CRLF return
    for para_list in get_chunks(chunk_size = config["vect"]["chunk_size"]):
        text: str = "\n\n".join(para_list)
        chunk: TextChunk = vec_store.add_chunk(text)
        #ic(chunk)

    # run a question/answer chat bot using RAG
    rag: RAG = RAG(
        config,
        vec_store,
        run_local = True,
    )

    rag.question_answer()

