#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the vector store.
"""

from lancedb.embeddings import get_registry
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from lancedb.pydantic import LanceModel, Vector
import lancedb
import polars as pl

EMBED_FUNC: SentenceTransformerEmbeddings = get_registry().get("sentence-transformers").create()


class TextChunk (LanceModel):
    """Represents one chunk of text from a document."""
    uid: int
    text: str = EMBED_FUNC.SourceField()
    vector: Vector(EMBED_FUNC.ndims()) = EMBED_FUNC.VectorField(default = None)


class VectorStore:
    """Manage the TextChunk objects in the vector store."""

    def __init__ (
        self,
        config: dict,
        ) -> None:
        """Constructor."""
        self.config: dict = config

        # the vector store in `LanceDB`
        self.lancedb_conn: lancedb.db.LanceDBConnection = lancedb.connect(
            self.config["vect"]["lancedb_uri"],
        )

        self.start_chunk_id: int = 0
        self.chunk_table: lancedb.table.LanceTable | None = None


    def open_vector_tables (
        self,
        *,
        create: bool = False,
        ) -> None:
        """Create or open the table for text chunk embeddings."""
        if create:
            # intialize and clear any previous table
            self.chunk_table = self.lancedb_conn.create_table(
                self.config["vect"]["chunk_table"],
                schema = TextChunk,
                mode = "overwrite",
            )
        else:
            # open existing table
            self.chunk_table = self.lancedb_conn.open_table(
                self.config["vect"]["chunk_table"],
            )

            df_chunks: pl.DataFrame = self.chunk_table.search().select([ "uid" ]).to_polars()
            uids: list[ int ] = [ uid for uid in df_chunks.iter_rows() ]

            if len(uids) > 0:
                self.start_chunk_id = max(uids)[0] + 1
            else:
                self.start_chunk_id = 0


    def add_chunk (
        self,
        text: str,
        ) -> TextChunk:
        """Add a chunk into the vector store."""
        chunk: TextChunk = TextChunk(
            uid = self.start_chunk_id,
            text = text,
        )

        self.chunk_table.add([ chunk ])
        self.start_chunk_id += 1

        return chunk
