#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implement a question/answer chat bot with RAG using DSPy.
"""

import os
import time
import traceback

from icecream import ic
import dspy

from .vec import VectorStore


class DSPy_RAG (dspy.Module):
    """DSPy implementation of a RAG signature."""

    def __init__(
        self,
        config: dict,
        *,
        run_local: bool = True,
        ) -> None:
        """Constructor."""
        self.config: dict = config

        # load the LLM
        if run_local:
            self.lm: dspy.LM = dspy.LM(
                self.config["rag"]["lm_name"],
                api_base = self.config["rag"]["api_base"],
                api_key = "",
                temperature = self.config["rag"]["temperature"],
                max_tokens = self.config["rag"]["max_tokens"],
                stop = None,
                cache = False,
            )
        else:
            OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")

            if OPENAI_API_KEY is None:
                raise ValueError(
                    "Environment variable 'OPENAI_API_KEY' is not set. Please set it to proceed."
                )

            self.lm = dspy.LM(
                self.config["rag"]["lm_name"],
                temperature = 0.0,
            )

        # set up the `DSPy` signature, using basic RAG
        dspy.configure(lm = self.lm)
        self.respond: dspy.Predict = dspy.Predict(
            "context, question -> response"
        )
        # NOTE: this method does the heavy lifting for LLM integration
        self.context: list[ str ] = []


    def forward (
        self,
        question: str,
        ) -> dspy.primitives.prediction.Prediction:
        """Invoke the RAG signature."""
        reply: dspy.primitives.prediction.Prediction = self.respond(
            context = self.context,
            question = question,
        )

        return reply


class RAG:
    """Wrapper for calling RAG in a question/answer chat bot."""

    def __init__ (
        self,
        config: dict,
        vec_store: VectorStore,
        *,
        run_local: bool = True,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.vec_store: VectorStore = vec_store

        self.rag: DSPy_RAG = DSPy_RAG(
            self.config,
            run_local = run_local,
        )


    def qa_signature (
        self,
        question: str,
        chunks: list[ str ],
        ) -> dspy.primitives.prediction.Prediction:
        """Run one question/answer cycle."""
        self.rag.context = chunks
        response: dspy.primitives.prediction.Prediction = self.rag(question)

        if False: # True  # disable if too verbose
            dspy.inspect_history()

        return response


    def question_answer (
        self,
        *,
        debug: bool = False,
        ) -> None:
        """Loop to answer questions."""
        try:
            # loop to answer questions
            while True:
                question: str = input("\nQuoi? ").strip()

                if len(question) < 1:
                    continue

                if question.lower() in [ "quitter", "bye", "adieu" ]:
                    break

                # LLM summarizes the text chunks in response to the question
                chunks: list[ str ] = self.query_chunks(question)

                response: dspy.primitives.prediction.Prediction = self.qa_signature(
                    question,
                    chunks,
                )

                ic(question)
                ic(response.response)
                print("-" * 10)

        except EOFError:
            print("")
        except Exception as ex:
            ic(ex)
            traceback.print_exc()
        finally:
            print("\nÀ bientôt!\n")
            time.sleep(.1)


    def query_chunks (
        self,
        question: str,
        *,
        max_chunks: int = 11,
        ) -> list[ str ]:
        """find text chunks near the question"""
        return self.vec_store.chunk_table.search(
            question
        ).select(
            [ "text", "_distance" ]
        ).limit(
            max_chunks
        ).to_polars()["text"].to_list()

