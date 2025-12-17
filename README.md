# min_rag

A minimal example of RAG, based on DSPy.


## set up

To set up the environment:

```bash
git clone https://github.com/DerwenAI/min_rag.git
cd min_rag

python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip wheel
python3 -m pip install -r requirements.txt
```

If you want to use ChatGPT instead of a locally hosted LLM:

  - set the `OPENAI_API_KEY` environment variable to your OpenAI API key
  - set the `run_local = False` flag in "demo.py"

Otherwise this uses [`ollama`](https://ollama.com/) to download and
orchestrate a local LLM.

The [`oss-gpt:20b`](https://huggingface.co/openai/gpt-oss-20b) model
is set by default, and to have it running locally:

```bash
ollama pull oss-gpt:20b
```

Or change the "rag.lm_name" configuration setting to a different model
which you have downloaded and run locally.


## running

To load the vector database from markdown files, then run a
question/answer chat bot based on RAG:

```bash
python3 demo.py
```

Then ask questions.

Change the markdown files in `data/talks` to add new content, or point
to a different directory.


---

> "For those we hold close, and for those we never meet."
