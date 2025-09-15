# MRAGChecker

> A generalized, backend‑agnostic **RAG & Multimodal** checker that works with **OpenAI GPT**, **Amazon Bedrock (Claude)**, and **open‑source models via vLLM**. Split into two pluggable parts:
>
> 1) **Text‑only RAGChecker** (compatible with the classic RAG checking workflow)
> 2) **MRAG (Multimodal) Checker** for image‑text QA and chart/slide/doc VQA

All code comments in this repo are in **English**.

---

## ✨ Why this repo?
- Keep the familiar workflow and JSON schemas from prior “ragchecker” scripts.
- Remove vendor lock‑in: use **GPT**, **Claude**, **or** open models via **vLLM** with one unified interface.
- Support both **text‑only** and **multimodal** (image + text) cases.
- Provide a clean **CLI** and **Python API** you can embed in your own pipelines.

---

## 🧱 Repository layout
```
MRAGChecker/
├─ pyproject.toml                      # or setup.cfg / setup.py
├─ README.md                           # this file
├─ mragchecker/
│  ├─ __init__.py
│  ├─ schema.py                        # dataclasses for I/O items
│  ├─ io_utils.py                      # jsonl read/write, packing/unpacking
│  ├─ prompts/
│  │  ├─ prompt_builder.py             # assemble prompts consistently
│  │  ├─ judge_templates.py            # LLM-as-judge templates
│  ├─ metrics/
│  │  ├─ text_metrics.py               # EM/F1/etc for text QA
│  │  ├─ mcq_metrics.py                # accuracy, letter extraction, etc.
│  │  └─ mrag_metrics.py               # multimodal extras (PED, path stats)
│  ├─ pipelines/
│  │  ├─ generate.py                   # run generation over datasets
│  │  ├─ evaluate.py                   # run judge LLM + compute metrics
│  │  └─ pack_for_checker.py           # pack your jsonl -> checking_inputs.json
│  ├─ datasets/
│  │  ├─ webqa.py                      # loaders/adapters (text+image ids)
│  │  ├─ visrag.py                     # Arxiv/Plot/Slide/Doc adapters
│  │  ├─ chartrag.py                   # Chart-MRAG adapter
│  │  └─ mrag_bench.py                 # MRAG-Bench adapter
│  └─ backends/
│     ├─ __init__.py
│     ├─ base.py                       # ChatModel interface (text/vision)
│     ├─ openai_backend.py             # GPT via OpenAI Responses API
│     ├─ bedrock_backend.py            # Claude via Bedrock Runtime
│     └─ vllm_backend.py               # open-source models via vLLM
├─ scripts/
│  ├─ convert_to_ragchecker_input.py   # convenience packer
│  └─ demo_quickstart.sh               # end-to-end runnable example
└─ examples/
   └─ checking_inputs.json             # sample packed input for evaluator
```

---

## 🔌 Unified model interface

Create a thin abstract interface so every backend behaves the same at call‑sites.

```python
# mragchecker/backends/base.py
# NOTE: Comments in English only.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
from PIL import Image

@dataclass
class GenerateConfig:
    max_tokens: int = 512
    temperature: float = 0.2

class ChatModel:
    """Unified interface for text or multimodal generation."""
    name: str

    def supports_vision(self) -> bool:
        return False

    def generate(
        self,
        prompt_text: str,
        retrieved_texts: Optional[List[str]] = None,
        image_inputs: Optional[List[Union[str, Image.Image]]] = None,
        cfg: Optional[GenerateConfig] = None,
    ) -> str:
        raise NotImplementedError
```

### OpenAI (GPT)
```python
# mragchecker/backends/openai_backend.py
from .base import ChatModel, GenerateConfig
# Implement using OpenAI Responses API; encode images as data URLs.
# generate() builds a single user turn with optional context + images + prompt.
```

### Bedrock (Claude)
```python
# mragchecker/backends/bedrock_backend.py
from .base import ChatModel, GenerateConfig
# Implement via boto3 bedrock-runtime; images in Anthropic input_image format.
```

### vLLM (open-source models)
```python
# mragchecker/backends/vllm_backend.py
from .base import ChatModel, GenerateConfig
# Wrap vLLM LLM + SamplingParams; format prompts per model family (llava/qwen/phi...).
```

> Each backend exposes: `supports_vision()` and `generate(...)`.

---

## 📦 I/O schema (compatible with RAGChecker-style JSON)

```python
# mragchecker/schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ContextDoc:
    doc_id: str
    text: str

@dataclass
class Sample:
    dataset: str
    query_id: str
    query: str
    gt_answer: str | Dict[str, Any] | None
    options: List[str] = field(default_factory=list)  # MCQ options if any
    images: List[str] = field(default_factory=list)   # local paths or URIs
    retrieved_context: List[ContextDoc] = field(default_factory=list)

@dataclass
class GenerationResult:
    sample: Sample
    response: str
    pred_letter: Optional[str] = None
    correct: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)
```

JSONL rows for generation cache mirror `GenerationResult` (flattened). A `checking_inputs.json` packs all rows under a top‑level `{ "results": [...] }` for the evaluator.

---

## 🧪 Pipelines

### Generation
`mragchecker/pipelines/generate.py`

```python
# Pseudocode outline (comments only in English)
from .schema import Sample, GenerationResult
from ..backends.base import ChatModel, GenerateConfig
from ..prompts.prompt_builder import assemble_prompt, extract_answer_letter

def run_generation(samples: list[Sample], model: ChatModel, cfg: GenerateConfig) -> list[GenerationResult]:
    out = []
    for s in samples:
        prompt, pm = assemble_prompt(s)  # returns prompt_text and prompt_meta
        resp = model.generate(prompt_text=prompt,
                              retrieved_texts=[d.text for d in s.retrieved_context],
                              image_inputs=s.images,
                              cfg=cfg)
        letter = extract_answer_letter(resp) if s.options else None
        correct = (letter == s.gt_answer) if (letter and isinstance(s.gt_answer, str) and len(s.gt_answer) == 1) else None
        out.append(GenerationResult(sample=s, response=resp, pred_letter=letter, correct=correct, meta={"prompt_meta": pm}))
    return out
```

### Evaluate (LLM‑as‑Judge + metrics)
`mragchecker/pipelines/evaluate.py`

```python
# Outline: run a judge model with a template and compute metrics.
# Supports both text-only and multimodal (images optional).
```

### Pack to RAGChecker input
`mragchecker/pipelines/pack_for_checker.py` or `scripts/convert_to_ragchecker_input.py`

---

## 🛠️ Datasets/adapters
Each adapter returns a list of `Sample` items with `query`, `gt_answer`, optional `options`, optional `images`, and `retrieved_context` (after calling your retriever).

- `webqa.py`: text + images, context facts by id.
- `visrag.py`: Arxiv/Plot/Slide/Doc – convert HF splits into `Sample` (with single image for slide/doc; multiple for plot if needed).
- `chartrag.py`: build `images` list and get `gt_answer` from Chart-MRAG.
- `mrag_bench.py`: populate `options` (A/B/C/D) and the MCQ ground truth letter.

Adapters are **pure data**: no generation inside. This keeps the pipeline testable.

---

## 🚀 Quickstart

### Install
```bash
pip install -e .
# or: pip install mragchecker  # when published
```

### Env (choose any backend)
```bash
# GPT (OpenAI)
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o            # or your Azure deployment name
# export OPENAI_BASE_URL=...          # Azure / OpenAI-compatible proxy

# Claude (Bedrock)
export AWS_REGION=us-east-1
# ensure your AWS credentials are configured (env or ~/.aws/credentials)

# vLLM (open-source)
# ensure vLLM is installed and GPU is available
```

### Run a tiny demo (text-only, GPT)
```bash
mragcheck gen \
  --dataset webqa --limit 16 \
  --backend gpt \
  --out runs/gpt/webqa.jsonl

mragcheck pack \
  --in-dir runs/gpt \
  --out examples/checking_inputs.json

mragcheck eval \
  --inputs examples/checking_inputs.json \
  --judge gpt \
  --metrics all \
  --out runs/gpt/checking_outputs.json
```

### Run a tiny demo (multimodal, vLLM llava)
```bash
mragcheck gen \
  --dataset visrag_arxiv --limit 8 \
  --backend llava \
  --out runs/llava/visrag_arxiv.jsonl

mragcheck pack --in-dir runs/llava --out examples/checking_inputs.json
mragcheck eval --inputs examples/checking_inputs.json --judge gpt --metrics all --out runs/llava/check.json
```

> The CLI is a thin wrapper over the pipeline functions. Internally it builds a `ChatModel` based on `--backend`.

---

## 🔧 CLI design (argparse or Typer)
```
mragcheck gen   --dataset {webqa,visrag_arxiv,visrag_plot,visrag_slide,visrag_doc,chartrag,mrag} \
                --backend {gpt,claude,llava,qwen,phi,pixtral,internvl3,mplug,deepseek} \
                [--limit N] [--max-tokens 512] [--temperature 0.2] \
                [--fewshot FILE.json] [--init-style ...] [--reasoning-process ...] \
                --out OUT.jsonl

mragcheck pack  --in-dir RUN_DIR --out checking_inputs.json

mragcheck eval  --inputs checking_inputs.json \
                --judge {gpt,claude,llava,qwen,...} \
                --metrics {all,text_only,multimodal,mcq} \
                --out checking_outputs.json
```

---

## 🔄 Migration guide from your current repo

1) **Create the package skeleton** using the layout above.
2) **Move your existing retrieval + prompt assembly** into `mragchecker/prompts/prompt_builder.py` and dataset adapters under `mragchecker/datasets/`.
3) **Replace ad‑hoc model calls** with `ChatModel` instances:
   - `OpenAIChatModel` (GPT)
   - `BedrockClaudeModel`
   - `VLLMModel` (configurable family key: llava/qwen/phi/...)
4) **Replace your `run_filter_rag_checker_prompt.py`** with the CLI entry `mragcheck gen` or keep it as a legacy script that just calls `pipelines.generate.run_generation`.
5) **Keep MRAG‑specific fields**: for MCQ, include `options` and keep your `extract_answer_letter` logic in `metrics/mcq_metrics.py`.
6) **Pack & evaluate** using the provided packer and evaluator, preserving the original input schema expected by your downstream RAGChecker flow.

> Minimal code edits in your legacy script:
> - Import `from mragchecker.backends import OpenAIChatModel, BedrockClaudeModel, VLLMModel` (or factory `build_model(kind=...)`).
> - Build `Sample` objects from your loaders.
> - Call `run_generation(samples, model, cfg)` and write JSONL.

---

## 🔐 Configuration
- Most secrets via environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AWS_REGION`, etc.).
- Optional YAML/JSON presets to map `--backend llava` → HF repo id; stored in `configs/backends.yaml`.

---

## 📏 Metrics (starter set)
- **Text**: EM, token‑F1, BLEU (optional), Rouge‑L (optional)
- **MCQ**: accuracy, letter extraction quality
- **Multimodal**: retrieval hit rate (if you log doc ids), **Path Edit Distance (PED)** vs a canonical path DSL (optional), step counts, correction‑after‑error rate.

> PED idea: define your ideal path in a lightweight DSL (`open(url) -> find(selector) -> click(id) -> extract(field)`), then compute edit distance between the system path and canonical path.

---

## 🧩 Extending backends
Add a new class implementing `ChatModel` and register it in a small factory:

```python
# mragchecker/backends/__init__.py
from .openai_backend import OpenAIChatModel
from .bedrock_backend import BedrockClaudeModel
from .vllm_backend import VLLMModel

def build_model(kind: str, **kwargs):
    kind = kind.lower()
    if kind in {"gpt", "openai"}:
        return OpenAIChatModel(**kwargs)
    if kind in {"claude", "bedrock"}:
        return BedrockClaudeModel(**kwargs)
    return VLLMModel(kind=kind, **kwargs)  # llava/qwen/phi/... via vLLM
```

---

## 📝 License
Choose a permissive license (MIT/Apache‑2.0) unless you have constraints. Note that some model weights (and APIs) have their own terms.

---

## 🤝 Contributing
- Keep comments in English.
- Add dataset adapters under `mragchecker/datasets/`.
- Add metrics in `mragchecker/metrics/` with unit tests.
- Backends should remain stateless (or lightly cached) and configured only via `GenerateConfig`/env.

---

## 📚 Roadmap
- Add batching and streaming support.
- Add structured judge outputs and confidence scoring.
- Add trace logging (prompt hash, model config snapshot) for full reproducibility.
- Optional web UI for browsing runs.
