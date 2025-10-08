import os
import json
import argparse
from glob import glob
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Dict, Any

from PIL import Image as PILImage, ImageDraw

# These follow your local package layout
from RAGChecker.ragchecker import RAGResult, RAGResults, RAGChecker
from RAGChecker.ragchecker.metrics import all_metrics


def _write_min_jsonl(input_dir: str) -> str:
    """Create a tiny JSONL file that matches the expected RAG schema."""
    os.makedirs(input_dir, exist_ok=True)

    img_dir = os.path.join(input_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    def _mk(text, path):
        img = PILImage.new("RGB", (320, 180), "white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, fill="black")
        img.save(path)

    img1 = os.path.join(img_dir, "ex1.png")
    img2 = os.path.join(img_dir, "ex2.png")
    _mk("A -> B", img1)
    _mk("X -> Y", img2)

    rows = [
        {
            "query_id": "ex-0001",
            "query": "Does any passage support A->B?",
            "gt_answer": "Yes",
            "rag_response": "The model states that A implies B.",
            "retrieved_context": [
                {"doc_id": "doc-1", "text": "Background about A and B."},
                {"doc_id": "doc-2", "text": "Evidence shows A -> B."}
            ],
            "retrieved_images": [img1]
        },
        {
            "query_id": "ex-0002",
            "query": "Is there a claim about X->Y?",
            "gt_answer": "No",
            "rag_response": {"content": [{"text": "Neutral."}]},
            "retrieved_context": [{"doc_id": "doc-3", "text": "No direct mention of X->Y."}],
            "retrieved_images": [img2]
        }
    ]

    fpath = os.path.join(input_dir, "rag_generation_outputs_min.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return fpath


def _resp_text(raw: Any) -> str:
    """Normalize rag_response to plain string."""
    if isinstance(raw, dict) and "content" in raw:
        return "".join(seg.get("text", "") for seg in raw["content"]).strip()
    return str(raw).strip()


def _load_rag_results(jsonl_path: str, limit: int | None = None) -> RAGResults:
    """Load JSONL into RAGResults. Supports local image paths."""
    results: List[RAGResult] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ctx = [
                SimpleNamespace(doc_id=c.get("doc_id"), text=c.get("text", ""))
                for c in r.get("retrieved_context", [])
            ]
            imgs = []
            for p in r.get("retrieved_images", []):
                if isinstance(p, str) and os.path.isfile(p):
                    img = PILImage.open(p)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    imgs.append(img)
            results.append(RAGResult(
                query_id=r["query_id"],
                query=r.get("query", ""),
                gt_answer=r.get("gt_answer", ""),
                response=_resp_text(r.get("rag_response", "")),
                retrieved_context=ctx,
                retrieved_images=imgs
            ))
    return RAGResults(results)


def _evaluate_both_modes(rag_results: RAGResults, eval_backend: str):
    """Run text and image modes with the same backend, adapting to older evaluators."""
    checker = RAGChecker(
        extractor_name=eval_backend,
        checker_name=eval_backend,
        multimodal_retrieval=True,
        batch_size_extractor=1,
        batch_size_checker=1
    )
    rag_text = deepcopy(rag_results)
    rag_img  = deepcopy(rag_results)

    has_text = any(len(r.retrieved_context) > 0 for r in rag_results.results)
    has_img  = any(len(r.retrieved_images) > 0 for r in rag_results.results)

    m_text = None
    m_img  = None

    # TEXT
    if has_text:
        try:
            m_text = checker.evaluate(rag_text, metrics=all_metrics, check_mode="text")
        except TypeError:
            if hasattr(checker, "evaluate_text"):
                m_text = checker.evaluate_text(rag_text, metrics=all_metrics)
            else:
                m_text = checker.evaluate(rag_text, metrics=all_metrics)

    # IMAGE
    if has_img:
        try:
            m_img = checker.evaluate(rag_img, metrics=all_metrics, check_mode="image")
        except TypeError:
            if hasattr(checker, "evaluate_image"):
                m_img = checker.evaluate_image(rag_img, metrics=all_metrics)
            else:
                # Some evaluators only do text; skip image
                m_img = None

    return m_text, m_img, rag_text, rag_img


def _save_results(mode: str, rag_results: RAGResults, metrics: Dict[str, Any], out_dir: str):
    """Save aggregated metrics + per-sample fields (compact)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{mode}.json")
    def _ser(rr: RAGResults):
        arr = []
        for r in rr.results:
            arr.append({
                "query_id": r.query_id,
                "query": r.query,
                "gt_answer": r.gt_answer,
                "response": r.response,
                "response_claims": r.response_claims,
                "gt_answer_claims": r.gt_answer_claims,
                "answer2response": r.answer2response,
                "response2answer": r.response2answer,
                "retrieved2response": r.retrieved2response,
                "retrieved2answer": r.retrieved2answer,
                # 🔧 add this block: materialize contexts for downstream training
                "retrieved_context": [
                    {"doc_id": getattr(c, "doc_id", None), "text": getattr(c, "text", "")}
                    for c in getattr(r, "retrieved_context", []) or []
                ],
                # (optional) keep image info if you need it later
                # "retrieved_images": getattr(r, "retrieved_images", []),
                "metrics": r.metrics,
            })
        return arr

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "metrics": metrics,
            "results": _ser(rag_results)
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {mode} results → {out_path}")


def _save_claims_dump(rag_results: RAGResults, out_path: str):
    """
    Save a line-oriented JSONL where each line contains all extracted claims
    and all pairwise checking labels. This is convenient for downstream analysis.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rag_results.results:
            row = {
                "query_id": r.query_id,
                "query": r.query,
                "gt_answer": r.gt_answer,
                "response": r.response,
                "response_claims": r.response_claims,
                "gt_answer_claims": r.gt_answer_claims,
                "answer2response": r.answer2response,
                "response2answer": r.response2answer,
                "retrieved2response": r.retrieved2response,
                "retrieved2answer": r.retrieved2answer,
                # 🔧 add contexts here too
                "retrieved_context": [
                    {"doc_id": getattr(c, "doc_id", None), "text": getattr(c, "text", "")}
                    for c in getattr(r, "retrieved_context", []) or []
                ],
            }

            # row = {
            #     "query_id": r.query_id,
            #     "query": r.query,
            #     "gt_answer": r.gt_answer,
            #     "response": r.response,
            #     "response_claims": r.response_claims,         # list[list[str]] or similar
            #     "gt_answer_claims": r.gt_answer_claims,       # list[list[str]] or similar
            #     "answer2response": r.answer2response,         # list[str] (labels)
            #     "response2answer": r.response2answer,         # list[str] (labels)
            #     "retrieved2response": r.retrieved2response,   # list[list[str]] per passage
            #     "retrieved2answer": r.retrieved2answer,       # list[list[str]] per passage
            # }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"📝 Saved claims & labels dump → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="tests/_min_input",
                        help="Where rag_generation_outputs_*.jsonl lives (auto-create a tiny one if empty).")
    parser.add_argument("--output-dir", default="tests/_min_output",
                        help="Where to save evaluation outputs.")
    parser.add_argument("--eval-backend", default="gpt-4o-mini",
                        help="Extractor/checker backend name (RAGChecker will route to OpenAI via your utils).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N lines from each input JSONL.")
    parser.add_argument("--claims-jsonl", default=None,
                        help="Optional path to save all extracted claims & labels as JSONL.")
    args = parser.parse_args()

    # Create a minimal JSONL if not present.
    paths = glob(os.path.join(args.input_dir, "rag_generation_outputs_*.jsonl"))
    if not paths:
        paths = [_write_min_jsonl(args.input_dir)]

    os.makedirs(args.output_dir, exist_ok=True)

    for path in paths:
        ds = os.path.basename(path).replace("rag_generation_outputs_", "").rsplit(".", 1)[0]
        base_out = os.path.join(args.output_dir, ds)
        print(f"\n=== Evaluating {ds} ===")

        rag_res = _load_rag_results(path, limit=args.limit)
        m_text, m_img, rag_text, rag_img = _evaluate_both_modes(rag_res, args.eval_backend)

        # Save aggregated results
        if m_text:
            _save_results("text", rag_text, m_text, os.path.join(base_out, "text_eval"))
            # Save claims dump if requested (use the richer object after evaluation)
            if args.claims_jsonl:
                # If user gives a directory, put a default file inside; else use as-is.
                dump_path = args.claims_jsonl
                if os.path.isdir(dump_path):
                    dump_path = os.path.join(dump_path, f"{ds}_claims_text.jsonl")
                _save_claims_dump(rag_text, dump_path)

        if m_img:
            _save_results("image", rag_img, m_img, os.path.join(base_out, "image_eval"))
            if args.claims_jsonl:
                dump_path = args.claims_jsonl
                if os.path.isdir(dump_path):
                    dump_path = os.path.join(dump_path, f"{ds}_claims_image.jsonl")
                _save_claims_dump(rag_img, dump_path)


if __name__ == "__main__":
    # Ensure key is present if your RAGChecker uses OpenAI under the hood.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please export OPENAI_API_KEY before running this script.")
    main()
