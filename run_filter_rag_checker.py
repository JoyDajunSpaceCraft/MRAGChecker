import os
import json
import argparse
from types import SimpleNamespace
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from prompting.prompt_builder import (
    PromptConfig, RetrievedDoc, assemble_prompt, extract_answer_letter
)

# python run_filter_rag_checker.py  --ds-ids chartrag visrag_arxiv mrag visrag_doc webqa visrag_plot visrag_slide visual_rag --backend internvl3 --filter
# python run_filter_rag_checker.py  --ds-ids chartrag visrag_arxiv mrag visrag_doc webqa visrag_plot visrag_slide visual_rag --backend pixtral --filter
# python run_filter_rag_checker.py --ds-ids chartrag visrag_arxiv mrag visual_rag visrag_doc visrag_plot visrag_slide webqa --backend qwen --filter --limit 100
#  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_filter_rag_checker.py  --ds-ids chartrag visrag_arxiv webqa   --limit 2 --backend phi --no-filter 
from datasets import load_dataset
# from rag.models.basic_rag import generate_rag_claude, generate_rag_llava,generate_rag_qwen,generate_rag_llama3, encode_payloads, fetch_chart_text
from rag.models.basic_rag import (
    generate_rag_claude,
    generate_rag_llava,
    generate_rag_qwen,
    generate_rag_llama3,
    generate_rag_phi,
    generate_rag_pixtral,
    generate_rag_internvl3,
    generate_rag_mplug,
    generate_rag_deepseek,
    encode_payloads,
    fetch_chart_text,

)
# from rag.models.vllm_mm import run_vllm_mm
from utils.load_visrag import extract_arxivqa_with_full_answer, extract_query_image_answer
DATA_ROOT = "/home/ubuntu/efs/jyuelyu/mm/MM-RAGChecker-main"
WEBQA_JSON = os.path.join(DATA_ROOT, "dataset/webqa/WebQA_train_val.json")
    
with open(WEBQA_JSON, "r", encoding="utf-8") as f:
        _webqa = json.load(f)
VISRAG_SPLITS = {
    "visrag_arxiv": "openbmb/VisRAG-Ret-Test-ArxivQA",
    "visrag_plot":  "openbmb/VisRAG-Ret-Test-PlotQA",
    "visrag_slide": "openbmb/VisRAG-Ret-Test-SlideVQA",
    "visrag_doc":  "openbmb/VisRAG-Ret-Test-MP-DocVQA",
}
MRAG_DS = "uclanlp/MRAG-Bench"
MRAG_SPLIT = "test"

# 3. Helper functions

def load_visrag_samples(repo_name):
    """
    Load VisRAG query-id -> sample dict from either ArxivQA or other splits.
    """
    if repo_name.endswith("ArxivQA"):
        raw = extract_arxivqa_with_full_answer(repo_name)
    else:
        raw = extract_query_image_answer(repo_name)
    return { s["query-id"]: s for s in raw }

def load_inat_annotations(data_root):
    inat_annos = {}
    anno_path = os.path.join(data_root, "dataset", "inat_comp21", "v2_anno.jsonl")
    with open(anno_path, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            qid = rec["sn"]
            inat_annos[qid] = {
                "question": rec["question"],
                "answer": rec["answer"][0] if rec["answer"] else ""
            }
    return inat_annos

def load_filtered_ids(path: str):
    """
    Read one ID per line from a TXT or JSONL file.
    """
    if not path:
        return []
    ids = []
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".txt":
            ids = [l.strip() for l in f if l.strip()]
        else:
            for line in f:
                obj = json.loads(line)
                ids.append(obj["id"])
    return ids


def fetch_webqa_fact(tid: str):
    """
    Given WebQA fact ID, return the corresponding fact text.
    """
    qid, _, tag, idx = tid.split("_", 3)
    return _webqa[qid][f"txt_{tag}Facts"][int(idx)]["fact"]

# if 'inat_annos' not in globals():



def serialize_context(ctx_objs):
    return [{"doc_id": c.doc_id, "text": c.text} for c in ctx_objs]


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-root", required=True)
    parser.add_argument("--ds-ids", nargs="+", default=["chartrag"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["claude","llava","qwen", "llama","internvl3","phi", "pixtral","mplug","deepseek"], default="llava")
    parser.add_argument("--model-id", choices=[ "us.anthropic.claude-3-5-sonnet-20241022-v2:0", "llava"], default="llava")
    parser.add_argument("--output-dir", default="rag/llava/eval")
    parser.add_argument("--filter", dest="filter", action="store_true",
                        help="Apply filtered IDs (default)")
    parser.add_argument("--no-filter", dest="filter", action="store_false",
                        help="Use full dataset IDs (no filtering)")
    

    # parser for all different types of rag 
    parser.add_argument("--init-style", choices=["plain","expert","mcq_set","assistant_cot"], default="plain")
    parser.add_argument("--example-style", choices=["ex1","ex2","ex3","ex4","ex5","ex6"], default="ex1")
    parser.add_argument("--fewshot-json", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--answer-format", choices=["letter","the_answer_is_letter","json"], default="the_answer_is_letter")
    parser.add_argument("--reasoning-mode", choices=["direct","cot","auto"], default="auto")
    parser.add_argument("--context-order", choices=["img_first","text_first"], default="img_first")
    parser.add_argument("--include-doc-ids", action="store_true")
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument("--subject", type=str, default="general")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.set_defaults(filter=True)

    args = parser.parse_args()
    mode = "filtered" if args.filter else "full"

    fewshot_examples = []
    if args.fewshot_json:
        with open(args.fewshot_json, "r", encoding="utf-8") as f:
            fewshot_examples = json.load(f)

    args.output_dir = f"rag/{args.backend}_eval/{mode}"
    os.makedirs(args.output_dir, exist_ok=True)
    DATA_ROOT = "/home/ubuntu/efs/jyuelyu/mm/MM-RAGChecker-main"
    WEBQA_JSON = os.path.join(DATA_ROOT, "dataset/webqa/WebQA_train_val.json")
    CHART_JSONL = os.path.join(DATA_ROOT, "dataset/chartRAG/text_corpus.jsonl")
    inat_annos = load_inat_annotations(DATA_ROOT)
    PROJECT_ROOT = os.getcwd()
    FILTER_ROOT = os.path.join(PROJECT_ROOT, "filters", "filter_res")


    id2path = {
        "visrag_arxiv": os.path.join(FILTER_ROOT, "visrag_arxiv", "filtered_ids.txt"),
        "visrag_slide": os.path.join(FILTER_ROOT, "visrag_slide", "filtered_ids.txt"),
        "visrag_plot":  os.path.join(FILTER_ROOT, "visrag_plot",  "filtered_ids.txt"),
        "visrag_doc":   os.path.join(FILTER_ROOT, "visrag_doc",   "filtered_ids.txt"),
        "webqa": os.path.join(FILTER_ROOT, "webqa",         "filtered_ids.txt"),
        "mrag":         os.path.join(FILTER_ROOT, "mrag",  "hard_by_at_least_two.jsonl"),
        "visual_rag":   os.path.join(FILTER_ROOT, "visualrag",     "hard_by_any.jsonl"),
        "chartrag":     "",
    }


    # CharTRAG (Chart-MRAG)
    chart_ds = load_dataset("ymyang/Chart-MRAG", split="train")
    _chart_map = { ex["id"]: ex for ex in chart_ds }

    import datasets.features.features as _ff # only when transformers version <4.40 then it will helpful

    # VisRAG
    _visrag_img_map, _visrag_query_map = {}, {}
    for ds_name, hf_id in VISRAG_SPLITS.items():
        ds_img = load_dataset(hf_id, name="corpus", split="train")
        for ex in ds_img:
            _visrag_img_map[f"{hf_id}::{ex['corpus-id']}"] = ex['image']
        _ff._FEATURE_TYPES["List"] = _ff.Sequence
        ds_q = load_dataset(hf_id, name="queries", split="train")
        for ex in ds_q:
            _visrag_query_map[f"{hf_id}::{ex['query-id']}"] = {
                "query": ex['query'],
                "answer": ex['answer'],
                "options": ex.get('options'),
                # "corpus_id": ex['corpus-id'], 
            }


    # MRAG mapping
    ds_mrag = load_dataset(MRAG_DS, split=MRAG_SPLIT)
    _mrag_mapping = { str(ex['id']): ex for ex in ds_mrag }

    

    ds_to_ids = {}
    for spec in args.ds_ids:
        if args.filter:
            # ------------------------------
            # branch A: use filtered IDs
            # ------------------------------
            path = id2path[spec]
            ids  = load_filtered_ids(path)
            # if chartrag path empty, fallback to full chart IDs
            if spec == "chartrag" and not ids:
                ids = list(_chart_map.keys())
        else:
            # ------------------------------
            # branch B: use full dataset IDs
            # ------------------------------
            if spec == "chartrag":
                ids = list(_chart_map.keys())
            elif spec == "webqa":
                ids = list(_webqa.keys())
                ids = ids[:4000]
            elif spec == "visual_rag":
                ids = list(inat_annos.keys())
            elif spec in VISRAG_SPLITS:
                hf_id = VISRAG_SPLITS[spec]
                # extract query-id suffix from the composite key
                ids = [
                    key.split("::", 1)[1]
                    for key in _visrag_query_map.keys()
                    if key.startswith(f"{hf_id}::")
                ]
            elif spec == "mrag":
                ids = list(_mrag_mapping.keys())
            else:
                raise ValueError(f"Unknown dataset: {spec}")
        # apply limit if needed
        if args.limit is not None:
            ids = ids[: args.limit]
        ds_to_ids[spec] = ids

    for ds, id_list in ds_to_ids.items():
        print(f"\n=== Dataset: {ds}, #ids={len(id_list)} ===")
        store_raw_image = None
        store_ctx = None
        all_results = []
        for qid in tqdm(id_list):
            # ----- build query, context, raw_imgs based on ds -----
            # [Insert per-dataset retrieval logic here]
            query, ctx, raw_imgs = None, [], []  # replace with actual logic
            if ds == "webqa":
                query = _webqa[qid]["Q"]
                gt_answer=  _webqa[qid]["A"][0]
                # doc retrieve RAGContext
                # here use  retrieve_webqa return [(tid,score),...]
                from retriever.retrieve_mix import retrieve_webqa
                txt_hits, img_hits = retrieve_webqa(query, k_text=2, k_img=2)
                store_ctx = [{"doc_id": tid, "text": fetch_webqa_fact(tid)} for tid,_ in txt_hits]
                from types import SimpleNamespace

                ctx = [
                    SimpleNamespace(doc_id=tid, text=fetch_webqa_fact(tid))
                    for tid,_ in txt_hits
                ]
                # local file
                from rag.models.basic_rag import resolve_webqa_image_path
                raw_imgs = [ resolve_webqa_image_path(tid) for tid,_ in img_hits ] 
            elif ds == "visual_rag":
                from retriever.retrieve_mix import retrieve_visualrag
                # print(qid)
                base = qid.rsplit("_", 1)[0]

                sn = base.replace("_", " ")

                annotation = inat_annos.get(sn)
                if annotation is None:
                    print(f"No annotation found for ID {qid}")
                else:
                    # Access question and answer
                    question = annotation["question"]
                    answer   = annotation["answer"]
                    print("Question:", question)
                    print("Answer:  ", answer)
                if annotation is None:
                    # no annotation => skip
                    continue

                # human‐written question
                query = annotation["question"]

                gt_answer = annotation["answer"]
                hits = retrieve_visualrag(query, None, topk=2)
                raw_imgs = [path for path,_ in hits]
                print("raw_imgs in visual",raw_imgs)
            elif ds in VISRAG_SPLITS:
                hf_id = VISRAG_SPLITS[ds]
                
                # 1) query + answer
                vis_key = f"{hf_id}::{qid}"
                store_raw_image = [vis_key]
                sample = _visrag_query_map[vis_key]
                options_raw = sample.get("options") or []
                norm_opts = []
                import re
                for opt in options_raw:
                    m = re.match(r"^[A-Z][)\.] ?(.+)$", opt)
                    norm_opts.append(m.group(1) if m else opt)
                options_list = norm_opts
                query = sample["query"]
                options = sample["options"]
                if "arxiv" in ds:
                    letter = sample["answer"]
                    import re
                    opt_map = {}
                    for opt in options:
                        # match lines like "A) Some text..." or "B. Other text..."
                        m = re.match(r"^([A-Z])[)\.] ?(.+)$", opt)
                        if m:
                            opt_map[m.group(1)] = m.group(2)
            
                    # lookup the actual answer string (fallback to letter if missing)
                    gt_answer = opt_map.get(letter, letter)
                query+= " ".join(options) if options else ""
                gt_answer = sample["answer"]

                jpg_count = qid.count(".jpg")
                if jpg_count > 1:
                    print(f"[warn] malformed slide qid (has {jpg_count} .jpg), skipping: {qid}")
                    continue
                # 2) image
                # img_key = qid.split("-",1)[0]                # "6519.png-1" -> "6519.png"
                if ds == "visrag_slide":
                    # strip off the trailing "query_number_…" suffix
                    corpus_id = qid.rsplit("query_number_", 1)[0]
                    
                else:
                    # your existing logic for other splits
                    corpus_id = qid.split("-", 1)[0]
                img_map_key = f"{hf_id}::{corpus_id}"
                try:
                
                    pil = _visrag_img_map[img_map_key]
                except Exception as e:
                    print("exception in read image", e)
                    continue
                raw_imgs = [ pil ]
                ctx = []  # VisRAG 
                
            elif ds == "chartrag":
           
                sample   = _chart_map[qid]  
                query    = sample["query"]
                gt_answer= sample["gt_answer"]
                from retriever.retrieve_mix import retrieve_chart
                txt_hits, img_hits = retrieve_chart(query,k_text=2, k_img=2)
                store_ctx = [{"doc_id": tid, "text": fetch_chart_text(tid)} for tid,_ in txt_hits]
                from types import SimpleNamespace
                ctx = [
                    SimpleNamespace(doc_id=tid, text=fetch_chart_text(tid))
                    for tid,_ in txt_hits
                ]
                raw_imgs = [os.path.join("dataset/chartRAG/images", fn) for fn,_ in img_hits]
            
            elif ds == "mrag":
                entry = _mrag_mapping[qid]
                query = entry["question"]
                options_A = entry["A"]
                options_B = entry["B"]
                options_C = entry["C"]
                options_D = entry["D"]
                options_list = [entry["A"], entry["B"], entry["C"], entry["D"]]
                query = query + "A:" + options_A + "B:"+options_B + "C:" + options_C + "D:" + options_D
                pil = entry["image"]
                gt_answer = entry["answer"]
                # MRAG support image+text retrieve
                from retriever.retrieve_mix import retrieve_mrag
                hits = retrieve_mrag(pil, query, topk=2)
                ctx = []  # MRAG Bench has no snippet
                raw_imgs = [path for path,_ in hits]
            
            else:
                raise ValueError(f"Unknown dataset: {ds}")
            # encode images
            img_payloads = encode_payloads(raw_imgs)
            # generate
             # ---- unified dispatch just like in basic_rag.py ----
            if args.backend == "claude":
                resp = generate_rag_claude(
                    query=query or "",
                    ctx=ctx,
                    img_payloads=img_payloads,
                    max_tokens=args.max_new_tokens,
                )
            elif args.backend == "llava":
                resp = generate_rag_llava(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "qwen":
                resp = generate_rag_qwen(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "llama":
                resp = generate_rag_llama3(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "phi":
                resp = generate_rag_phi(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "pixtral":
                resp = generate_rag_pixtral(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "internvl3":
                resp = generate_rag_internvl3(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "mplug":
                resp = generate_rag_mplug(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            elif args.backend == "deepseek":
                resp = generate_rag_deepseek(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            # elif args.backend == "llava_next":
            #     resp = generate_rag_llava_next(query, ctx, raw_imgs, max_new_tokens=args.max_new_tokens)
            else:
                raise ValueError(f"Unsupported backend: {args.backend}")  # should never hit now

            # if args.backend == "claude":
            #     resp = generate_rag_claude(
            #         query=query or "",
            #         ctx=ctx,
            #         img_payloads=img_payloads,
            #         # model_id=args.model_id,
            #         # max_tokens=512,
            #         # temperature=0.0
            #     )
            # elif args.backend == "llava":
            #     resp = generate_rag_llava(query,
            #                                 ctx, 
            #                                 raw_imgs,
            #                                 max_new_tokens=1024)
            # elif args.backend == "qwen":
            #     resp = generate_rag_qwen(
            #         query,
            #         ctx,
            #         raw_imgs,
            #         max_new_tokens=512,
            #     )
            # elif args.backend == "llama":
            #     resp = generate_rag_llama3(
            #         query,
            #         ctx,
            #         raw_imgs,
            #         max_new_tokens=512,
            #     )            
            # else:
                raise ValueError(f"Unsupported backend: {args.backend}")
            print("resp",resp)
            # record
            record = {
                "dataset": ds,
                "query_id": qid,
                "gt_answer":gt_answer,
                "query": query,
                "retrieved_context": ctx,
                "retrieved_images":  store_raw_image if store_raw_image is not None else raw_imgs ,
                "rag_response": resp
            }
            all_results.append(record)
            


        serializable = []
        for rec in all_results:
            serializable.append({
                "query_id":        rec["query_id"],
                "query":           rec["query"],
                "gt_answer":       rec["gt_answer"],
                "rag_response":    rec["rag_response"],
                "retrieved_context":[
                    {"doc_id": c.doc_id, "text": c.text}
                    for c in rec["retrieved_context"]
                ],
                # if rec["retrieved_images"] is a list of file‐paths:
                "retrieved_images": rec["retrieved_images"],
            })

        out_path = os.path.join(args.output_dir, f"rag_generation_outputs_{ds}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in serializable:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("✅ Serializable results saved to", out_path)
if __name__ == "__main__":
    main()



#     python run_filter_rag_checker.py \
#   --ds-ids chartrag visrag_arxiv visrag_slide visrag_plot visrag_doc webqa visual_rag mrag \
#   [--limit N] \ if have this then will process
#   [--backend claude|llava|qwen|llama] \
#   [--no-filter] \ if filter then process with filter
#   [--output-dir path] not required
    # python run_filter_rag_checker.py  --ds-ids chartrag visrag_arxiv visrag_doc visrag_slide visrag_plot visual_rag  webqa   --limit 2 --backend llava 
