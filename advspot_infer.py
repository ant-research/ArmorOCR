#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, base64, csv, json, math, re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-prefix")
    p.add_argument("--metrics-file", type=Path)
    p.add_argument("--badcases-file", type=Path)
    p.add_argument("--model-arch", choices=["Qwen3-VL-8B", "Qwen3-VL-MoE"], default="Qwen3-VL-8B")
    p.add_argument("--system-message")
    p.add_argument("--image-patch-size", type=int, default=16)
    p.add_argument("--image-min-token-num", type=int, default=64)
    p.add_argument("--image-max-token-num", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for data-parallel inference")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--index-key", default="id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-badcases", action="store_true")
    p.add_argument("--keep-special-tokens", action="store_true")
    return p.parse_args()


def load_data(path):
    with path.open(encoding="utf-8") as f:
        try:
            value = json.load(f)
        except json.JSONDecodeError as e:
            if e.msg != "Extra data": raise
            f.seek(0); value = [json.loads(x) for x in f if x.strip()]
    if isinstance(value, list): return value
    if isinstance(value, dict) and isinstance(value.get("data"), list): return value["data"]
    if isinstance(value, dict) and all(isinstance(x, dict) for x in value.values()): return list(value.values())
    raise ValueError("test file must be JSONL, a JSON list, or a sample mapping")


def smart_resize(image, a):
    factor = a.image_patch_size * 2; w, h = image.size
    if min(w, h) <= 0 or max(w, h) / min(w, h) > 200: raise ValueError(f"invalid image size {w}x{h}")
    nh, nw = max(factor, round(h/factor)*factor), max(factor, round(w/factor)*factor)
    lo, hi = a.image_min_token_num*factor**2, a.image_max_token_num*factor**2
    if nh*nw > hi:
        s = math.sqrt(h*w/hi); nh, nw = max(factor, math.floor(h/s/factor)*factor), max(factor, math.floor(w/s/factor)*factor)
    elif nh*nw < lo:
        s = math.sqrt(lo/(h*w)); nh, nw = math.ceil(h*s/factor)*factor, math.ceil(w*s/factor)*factor
    return image.resize((nw, nh))


def open_image(source, a):
    from PIL import Image
    if source.startswith("data:image"):
        image = Image.open(BytesIO(base64.b64decode(source.split("base64,", 1)[1])))
    elif source.startswith(("http://", "https://")):
        import requests
        r = requests.get(source, timeout=60); r.raise_for_status(); image = Image.open(BytesIO(r.content))
    else: image = Image.open(source.removeprefix("file://"))
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, "white"); bg.paste(image, mask=image.getchannel("A")); image = bg
    else: image = image.convert("RGB")
    return smart_resize(image, a)


def sample_messages(sample):
    if "conversations" in sample:
        turns = sample["conversations"]; rk, tk = "from", "value"; roles = {"human":"user", "gpt":"assistant"}
    else:
        turns = sample["messages"]; rk, tk = "role", "content"; roles = {}
    if roles.get(turns[-1][rk], turns[-1][rk]) != "assistant": raise ValueError("last turn must be assistant GT")
    return turns[:-1], str(turns[-1][tk]), str(turns[0][tk]), rk, tk, roles


def model_inputs(sample, processor, a):
    turns, _, _, rk, tk, roles = sample_messages(sample)
    sources = sample.get("image", []); sources = [sources] if isinstance(sources, str) else list(sources)
    messages, images, n = [], [], 0
    if a.system_message is not None: messages.append({"role":"system", "content":a.system_message})
    for turn in turns:
        content = []
        for part in re.split(r"(<image>)", str(turn[tk])):
            if part == "<image>":
                if n >= len(sources): raise ValueError("missing image path")
                image = open_image(str(sources[n]), a); n += 1; images.append(image); content.append({"type":"image", "image":image})
            elif part.strip(): content.append({"type":"text", "text":part.strip()})
        messages.append({"role":roles.get(turn[rk], turn[rk]), "content":content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=text, images=images or None, patch_size=a.image_patch_size, do_resize=False, return_tensors="pt")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)


def infer(data, a, path, rank=0, world_size=1):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
    out = json.load(path.open()) if a.resume and path.is_file() else {}
    cls = Qwen3VLMoeForConditionalGeneration if a.model_arch == "Qwen3-VL-MoE" else Qwen3VLForConditionalGeneration
    if world_size > 1:
        torch.cuda.set_device(rank)
        device_map = {"": f"cuda:{rank}"}
    else:
        device_map = "auto"
    model = cls.from_pretrained(str(a.model), torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device_map)
    processor = AutoProcessor.from_pretrained(str(a.model), use_fast=True)
    shard = data[rank::world_size]
    for pos, sample in enumerate(shard, 1):
        sid = str(sample[a.index_key])
        if sid in out: continue
        try:
            inputs = model_inputs(sample, processor, a).to(model.device)
            ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            ids = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
            pred = processor.batch_decode(ids, skip_special_tokens=not a.keep_special_tokens, clean_up_tokenization_spaces=False)[0]
            out[sid] = {"pred":pred}; print(f"[GPU {rank} {pos}/{len(shard)}] {sid}: {pred}", flush=True)
        except Exception as e:
            out[sid] = {"pred":"", "error":repr(e)}; print(f"[GPU {rank} {pos}/{len(shard)}] {sid} failed: {e}", flush=True)
        if pos % a.save_interval == 0: save_json(path, out)
    save_json(path, out); return out


def rank_path(result_path, rank):
    return result_path.with_name(f"{result_path.stem}.rank{rank}{result_path.suffix}")


def gpu_worker(rank, world_size, a, result_path):
    data = load_data(a.data_file)
    infer(data, a, rank_path(Path(result_path), rank), rank, world_size)


def parallel_infer(data, a, result_path):
    import torch
    import torch.multiprocessing as mp
    available = torch.cuda.device_count()
    if a.num_gpus < 1: raise ValueError("--num-gpus must be at least 1")
    if available < a.num_gpus: raise RuntimeError(f"requested {a.num_gpus} GPUs, but only {available} are visible")
    if a.resume and result_path.is_file():
        with result_path.open(encoding="utf-8") as f: merged = json.load(f)
        for rank in range(a.num_gpus):
            subset = {str(s[a.index_key]): merged[str(s[a.index_key])] for s in data[rank::a.num_gpus] if str(s[a.index_key]) in merged}
            save_json(rank_path(result_path, rank), subset)
    print(f"Launching {a.num_gpus} GPU inference processes", flush=True)
    mp.spawn(gpu_worker, args=(a.num_gpus, a, str(result_path)), nprocs=a.num_gpus, join=True)
    merged = {}
    for rank in range(a.num_gpus):
        path = rank_path(result_path, rank)
        with path.open(encoding="utf-8") as f: merged.update(json.load(f))
    save_json(result_path, merged)
    return merged


def as_list(v):
    if isinstance(v, list): return v
    if not isinstance(v, str) or not v.strip(): return []
    try:
        x = ast.literal_eval(v); return x if isinstance(x, list) else [v]
    except Exception: return [v]


def norm(v): return re.sub(r"[^\u4e00-\u9fa50-9a-zA-Z]", "", str(v).lower())
def bbox(v):
    x = as_list(v)
    try: return [int(i) for i in x[:4]] if len(x) >= 4 else []
    except Exception: return []
def pred_bbox(v):
    m = re.search(r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?", v)
    return [int(x) for x in m.groups()] if m else []
def box_iou(a,b):
    if len(a)<4 or len(b)<4: return 0.0
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union else 0.0


def rows_for(data, results, key):
    rows=[]
    for s in data:
        _, gt, question, *_ = sample_messages(s); sid=str(s[key]); images=s.get("image", "")
        rows.append({"id":sid,"image":"\n".join(images) if isinstance(images,list) else images,"input":question,"think":"","pred":results.get(sid,{}).get("pred",""),"label":gt,"labels":str(s.get("labels",[])),"bbox":str(s.get("bbox",[]))})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate(rows):
    buckets=defaultdict(list); bad=[]
    for r in rows:
        raw="" if "<empty>" in r["pred"] else r["pred"]; p=norm(raw)
        gts=[norm(x) for x in as_list(r["label"])] or [norm(r["label"])]
        ok=any(x and x in p for x in gts); score=box_iou(pred_bbox(raw),bbox(r["bbox"]))
        buckets["total"].append((ok,score))
        for label in as_list(r["labels"]):
            label=str(label); buckets["l1::"+label.split("-",1)[0]].append((ok,score)); buckets["l2::"+label].append((ok,score))
        if not ok: bad.append(r)
    def calc(x):
        n=len(x); return {"accuracy":sum(a for a,_ in x)/n if n else 0,"num_correct":sum(a for a,_ in x),"num_samples":n,"bbox_mean_iou":sum(b for _,b in x)/n if n else 0}
    metrics={"total":calc(buckets["total"]),"l1":{},"l2":{}}
    for k,v in buckets.items():
        if k.startswith("l1::"): metrics["l1"][k[4:]]=calc(v)
        if k.startswith("l2::"): metrics["l2"][k[4:]]=calc(v)
    return metrics,bad


def print_metrics(metrics):
    total=metrics["total"]
    print("="*72)
    print("Total Results")
    print(f"Accuracy: {total['accuracy']:.4f} ({total['num_correct']}/{total['num_samples']}) | IoU: {total['bbox_mean_iou']:.4f}")
    print("="*72)
    print("Per-Label Results")
    print("-"*72)
    for l1 in sorted(metrics["l1"]):
        value=metrics["l1"][l1]
        print(f"{l1}: {value['accuracy']:.4f} ({value['num_correct']}/{value['num_samples']}) | IoU: {value['bbox_mean_iou']:.4f}")
        prefix=l1+"-"
        for full_label in sorted(metrics["l2"]):
            if full_label.startswith(prefix):
                value=metrics["l2"][full_label]
                print(f"    |- {full_label[len(prefix):]}: {value['accuracy']:.4f} ({value['num_correct']}/{value['num_samples']}) | IoU: {value['bbox_mean_iou']:.4f}")
    print("="*72)


def main():
    a=args_parser(); a.model=a.model.expanduser().resolve(); a.data_file=a.data_file.expanduser().resolve()
    if not a.model.is_dir(): raise SystemExit(f"model directory missing: {a.model}")
    if not a.data_file.is_file(): raise SystemExit(f"test file missing: {a.data_file}")
    data=load_data(a.data_file); outdir=(a.output_dir or a.data_file.parent).expanduser().resolve(); prefix=a.output_prefix or f"{a.model.name}#advspot"
    result,csvfile=outdir/f"{prefix}.json",outdir/f"{prefix}.csv"
    metrics_file=a.metrics_file.expanduser().resolve() if a.metrics_file else outdir/f"{prefix}#metrics.json"
    badfile=a.badcases_file.expanduser().resolve() if a.badcases_file else outdir/f"{prefix}#badcases.csv"
    if a.skip_generate:
        results=json.load(result.open())
    elif a.num_gpus > 1:
        results=parallel_infer(data,a,result)
    else:
        results=infer(data,a,result)
    rows=rows_for(data,results,a.index_key); write_csv(csvfile,rows); metrics,bad=evaluate(rows); save_json(metrics_file,metrics)
    if bad and not a.skip_badcases: write_csv(badfile,bad)
    print_metrics(metrics)
    print(f"Predictions: {result}\nCSV: {csvfile}\nMetrics: {metrics_file}\nBadcases: {badfile} ({len(bad)})")


if __name__ == "__main__": main()
