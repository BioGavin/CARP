"""Predict with the final CARP model, streaming over a FASTA file of any size.

One CSV row per sequence, in FASTA order:
    seq_id, seq, cls_prob, cls_label, EC_pMIC, EC_MIC(uM), SA_pMIC, SA_MIC(uM)
  cls_prob    P(AMP) from the shared classification head
  cls_label   "pos" if cls_prob >= 0.5, else "neg"
  <org>_pMIC  predicted -log10(MIC in uM) from that organism's regression head
  <org>_MIC   10 ** (-pMIC), in uM

The FASTA is parsed as a generator and every batch is appended to --out as soon as it is
predicted, so memory does not grow with the file. --resume continues a killed run: it drops a
half-written last row, then skips as many input sequences as the CSV already holds.

    python3 scripts/predict.py --ckpt carp_seed0.pt --fasta peptides.fasta --out preds.csv
    python3 scripts/predict.py --ckpt carp_seed0.pt --fasta peptides.fasta --out preds.csv --resume
"""
import argparse
import io
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch

from carp.dataset import ORGANISM_CODE
from carp.esmfeature import ESM_NAME, get_tokenizer
from carp.models import ESMMultiOrganismRegressor, _key

TRAIN_LEN = (5, 60)  # peptide length range of the regression training data


def code(org):
    return ORGANISM_CODE.get(org, _key(org))


def iter_fasta(path):
    """Yield (id, sequence) for each non-empty record; handles multi-line sequences."""
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if name is not None and seq:
                    yield name, "".join(seq)
                name, seq = (line[1:].split() or [""])[0], []
            else:
                seq.append(line.upper())
    if name is not None and seq:
        yield name, "".join(seq)


def trim_partial_row(path):
    """Cut a half-written last line off the CSV. -> (data rows, header columns, id of the last row)."""
    pos = end = n = 0
    header = last = None
    with open(path, "rb") as f:
        for line in f:
            pos += len(line)
            if not line.endswith(b"\n"):
                break
            end, n = pos, n + 1
            if n == 1:
                header = line.decode()
            else:
                last = line.decode()
    os.truncate(path, end)
    if header is None:
        return 0, None, None
    last = pd.read_csv(io.StringIO(header + (last or "")), dtype=str, keep_default_na=False)
    return max(n - 1, 0), last.columns.tolist(), (last["seq_id"].iloc[0] if len(last) else None)


def load_model(ckpt_path, esm_name):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    esm_name = esm_name or ckpt.get("args", {}).get("esm_name") or ESM_NAME
    model = ESMMultiOrganismRegressor(esm_name)
    for org in ckpt["organisms"]:
        model.add_organism(org)
    model._ensure_cls_head()
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["organisms"], esm_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="CARP checkpoint, e.g. carp_seed0.pt")
    ap.add_argument("--fasta", required=True, help="input FASTA file")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--resume", action="store_true",
                    help="continue --out where a previous run stopped")
    ap.add_argument("--esm-name", default=None, help="ESM-2 name (default: read from the checkpoint)")
    ap.add_argument("--max-len", type=int, default=64, help="tokenizer truncation length")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--flush-every", type=int, default=100000, help="fsync and report every N sequences")
    ap.add_argument("--limit", type=int, default=0, help="stop after N sequences (quick test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, organisms, esm_name = load_model(args.ckpt, args.esm_name)
    model.to(args.device)
    tok = get_tokenizer(esm_name)
    keys = [_key(o) for o in organisms]
    header = ["seq_id", "seq", "cls_prob", "cls_label"]
    for o in organisms:
        header += [f"{code(o)}_pMIC", f"{code(o)}_MIC(uM)"]
    print(f"loaded {os.path.basename(args.ckpt)} organisms={organisms} device={args.device} esm={esm_name}",
          file=sys.stderr, flush=True)

    records = iter_fasta(args.fasta)
    skip, mode = 0, "w"
    if os.path.exists(args.out):
        if not args.resume:
            sys.exit(f"{args.out} exists; use --resume to continue it or choose another --out")
        skip, got, last_id = trim_partial_row(args.out)
        if got is not None and got != header:
            sys.exit(f"{args.out} has a different header; not resuming")
        mode = "a" if got is not None else "w"
        if skip:
            for _ in range(skip - 1):
                if next(records, None) is None:
                    sys.exit(f"{args.out} has more rows than {args.fasta} has sequences; not resuming")
            rec = next(records, None)
            if rec is None or rec[0] != last_id:
                sys.exit(f"{args.out} does not match {args.fasta}; not resuming")
        print(f"[resume] {skip} rows already done", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, mode, newline="")
    if mode == "w":
        pd.DataFrame(columns=header).to_csv(fout, index=False, lineterminator="\n")

    @torch.no_grad()
    def predict(batch):
        enc = tok([s for _, s in batch], padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt").to(args.device)
        z = model.backbone(enc["input_ids"], enc["attention_mask"])
        logit = model.cls_head(z).cpu()
        prob = torch.sigmoid(logit)
        pmic = [model.reg_heads[k](z).cpu() for k in keys]
        df = pd.DataFrame({"seq_id": [n for n, _ in batch], "seq": [q for _, q in batch],
                           "cls_prob": [f"{v:.4f}" for v in prob.tolist()],
                           "cls_label": ["pos" if v >= 0 else "neg" for v in logit.tolist()]})
        for o, p in zip(organisms, pmic):
            df[f"{code(o)}_pMIC"] = [f"{v:.4f}" for v in p.tolist()]
            df[f"{code(o)}_MIC(uM)"] = [f"{10.0 ** -v:.4g}" for v in p.tolist()]
        df.to_csv(fout, header=False, index=False, lineterminator="\n")

    t0, done, since, odd, batch = time.time(), 0, 0, 0, []
    for rec in records:
        batch.append(rec)
        odd += not TRAIN_LEN[0] <= len(rec[1]) <= TRAIN_LEN[1]
        if len(batch) == args.batch_size:
            predict(batch)
            done, since, batch = done + len(batch), since + len(batch), []
            if since >= args.flush_every:
                fout.flush()
                os.fsync(fout.fileno())
                since = 0
                print(f"  {skip + done:,} done | {done / (time.time() - t0):,.0f} seq/s",
                      file=sys.stderr, flush=True)
        if args.limit and done + len(batch) >= args.limit:
            break
    if batch:
        predict(batch)
        done += len(batch)
    fout.flush()
    os.fsync(fout.fileno())
    fout.close()
    print(f"done: {done:,} sequences -> {args.out}", file=sys.stderr)
    if odd:
        print(f"warning: {odd} sequences were outside {TRAIN_LEN[0]}-{TRAIN_LEN[1]} residues, "
              f"the range of the training data", file=sys.stderr)


if __name__ == "__main__":
    main()
