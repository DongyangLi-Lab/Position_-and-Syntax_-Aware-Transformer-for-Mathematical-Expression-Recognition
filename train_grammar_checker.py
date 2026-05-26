import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from model_parts.grammar_checker import GrammarChecker


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], List[str]]:
    """
    支持一行一个 token 的 vocab 文件。
    如果每行是 "token id" 或 "token\tid"，优先使用显式 id。
    """
    token2id = {}
    id2token_tmp = {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            line = raw.rstrip("\n")
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                token = " ".join(parts[:-1])
                idx = int(parts[-1])
            else:
                token = line.strip()
                idx = len(token2id)

            if token not in token2id:
                token2id[token] = idx
                id2token_tmp[idx] = token

    if not token2id:
        raise ValueError(f"Empty vocab file: {vocab_path}")

    max_id = max(id2token_tmp.keys())
    id2token = [""] * (max_id + 1)
    for idx, tok in id2token_tmp.items():
        id2token[idx] = tok

    # 补齐可能的空洞，避免 embedding size 不够
    for i, tok in enumerate(id2token):
        if tok == "":
            id2token[i] = f"<unused_{i}>"

    return token2id, id2token


def find_special_id(token2id: Dict[str, int], candidates: List[str], fallback: int = None) -> int:
    for tok in candidates:
        if tok in token2id:
            return token2id[tok]
    if fallback is not None:
        return fallback
    raise KeyError(f"Cannot find any special token from {candidates} in vocab")


def read_formulas(csv_path: str, formula_col: str = "formula", max_rows: int = None) -> List[str]:
    """
    读取 im2latexv2_train_cut.csv / test_cut.csv。

    优先读取名为 formula 的列。
    如果没有表头或列名不匹配，则自动选择最像公式的列：
        - 忽略 images/image/img/path/filename/id/index 等列
        - 优先选择包含反斜杠、花括号、空格分隔 token 的列
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)

    formulas = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        # 数据看起来是 csv/tsv 都可能，这里让 sniffer 判断，失败则默认逗号
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        return formulas

    header = rows[0]
    has_header = any(str(c).strip().lower() in {formula_col.lower(), "formula", "latex"} for c in header)

    start_idx = 1 if has_header else 0

    if has_header:
        lower = [str(c).strip().lower() for c in header]
        if formula_col.lower() in lower:
            col_idx = lower.index(formula_col.lower())
        elif "formula" in lower:
            col_idx = lower.index("formula")
        elif "latex" in lower:
            col_idx = lower.index("latex")
        else:
            col_idx = 0
    else:
        # 自动猜测公式列
        candidate_scores = []
        n_cols = max(len(r) for r in rows[: min(len(rows), 50)])
        for c in range(n_cols):
            values = [r[c] for r in rows[start_idx : min(len(rows), start_idx + 50)] if len(r) > c]
            if not values:
                continue
            joined = " ".join(values[:10])
            score = 0
            score += joined.count("\\") * 3
            score += joined.count("{") * 2
            score += joined.count("}") * 2
            score += joined.count(" ") // 3
            lower_val = joined.lower()
            if lower_val.endswith(".png") or ".png" in lower_val or ".jpg" in lower_val:
                score -= 100
            candidate_scores.append((score, c))
        if not candidate_scores:
            raise ValueError(f"Cannot infer formula column from {csv_path}")
        col_idx = max(candidate_scores)[1]

    for r in rows[start_idx:]:
        if max_rows is not None and len(formulas) >= max_rows:
            break
        if len(r) <= col_idx:
            continue
        formula = r[col_idx].strip()
        if not formula:
            continue
        # 跳过图片路径误读
        lower = formula.lower()
        if lower.endswith((".png", ".jpg", ".jpeg")):
            continue
        formulas.append(formula)

    return formulas


class PrefixCandidateDataset(Dataset):
    """
    从公式构造样本:
        正样本: prefix + 真实 next token
        负样本: prefix + 随机错误 candidate token

    label:
        1.0 = 合法/合理
        0.0 = 不合法/不合理
    """

    def __init__(
        self,
        formulas: List[str],
        token2id: Dict[str, int],
        id2token: List[str],
        pad_id: int,
        start_id: int,
        end_id: int,
        unk_id: int,
        max_prefix_len: int = 128,
        neg_per_pos: int = 3,
        max_samples: int = None,
        seed: int = 1234,
        hard_negative_ratio: float = 0.5,
    ):
        self.token2id = token2id
        self.id2token = id2token
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.max_prefix_len = max_prefix_len
        self.neg_per_pos = neg_per_pos
        self.rng = random.Random(seed)

        # 不作为负样本候选的 token
        self.forbidden_negative_ids = {pad_id, start_id}

        self.syntax_token_ids = [
            token2id[t]
            for t in ["{", "}", "^", "_", "\\frac", "\\sqrt", "\\left", "\\right", "(", ")", "[", "]"]
            if t in token2id
        ]

        normal_candidate_ids = [
            i for i in range(len(id2token))
            if i not in self.forbidden_negative_ids and id2token[i] and not id2token[i].startswith("<unused_")
        ]
        self.normal_candidate_ids = normal_candidate_ids

        samples = []
        for formula in formulas:
            ids = self.encode_formula(formula)
            # ids = [_START_, ..., _END_]
            for pos in range(1, len(ids)):
                prefix = ids[:pos]
                true_next = ids[pos]
                prefix = prefix[-max_prefix_len:]

                samples.append((prefix, true_next, 1.0))

                for _ in range(neg_per_pos):
                    neg = self.sample_negative(true_next, hard_negative_ratio)
                    samples.append((prefix, neg, 0.0))

                if max_samples is not None and len(samples) >= max_samples:
                    break

            if max_samples is not None and len(samples) >= max_samples:
                break

        self.samples = samples

    def encode_formula(self, formula: str) -> List[int]:
        tokens = formula.strip().split()
        ids = [self.start_id]
        ids.extend(self.token2id.get(tok, self.unk_id) for tok in tokens)
        ids.append(self.end_id)
        return ids

    def sample_negative(self, true_next: int, hard_negative_ratio: float) -> int:
        # 一部分负样本从结构 token 里抽，让模型学括号、上下标、frac 这类语法
        use_hard = self.syntax_token_ids and self.rng.random() < hard_negative_ratio
        pool = self.syntax_token_ids if use_hard else self.normal_candidate_ids

        if len(pool) <= 1:
            pool = self.normal_candidate_ids

        neg = true_next
        for _ in range(20):
            neg = self.rng.choice(pool)
            if neg != true_next and neg not in self.forbidden_negative_ids:
                return neg

        # fallback
        while neg == true_next or neg in self.forbidden_negative_ids:
            neg = self.rng.choice(self.normal_candidate_ids)
        return neg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, cand, label = self.samples[idx]
        return {
            "prefix": torch.tensor(prefix, dtype=torch.long),
            "candidate": torch.tensor(cand, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def collate_fn(batch, pad_id: int):
    max_len = max(x["prefix"].size(0) for x in batch)
    prefixes = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    candidates = torch.empty(len(batch), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.float)

    for i, item in enumerate(batch):
        p = item["prefix"]
        prefixes[i, : p.size(0)] = p
        candidates[i] = item["candidate"]
        labels[i] = item["label"]

    return prefixes, candidates, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    tp = fp = tn = fn = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    all_probs = []
    all_labels = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Evaluating", dynamic_ncols=True, leave=False)

    for prefixes, candidates, labels in iterator:
        prefixes = prefixes.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        logits = model(prefixes, candidates)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item()
        total += labels.numel()
        correct += (preds == labels).sum().item()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

        all_probs.extend(probs.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "num_samples": total,
    }


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    token2id, id2token = load_vocab(args.vocab)

    pad_id = find_special_id(token2id, ["_PAD_", "<pad>", "<PAD>", "[PAD]"], fallback=0)
    start_id = find_special_id(token2id, ["_START_", "<s>", "<bos>", "[BOS]"])
    end_id = find_special_id(token2id, ["_END_", "</s>", "<eos>", "[EOS]"])
    unk_id = find_special_id(token2id, ["_UNK_", "<unk>", "<UNK>", "[UNK]"], fallback=pad_id)

    print(f"Vocab size: {len(id2token)}")
    print(f"Special ids: pad={pad_id}, start={start_id}, end={end_id}, unk={unk_id}")

    train_formulas = read_formulas(args.train_csv, formula_col=args.formula_col, max_rows=args.max_train_rows)
    test_formulas = read_formulas(args.test_csv, formula_col=args.formula_col, max_rows=args.max_test_rows)

    print(f"Loaded train formulas: {len(train_formulas)}")
    print(f"Loaded test formulas:  {len(test_formulas)}")
    if train_formulas:
        print(f"Example formula: {train_formulas[0][:160]}")

    train_ds = PrefixCandidateDataset(
        formulas=train_formulas,
        token2id=token2id,
        id2token=id2token,
        pad_id=pad_id,
        start_id=start_id,
        end_id=end_id,
        unk_id=unk_id,
        max_prefix_len=args.max_prefix_len,
        neg_per_pos=args.neg_per_pos,
        max_samples=args.max_train_samples,
        seed=args.seed,
        hard_negative_ratio=args.hard_negative_ratio,
    )

    test_ds = PrefixCandidateDataset(
        formulas=test_formulas,
        token2id=token2id,
        id2token=id2token,
        pad_id=pad_id,
        start_id=start_id,
        end_id=end_id,
        unk_id=unk_id,
        max_prefix_len=args.max_prefix_len,
        neg_per_pos=args.neg_per_pos,
        max_samples=args.max_test_samples,
        seed=args.seed + 1,
        hard_negative_ratio=args.hard_negative_ratio,
    )

    print(f"Train prefix-candidate samples: {len(train_ds)}")
    print(f"Test prefix-candidate samples:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=args.device.startswith("cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=args.device.startswith("cuda"),
    )

    model = GrammarChecker(
        vocab_size=len(id2token),
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=pad_id,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = output_dir / "grammar_checker_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        start_time = time.time()

        iterator = enumerate(train_loader, start=1)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(train_loader),
                desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
                dynamic_ncols=True,
            )

        for step, (prefixes, candidates, labels) in iterator:
            prefixes = prefixes.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(prefixes, candidates)
            loss = criterion(logits, labels)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs

            avg_loss = running_loss / max(seen, 1)

            if tqdm is not None:
                iterator.set_postfix(train_loss=f"{avg_loss:.5f}")
            elif step % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                    f"train_loss={avg_loss:.5f}"
                )

        print("Evaluating on test set...")
        metrics = evaluate(model, test_loader, device)
        elapsed = time.time() - start_time

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={running_loss / max(seen, 1):.5f} "
            f"test_loss={metrics['loss']:.5f} "
            f"acc={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"time={elapsed:.1f}s"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab_path": args.vocab,
            "token2id": token2id,
            "id2token": id2token,
            "special_ids": {
                "pad_id": pad_id,
                "start_id": start_id,
                "end_id": end_id,
                "unk_id": unk_id,
            },
            "metrics": metrics,
        }

        torch.save(ckpt, output_dir / "grammar_checker_last.pt")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint to: {best_path}")

    print(f"Best F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="./data/im2latex-kaggle/im2latexv2_train_cut.csv")
    parser.add_argument("--test_csv", default="./data/im2latex-kaggle/im2latexv2_test_cut.csv")
    parser.add_argument("--vocab", default="./data/vocabs/im2latexv2.vocab")
    parser.add_argument("--formula_col", default="formula")
    parser.add_argument("--output_dir", default="./checkpoints/grammar_checker")

    parser.add_argument("--max_prefix_len", type=int, default=128)
    parser.add_argument("--neg_per_pos", type=int, default=3)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
