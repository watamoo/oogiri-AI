"""大喜利データのロード・整形ユーティリティ

`YANS-official/ogiri-bokete` は train/ 直下に metadata.jsonl + 700枚の jpg を持つ。
text_to_text しか使わないので metadata.jsonl のみ DL し、画像はスキップする。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

# SFTで使うシステムプロンプト
SYSTEM_PROMPT = "あなたは大喜利の達人です。お題に対して、短く面白い回答を一つだけ返してください。"

HF_DATASET = "YANS-official/ogiri-bokete"
METADATA_FILE = "train/metadata.jsonl"


def fetch_oogiri_t2t() -> pd.DataFrame:
    """metadata.jsonl のみ取得 (画像はスキップ) → text_to_text を行展開して返す

    Returns:
        カラム [odai_id, odai, response_id, text, score] の DataFrame
    """
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=METADATA_FILE,
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows)
    df = df[df["type"] == "text_to_text"].reset_index(drop=True)

    # responses (list[dict]) を行展開
    df = df.explode("responses", ignore_index=True)
    df = pd.concat(
        [df.drop(columns=["responses"]), pd.json_normalize(df["responses"])],
        axis=1,
    )
    cols = ["odai_id", "odai", "response_id", "text", "score"]
    return df[cols]


def load_oogiri_t2t(
    eval_ratio: float = 0.1,
    seed: int = 42,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """text_to_text を取得 → top_k フィルタ → お題単位で train/eval 分割

    Args:
        eval_ratio: eval に回すお題の割合
        seed: 分割の乱数シード
        top_k: 各お題で score 上位 top_k 件のみ残す（None なら全件）

    Returns:
        (train_df, eval_df). カラムは [odai_id, odai, response_id, text, score]
    """
    df = fetch_oogiri_t2t()

    if top_k is not None:
        df = (
            df.sort_values("score", ascending=False)
            .groupby("odai_id", group_keys=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    # 同じお題の回答が train と eval に混ざらないよう、お題IDで分割
    # ArrowStringArray のままだと shuffle の警告が出るので numpy ndarray に変換
    odai_ids = np.asarray(df["odai_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(odai_ids)
    n_eval = max(1, int(len(odai_ids) * eval_ratio))
    eval_set = set(odai_ids[:n_eval])

    train_df = df[~df["odai_id"].isin(eval_set)].reset_index(drop=True)
    eval_df = df[df["odai_id"].isin(eval_set)].reset_index(drop=True)

    return train_df, eval_df


def to_chat_dataset(df: pd.DataFrame, system_prompt: str = SYSTEM_PROMPT) -> Dataset:
    """DataFrame を messages 形式 (ChatML想定) の HuggingFace Dataset に変換する"""
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["odai"]},
            {"role": "assistant", "content": row["text"]},
        ]
        for _, row in df.iterrows()
    ]
    return Dataset.from_dict({"messages": messages})
