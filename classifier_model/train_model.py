# train_router_from_lists.py
# -*- coding: utf-8 -*-
import os
import csv
import random
import pickle
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from examples import FAQ_EXAMPLES, ORDER_EXAMPLES, PERFUME_EXAMPLES, SMALLTALK_EXAMPLES

random.seed(42)
np.random.seed(42)


OUTPUT_MODEL_PATH = "router_clf.pkl"
OUTPUT_DATASET_CSV = "router_dataset.csv"
OUTPUT_REPORT_TXT = "report.txt"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 3000

# Мин. кол-во примеров на класс после балансировки
TARGET_PER_CLASS = None  # поставь число (например 1200), если хочешь жёстко добалансировать

# Лёгкие аугментации текста
def augment_text(s: str) -> str:
    t = s
    if random.random() < 0.25:
        t = t.lower()
    if random.random() < 0.15:
        t += "?"
    if random.random() < 0.10:
        t += "!"
    return t

# -----------------------------
# 3) Сборка DataFrame
# -----------------------------
def build_df() -> pd.DataFrame:
    rows = []
    rows += [(x, "FAQ") for x in FAQ_EXAMPLES]
    rows += [(x, "ORDER") for x in ORDER_EXAMPLES]
    rows += [(x, "PERFUME") for x in PERFUME_EXAMPLES]
    rows += [(x, "SMALLTALK") for x in SMALLTALK_EXAMPLES]
    df = pd.DataFrame(rows, columns=["text", "label"]).dropna()
    # убираем дубликаты
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"])
    return df

# Балансировка (оверсемпл до таргета)
def balance_df(df: pd.DataFrame, target_per_class: int = None) -> pd.DataFrame:
    if target_per_class is None:
        return df
    parts = []
    for label, g in df.groupby("label"):
        if len(g) >= target_per_class:
            parts.append(g.sample(target_per_class, random_state=RANDOM_STATE))
        else:
            # оверсемплинг с аугментацией
            need = target_per_class - len(g)
            aug_rows = []
            src = g["text"].tolist()
            for i in range(need):
                base = random.choice(src)
                aug_rows.append((augment_text(base), label))
            g_aug = pd.DataFrame(aug_rows, columns=["text", "label"])
            parts.append(pd.concat([g, g_aug], ignore_index=True))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out

# Аугментации «по мелочи»
def light_augment_df(df: pd.DataFrame, p: float = 0.35) -> pd.DataFrame:
    aug = []
    for _, r in df.iterrows():
        t = r["text"]
        if random.random() < p:
            aug.append((augment_text(t), r["label"]))
    if not aug:
        return df
    add = pd.DataFrame(aug, columns=["text", "label"])
    full = pd.concat([df, add], ignore_index=True)
    full = full.drop_duplicates(subset=["text"])
    return full.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# -----------------------------
# 4) Векторизация + модель
# -----------------------------
def make_pipeline() -> Pipeline:
    bow = ("bow", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        lowercase=True,
        token_pattern=r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’\-]+"
    ))
    charwb = ("charwb", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True
    ))
    feats = FeatureUnion([bow, charwb])
    clf = LogisticRegression(
        max_iter=MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto"
    )
    pipe = Pipeline([("feats", feats), ("clf", clf)])
    return pipe

# -----------------------------
# 5) Тренировка
# -----------------------------
def main():
    df = build_df()
    assert not df.empty, "Пустой датасет: заполни списки примеров."

    # Лёгкая аугментация и (опционально) балансировка
    df = light_augment_df(df, p=0.35)
    if TARGET_PER_CLASS:
        df = balance_df(df, TARGET_PER_CLASS)

    # Сохраним датасет
    df.to_csv(OUTPUT_DATASET_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✔ Saved dataset: {OUTPUT_DATASET_CSV} (N={len(df)})")
    print(df["label"].value_counts())

    # Трейн/вал
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"].tolist()
    )

    pipe = make_pipeline()
    print("✔ Training…")
    pipe.fit(X_train, y_train)

    print("✔ Evaluating…")
    y_pred = pipe.predict(X_val)
    report = classification_report(y_val, y_pred, digits=3)
    labels_order = ["FAQ", "ORDER", "PERFUME", "SMALLTALK"]
    cm = confusion_matrix(y_val, y_pred, labels=labels_order)

    print(report)
    print("Confusion matrix (labels: FAQ, ORDER, PERFUME, SMALLTALK):")
    print(cm)

    # Сохраняем отчёт
    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write("Confusion matrix (labels: FAQ, ORDER, PERFUME, SMALLTALK):\n")
        for row in cm:
            f.write(",".join(map(str, row)) + "\n")
    print(f"✔ Saved report: {OUTPUT_REPORT_TXT}")

    # Сохраняем модель
    labels_sorted = sorted(df["label"].unique().tolist())
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump((pipe, labels_sorted), f)
    print(f"✔ Saved model: {OUTPUT_MODEL_PATH} with labels: {labels_sorted}")

    # Проба предикта и вероятностей
    demo_samples = [
        "Є в наявності Royal Canin 2 кг?",
        "Как вернуть товар и какие документы нужны?",
        "Порадьте ванільно-амбровий аромат",
        "Привіт! Як ваш настрій сьогодні?"
    ]
    proba = pipe.predict_proba(demo_samples)
    print("\nDemo predictions:")
    for txt, probs in zip(demo_samples, proba):
        pred = pipe.predict([txt])[0]
        cls_idx = np.argsort(-probs)
        print(f"- {txt}\n  → pred={pred}, top={[(pipe.classes_[i], round(float(probs[i]),3)) for i in cls_idx[:3]]}")

if __name__ == "__main__":
    main()
