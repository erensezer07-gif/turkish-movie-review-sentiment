import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

# =========================
# 0) PROJE YOL AYARLARI
# =========================
# Bu dosya:  <root>/scripts/continue_train_from_current.py
# Root klasör = scripts'in bir üstü
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "Eski_Yedekler" / "dataset_3cls.parquet"
MODEL_PATH = ROOT_DIR / "yapay_zeka_servisi" / "benim_bert_modelim_3cls"
OUT_DIR = ROOT_DIR / "yapay_zeka_servisi" / "benim_bert_modelim_3cls_v2"

# =========================
# 1) HYPERPARAMS (GÜÇLENDİRME)
# =========================
set_seed(42)

MAX_LEN = 384          # 256 -> 384: uzun yorum kesilmesi azalır. (GPU yetmezse 256 yap)
BATCH = 8              # GPU RAM azsa 4 yap. Güçlü GPU varsa 16 yap.
EPOCHS = 2             # devam eğitimde 1-2 ideal
LR = 1e-5              # düşük LR = “ince ayar”
WARMUP_RATIO = 0.06

# Windows'ta bazen worker sorunu çıkarır, 0 güvenli
DATALOADER_WORKERS = 0

# =========================
# 2) DATA OKU
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model klasörü bulunamadı: {MODEL_PATH}")

df = pd.read_parquet(DATA_PATH).dropna()

TEXT_COL_CANDIDATES = ["Yorum", "text", "yorum", "review"]
LABEL_COL_CANDIDATES = ["label", "Label", "etiket", "sinif", "class", "Durum"]

def pick_col(_df, candidates):
    for c in candidates:
        if c in _df.columns:
            return c
    raise ValueError(f"Kolon bulunamadı. Mevcut kolonlar: {list(_df.columns)}")

text_col = pick_col(df, TEXT_COL_CANDIDATES)
label_col = pick_col(df, LABEL_COL_CANDIDATES)

df = df[[text_col, label_col]].dropna()
df[text_col] = df[text_col].astype(str)
df[label_col] = df[label_col].astype(int)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df[label_col]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# =========================
# 3) TOKENIZER + MODEL (MEVCUT CHECKPOINT'TEN)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

def tokenize(batch):
    return tokenizer(batch[text_col], truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column(label_col, "labels")
val_ds = val_ds.rename_column(label_col, "labels")
train_ds = train_ds.remove_columns([text_col])
val_ds = val_ds.remove_columns([text_col])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 4) CLASS WEIGHTS (NÖTR’Ü GÜÇLENDİRİR)
# =========================
y_train = train_df[label_col].values
classes = np.array([0, 1, 2])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels").to(model.device)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

# =========================
# 5) TRAIN ARGS
# =========================
args = TrainingArguments(
    output_dir=str(OUT_DIR),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,

    fp16=torch.cuda.is_available(),
    dataloader_num_workers=DATALOADER_WORKERS,
    logging_steps=100,
    report_to="none",
    save_total_limit=2,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

print("\n✅ Başlıyor...")
print("DATA_PATH:", DATA_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("OUT_DIR:", OUT_DIR)
print("CUDA:", torch.cuda.is_available())

trainer.train()

# =========================
# 6) RAPOR
# =========================
pred = trainer.predict(val_ds)
preds = np.argmax(pred.predictions, axis=1)
labels = pred.label_ids

print("\nCONFUSION MATRIX:")
print(confusion_matrix(labels, preds))
print("\nCLASSIFICATION REPORT:")
print(classification_report(labels, preds, digits=4))

OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"\n✅ Kaydedildi: {OUT_DIR}")
