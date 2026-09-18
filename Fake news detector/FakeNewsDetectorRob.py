import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, set_seed
import numpy as np
import torch
import re

SEED = 3
set_seed(SEED)

LIAR_DIR = "./liar_dataset"
 
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context",
]
 
# Standard binary collapse of LIAR's 6-way truthfulness label.
LIAR_LABEL_MAP = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 0,
    "barely-true": 1,
    "false": 1,
    "pants-fire": 1,
}

fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

fake_df['label'] = 1
true_df['label'] = 0

liar_parts = []
for split_file in ("train.tsv", "test.tsv", "valid.tsv"):
    liar_part = pd.read_csv(
        f"{LIAR_DIR}/{split_file}", sep="\t", header=None, names=LIAR_COLUMNS
    )
    liar_parts.append(liar_part)
liar_df = pd.concat(liar_parts, ignore_index=True)
 
liar_df["label"] = liar_df["label"].map(LIAR_LABEL_MAP)
liar_df = liar_df.dropna(subset=["label"])
liar_df["label"] = liar_df["label"].astype(int)
liar_df["content"] = liar_df["statement"]
liar_df = liar_df[["content", "label"]]

fake_df['content'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
true_df['content'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')

fake_df = fake_df[['content', 'label']]
true_df = true_df[['content', 'label']]

fake_df["source"] = "isot_fake"
true_df["source"] = "isot_true"
liar_df["source"] = "liar"

df = pd.concat([fake_df, true_df, liar_df], ignore_index=True)
df  = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z.,\s]{0,40}\((Reuters|AP|AFP)\)\s*-\s*", flags=re.MULTILINE
)

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = DATELINE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['content'] = df['content'].apply(clean_text)
df = df.drop_duplicates(subset='content').reset_index(drop=True)

x = df['content']
y = df['label']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

def tokenize(texts):

    return tokenizer(
        texts, truncation=True,
        max_length=256
    )

train_encodings = tokenize(x_train.tolist())
val_encodings = tokenize(x_val.tolist())
test_encodings = tokenize(x_test.tolist())

class FakeNewsDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(
    train_encodings,
    y_train.tolist()
)
val_dataset = FakeNewsDataset(
    val_encodings, 
    y_val.tolist()
)
test_dataset = FakeNewsDataset(
    test_encodings,
    y_test.tolist()
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    seed=SEED
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted"
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()

model.save_pretrained("./fake_news_model")
tokenizer.save_pretrained("./fake_news_model")

print("Model saved successfully!")

#Final, one-time evaluation on the untouched test set.

test_results = trainer.evaluate(test_dataset)
print("\nFinal test-set metrics:", test_results)
 
raw_preds = trainer.predict(test_dataset)
y_pred = np.argmax(raw_preds.predictions, axis=-1)
y_prob_fake = torch.softmax(torch.tensor(raw_preds.predictions), dim=1)[:, 1].numpy()
 
print("\nConfusion matrix (rows=true, cols=pred), labels=[Real, Fake]:")
print(confusion_matrix(y_test, y_pred))
 
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
 
print("ROC-AUC:", roc_auc_score(y_test, y_prob_fake))
