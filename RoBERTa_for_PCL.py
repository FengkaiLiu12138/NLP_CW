import pandas as pd
import re
import random
import numpy as np
import torch
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import wordnet, stopwords

# Global constant: maximum token length for RoBERTa
MAX_LENGTH = 256


def clean_text(text):
    """
    Clean the input text by removing non-alphabetic characters, extra spaces, and converting to lowercase.
    """
    text = str(text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def synonym_replace(sentence):
    """
    Replace one randomly selected non-stopword in the sentence with one of its synonyms from WordNet.
    """
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    if len(words) < 1:
        return sentence
    for _ in range(10):  # attempt up to 10 times
        idx = random.randrange(len(words))
        word = words[idx]
        if word.lower() in stop_words:
            continue
        syns = wordnet.synsets(word)
        synonyms = []
        for syn in syns:
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower():
                    synonyms.append(lemma_name)
        if synonyms:
            new_word = random.choice(synonyms)
            new_words = words.copy()
            new_words[idx] = new_word
            return " ".join(new_words)
    return sentence


class PCLDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Patronizing and Condescending Language (PCL) detection.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(pred):
    """
    Compute evaluation metrics (accuracy, precision, recall, F1) for the positive (PCL) class.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1)
    rec = recall_score(labels, preds, pos_label=1)
    f1 = f1_score(labels, preds, pos_label=1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    # --------------------- Data Loading and Preprocessing ---------------------
    # Load the dataset (adjust the file path if needed)
    df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t')

    # Clean the text and create a new column 'text_clean'
    df['text_clean'] = df['text'].apply(clean_text)

    # Convert multi-class labels (0-4) into binary labels:
    # 0 -> Non-PCL (labels 0 or 1) and 1 -> PCL (labels 2,3,4). Here, we mark any label > 0 as PCL.
    df['label_binary'] = (df['label'] > 0).astype(int)

    print("Full dataset class distribution:")
    print(df['label_binary'].value_counts())

    # --------------------- Train/Validation Split and Oversampling ---------------------
    # Split data into training (80%) and validation (20%) sets, stratified by the binary label.
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_binary'], random_state=42)

    # Oversample the minority class (PCL examples) in the training set.
    train_majority = train_df[train_df.label_binary == 0]
    train_minority = train_df[train_df.label_binary == 1]

    print("Training class distribution before oversampling:", train_df.label_binary.value_counts().to_dict())

    minority_count = len(train_minority)
    majority_count = len(train_majority)
    n_to_add = majority_count - minority_count  # number of minority examples needed

    if n_to_add > 0:
        train_minority_oversampled = train_minority.sample(n=n_to_add, replace=True, random_state=42)
        train_df = pd.concat([train_majority, train_minority, train_minority_oversampled], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Training class distribution after oversampling:", train_df.label_binary.value_counts().to_dict())

    # --------------------- Data Augmentation ---------------------
    # Augment minority (PCL) examples using synonym replacement.
    augmented_texts = []
    for text in train_df[train_df['label_binary'] == 1]['text_clean']:
        aug_text = synonym_replace(text)
        if aug_text and aug_text != text:
            augmented_texts.append((aug_text, 1))

    print(f"Generated {len(augmented_texts)} augmented examples using synonym replacement.")

    aug_df = pd.DataFrame(augmented_texts, columns=['text_clean', 'label_binary'])
    train_df_aug = pd.concat([train_df, aug_df], ignore_index=True)
    train_df_aug = train_df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Training set size after augmentation:", len(train_df_aug))
    print("Training class distribution after augmentation:", train_df_aug.label_binary.value_counts().to_dict())

    # --------------------- Tokenization ---------------------
    # Initialize the RoBERTa-large tokenizer.
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    train_encodings = tokenizer(
        list(train_df_aug['text_clean']),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    val_encodings = tokenizer(
        list(val_df['text_clean']),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    train_dataset = PCLDataset(train_encodings, list(train_df_aug['label_binary']))
    val_dataset = PCLDataset(val_encodings, list(val_df['label_binary']))

    print("Sample tokenized input (first 10 tokens):")
    print(train_dataset[0]['input_ids'][:10])
    print(train_dataset[0]['attention_mask'][:10])

    # --------------------- Model Setup and Training ---------------------
    # Load the pre-trained RoBERTa-large model for sequence classification.
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # suitable for an NVIDIA 3070Ti
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # high max epochs; early stopping will handle overfitting
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none"  # disable logging to external services
    )

    # Initialize Trainer with early stopping.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(monitor="eval_f1", patience=2)]
    )

    print("Starting model training...")
    trainer.train()
    print("Training completed.")

    # --------------------- Postprocessing: Threshold Optimization ---------------------
    # Get raw predictions on the validation set.
    val_outputs = trainer.predict(val_dataset)
    val_logits = val_outputs.predictions
    # Calculate probabilities for class 1 using softmax.
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
    val_labels = val_outputs.label_ids

    # Find the optimal probability threshold for class 1 based on F1 score.
    best_thresh = 0.5
    best_f1 = f1_score(val_labels, (val_probs >= 0.5).astype(int), pos_label=1)
    for t in np.linspace(0.0, 1.0, 101):
        preds = (val_probs >= t).astype(int)
        f1_temp = f1_score(val_labels, preds, pos_label=1)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = t
    print(f"Optimal probability threshold on validation set: {best_thresh:.2f} with F1 = {best_f1:.4f}")

    # --------------------- Evaluation ---------------------
    # Apply the optimal threshold to get final predictions.
    val_pred_labels = (val_probs >= best_thresh).astype(int)

    acc = accuracy_score(val_labels, val_pred_labels)
    prec = precision_score(val_labels, val_pred_labels, pos_label=1)
    rec = recall_score(val_labels, val_pred_labels, pos_label=1)
    f1_val = f1_score(val_labels, val_pred_labels, pos_label=1)

    print("\nValidation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"PCL Precision: {prec:.4f}")
    print(f"PCL Recall: {rec:.4f}")
    print(f"PCL F1-score: {f1_val:.4f}")

    # Display the confusion matrix.
    cm = confusion_matrix(val_labels, val_pred_labels)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(val_labels, val_pred_labels, target_names=['Non-PCL', 'PCL'], digits=4))

    # --------------------- Optional: Save Predictions ---------------------
    # For instance, save the validation (dev) predictions to a file dev.txt.
    with open("dev.txt", "w") as f:
        for pred in val_pred_labels:
            f.write(str(pred) + "\n")
    print("\nDev set predictions saved to dev.txt")

    # In a similar way, you could process a test set and save predictions to test.txt.


if __name__ == "__main__":
    main()
