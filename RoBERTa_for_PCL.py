import pandas as pd
import numpy as np
import random, copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, \
    pipeline
from sklearn.metrics import f1_score
from sklearn.utils import resample
import nltk
from nltk.corpus import wordnet, stopwords

# Download NLTK data (WordNet for synonyms, stopwords list)
nltk.download('wordnet')
nltk.download('stopwords')

# Configuration
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
PATIENCE = 3
WARMUP_PROPORTION = 0.1
LR_CANDIDATES = [1e-5, 2e-5, 5e-5]
ENSEMBLE_SEEDS = [42, 52, 62]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Data augmentation: Synonym Replacement
stop_words = set(stopwords.words('english'))


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    # Identify candidate words (non-stopwords, alphabetic only)
    candidates = [i for i, w in enumerate(words) if w.lower() not in stop_words and w.isalpha()]
    random.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        if replaced >= n:
            break
        synonyms = set()
        for syn in wordnet.synsets(words[idx]):
            for lemma in syn.lemmas():
                if lemma.name().lower() != words[idx].lower():
                    synonyms.add(lemma.name().replace('_', ' '))
        if synonyms:
            new_words[idx] = random.choice(list(synonyms))
            replaced += 1
    return " ".join(new_words)


# # Data augmentation: Back-Translation (English -> Spanish -> English)
# translator_en_to_es = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device=-1)
# translator_es_to_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en", device=-1)
#
#
# def back_translate(sentence):
#     try:
#         es_text = translator_en_to_es(sentence, max_length=MAX_LENGTH)[0]['translation_text']
#         back_text = translator_es_to_en(es_text, max_length=MAX_LENGTH)[0]['translation_text']
#         return back_text
#     except Exception as e:
#         return sentence  # fallback to original if translation fails
#
#
# # Augment training data (apply to minority class samples)
# def augment_data(df):
#     augmented_texts = []
#     augmented_labels = []
#     for _, row in df[df['label'] == 1].iterrows():
#         text = row['paragraph']
#         # Synonym replacement augmentation
#         aug_text1 = synonym_replacement(text, n=1)
#         # Back-translation augmentation
#         aug_text2 = back_translate(text)
#         augmented_texts.extend([aug_text1, aug_text2])
#         augmented_labels.extend([1, 1])
#     aug_df = pd.DataFrame({'paragraph': augmented_texts, 'label': augmented_labels})
#     # Combine augmented samples with original data
#     return pd.concat([df, aug_df], ignore_index=True)
#
#
# # Oversample minority class to balance the dataset
# def oversample_data(df):
#     df_majority = df[df['label'] == 0]
#     df_minority = df[df['label'] == 1]
#     if len(df_minority) == 0 or len(df_majority) == 0:
#         return df
#     if len(df_minority) < len(df_majority):
#         df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=0)
#         df_balanced = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
#     else:
#         df_majority_upsampled = resample(df_majority, replace=True, n_samples=len(df_minority), random_state=0)
#         df_balanced = pd.concat([df_minority, df_majority_upsampled], ignore_index=True)
#     # Shuffle after oversampling
#     df_balanced = df_balanced.sample(frac=1, random_state=0).reset_index(drop=True)
#     return df_balanced


# Create DataLoader for a given DataFrame
def build_dataloader(df, tokenizer, batch_size, shuffle=False):
    texts = list(df['paragraph'])
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    if 'label' in df.columns:
        labels = torch.tensor(df['label'].values, dtype=torch.long)
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    else:
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


# Training function for one model
def train_model(train_loader, val_loader, learning_rate, num_epochs=NUM_EPOCHS, patience=PATIENCE, seed=42):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(WARMUP_PROPORTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            # Forward pass (with mixed precision)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss / len(train_loader):.4f} - Val F1: {val_f1:.4f}")
        # Check for improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights and return model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_f1


# Main execution
if __name__ == "__main__":
    # Load and preprocess dataset
    df = pd.read_csv('./data/dontpatronizeme_pcl.tsv', sep='\t', header=None,
                     names=["par_id", "art_id", "keyword", "country", "paragraph", "orig_label"], skiprows=4)
    df['label'] = df['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    try:
        dev_df = pd.read_csv('dev.txt', sep='\t', header=None,
                             names=["par_id", "art_id", "keyword", "country", "paragraph", "orig_label"])
        dev_df['label'] = dev_df['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
        train_df = df[~df['par_id'].isin(dev_df['par_id'])].reset_index(drop=True)
    except FileNotFoundError:
        from sklearn.model_selection import train_test_split

        train_df, dev_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=0)
        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
    try:
        test_df = pd.read_csv('test.txt', sep='\t', header=None,
                              names=["par_id", "art_id", "keyword", "country", "paragraph"])
    except FileNotFoundError:
        test_df = None

    print(train_df.head())
    print(train_df.info())
    # Augment and balance training data
    # train_df_aug = augment_data(train_df)
    # train_df_final = oversample_data(train_df_aug)
    train_df = train_df.dropna(subset=['paragraph'])
    train_df['paragraph'] = train_df['paragraph'].astype(str)

    for idx, row in train_df.iterrows():
        if not isinstance(row['paragraph'], str):
            print(f"Row {idx} has invalid type: {type(row['paragraph'])}, value = {row['paragraph']}")

    # Tokenize and create DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = build_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = build_dataloader(dev_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = build_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE,
                                   shuffle=False) if test_df is not None else None

    # Hyperparameter tuning (learning rate)
    best_lr = None
    best_val_f1 = 0.0
    print("Hyperparameter tuning for learning rate...")
    for lr in LR_CANDIDATES:
        print(f"Training with lr={lr}")
        model, val_f1 = train_model(train_loader, dev_loader, learning_rate=lr, num_epochs=3, patience=2, seed=42)
        print(f"Val F1 = {val_f1:.4f} for lr={lr}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_lr = lr
        # Free GPU memory before next trial
        del model
        torch.cuda.empty_cache()
    print(f"Best LR: {best_lr}, Val F1: {best_val_f1:.4f}")

    # Train ensemble models with best hyperparameters
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        print(f"Training model with seed {seed} (lr={best_lr})...")
        model, val_f1 = train_model(train_loader, dev_loader, learning_rate=best_lr, num_epochs=NUM_EPOCHS,
                                    patience=PATIENCE, seed=seed)
        print(f"Model seed {seed} Val F1 = {val_f1:.4f}")
        ensemble_models.append(model)

    # Ensemble predictions on validation set (majority voting)
    all_dev_preds = []
    for model in ensemble_models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        all_dev_preds.append(preds)
    # Transpose and vote
    all_dev_preds = np.array(all_dev_preds)  # shape: (num_models, num_examples)
    ensemble_dev_preds = []
    for j in range(all_dev_preds.shape[1]):
        votes = np.sum(all_dev_preds[:, j])
        ensemble_dev_preds.append(1 if votes > len(ensemble_models) / 2 else 0)
    dev_f1 = f1_score(dev_df['label'].values, ensemble_dev_preds, average='binary')
    print(f"Ensemble Validation F1 = {dev_f1:.4f}")

    # Ensemble predictions on test set
    ensemble_test_preds = []
    if test_loader is not None:
        all_test_preds = []
        for model in ensemble_models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    preds.extend(batch_preds)
            all_test_preds.append(preds)
        all_test_preds = np.array(all_test_preds)
        for j in range(all_test_preds.shape[1]):
            votes = np.sum(all_test_preds[:, j])
            ensemble_test_preds.append(1 if votes > len(ensemble_models) / 2 else 0)

    # Save predictions to files
    with open("predictions_dev.txt", "w") as f:
        for pid, pred in zip(dev_df['par_id'], ensemble_dev_preds):
            f.write(f"{pid}\t{pred}\n")
    if test_df is not None:
        with open("predictions_test.txt", "w") as f:
            for pid, pred in zip(test_df['par_id'], ensemble_test_preds):
                f.write(f"{pid}\t{pred}\n")
    print("Prediction files generated: predictions_dev.txt, predictions_test.txt")
