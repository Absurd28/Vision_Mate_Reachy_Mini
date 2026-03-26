import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import os
import json

# --- TRAINING CONFIGURATION ---
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_DIR = './reachy_nlp_model/'
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class IntentDataset(Dataset):
    """PyTorch Dataset for handling the encoded text and intent labels."""
    def __init__(self, csv_file, tokenizer, max_len=64):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = self.df.iloc[idx]['label_id']
        
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    """Trains the DistilBERT classification head for intent detection."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"NLP Train: Starting training on device: {device}")

    # Load Label Mapping metadata
    if not os.path.exists("label_mapping.json"):
        print("Error: label_mapping.json not found. Run dataset_prep.py first.")
        return
        
    with open("label_mapping.json", "r") as f:
        mapping = json.load(f)
        num_labels = len(mapping)

    # 1. Initialize Tokenizer and Lightweight Model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(device)

    # 2. Load Processed CSV Data
    train_set = IntentDataset("train_data.csv", tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Standard PyTorch Training Loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=loss.item())

    # 4. Save Trained Weights and Tokenizer Config
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"NLP Train: Success. Model saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    train_model()
