import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def prepare_dataset():
    """
    Downloads and preprocesses an intent classification dataset from Kaggle.
    Requires kaggle.json in ~/.kaggle/ for authentication.
    """
    # 1. AUTHENTICATE KAGGLE
    print("NLP Data: Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    # 2. DOWNLOAD DATASET 
    # Using a high-quality intent dataset for virtual assistants
    dataset_name = "bitext/training-dataset-for-chatbots-virtual-assistants"
    download_path = "./data"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    print(f"NLP Data: Downloading {dataset_name}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    # 3. LOAD DATA
    # Looking for the main training file in the unzipped folder
    csv_file = os.path.join(download_path, "20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv")
    df = pd.read_csv(csv_file)
    
    # Selection of required columns
    df = df[['utterance', 'intent']]
    df.columns = ['text', 'label']

    # 4. ENCODE LABELS
    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df['label'])
    
    # Save label mapping for inference phase
    mapping = {int(k): v for k, v in dict(enumerate(label_encoder.classes_)).items()}
    with open("label_mapping.json", "w") as f:
        json.dump(mapping, f, indent=4)

    # 5. SPLIT DATA
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # 6. SAVE PROCESSED DATA
    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)
    print(f"NLP Data: Preprocessing complete. {len(train_df)} training samples ready.")

if __name__ == "__main__":
    prepare_dataset()
