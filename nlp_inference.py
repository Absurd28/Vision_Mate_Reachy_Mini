import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os

class CommandClassifier:
    """
    Inference wrapper for intent classification.
    Optimized for CPU-based real-time processing within the Reachy Mini platform.
    """
    def __init__(self, model_dir='./reachy_nlp_model/'):
        print(f"NLP Engine: Loading model from {model_dir}...")
        self.device = torch.device('cpu')
        
        # 1. Load Tokenizer and Trained Model Weights
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode

        # 2. Load the intent label mapping
        # Looks for label_mapping.json in the project root
        mapping_path = "label_mapping.json"
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file '{mapping_path}' not found. Did you run dataset_prep.py?")
            
        with open(mapping_path, "r") as f:
            self.label_mapping = json.load(f)

    def predict_intent(self, spoken_text):
        """
        Runs the model forward pass to identify the intent of a spoken string.
        Returns: (label_string, confidence_score)
        """
        with torch.no_grad(): # Disable gradient calculation for inference speed
            # Tokenize input text
            inputs = self.tokenizer(
                spoken_text, 
                add_special_tokens=True, 
                max_length=64,
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            mask = inputs['attention_mask'].to(self.device)

            # Perform inference
            outputs = self.model(input_ids, attention_mask=mask)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            # Map index to string label
            label = self.label_mapping[str(predicted_idx.item())]
            
            return label, confidence.item()

if __name__ == "__main__":
    # Quick Test Loop
    try:
        classifier = CommandClassifier()
        test_phrases = [
            "Reachy, please find the water bottle",
            "Stop everything now",
            "Move your arm to the left"
        ]
        
        for phrase in test_phrases:
            label, conf = classifier.predict_intent(phrase)
            print(f"Phrase: '{phrase}' -> Intent: {label} (Conf: {conf:.2f})")
    except Exception as e:
        print(f"NLP Inference Test Failed: {e}")
