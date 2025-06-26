import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class IMBDDataset(Dataset):
    """Custom dataset for IMDB sentiment analysis with robust text processing."""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Make sure we have a valid string to work with
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Convert label to integer if needed
        if not isinstance(label, int):
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0  # Use 0 as default for invalid labels

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
class BERTClassifier(nn.Module):
    """BERT-based text classifier with configurable dropout."""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
    

class BertTrainer:
    """Complete BERT training and evaluation pipeline."""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_data_loader(self, texts, labels, batch_size, max_length=128):
        """Create a DataLoader with validated and cleaned data."""
        # Clean up the input data to avoid training issues
        clean_texts = []
        clean_labels = []
        
        for i in range(len(texts)):
            text = texts[i]
            label = labels[i] if i < len(labels) else 0
            
            # Only keep non-empty string texts
            if isinstance(text, str) and text.strip():
                clean_texts.append(text.strip())
            else:
                # Skip invalid text entries
                continue
                
            # Ensure we have a valid integer label
            try:
                clean_labels.append(int(label))
            except (ValueError, TypeError):
                clean_labels.append(0)
        
        print(f"Data validation: {len(clean_texts)} valid texts from {len(texts)} original")
        
        dataset = IMBDDataset(clean_texts, clean_labels, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_and_evaluate(self,
                           train_texts,
                           train_labels,
                           val_texts,
                           val_labels,
                           learning_rate=2e-5,
                           batch_size=16,
                           epochs=3,
                           dropout_rate=0.1,
                           max_length=128,
                           warmup_ratio=0.1):
        """
        Train and evaluate the BERT model with the given parameters.

        Args:
            train_texts: List of training text samples
            train_labels: List of training labels
            val_texts: List of validation text samples
            val_labels: List of validation labels
            learning_rate: Learning rate for the AdamW optimizer
            batch_size: Number of samples per training batch
            epochs: Number of training epochs to run
            dropout_rate: Dropout rate for the classification layer
            max_length: Maximum sequence length for tokenization
            warmup_ratio: Proportion of training steps for learning rate warmup
        
        Returns:
            dict: Training results including accuracy, f1_score, and loss
        """

        # Set up the model with the specified dropout rate
        model = BERTClassifier(
            model_name=self.model_name,
            dropout=dropout_rate
        ).to(self.device)
        
        # Create data loaders for training and validation
        train_loader = self.create_data_loader(train_texts, train_labels, batch_size, max_length)
        val_loader = self.create_data_loader(val_texts, val_labels, batch_size, max_length)

        # Set up the optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # Backward pass and parameter update
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        # Evaluation on validation set
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)

                predictions.extend(predicted.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        # Calculate final evaluation metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'loss': total_loss / len(train_loader)
        }
        

if __name__ == "__main__":
    trainer = BertTrainer()
    print("BERT Trainer initialized successfully")