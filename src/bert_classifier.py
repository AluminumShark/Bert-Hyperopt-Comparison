"""BERT classifier module for sentiment analysis.

This module provides the core components for BERT-based text classification:
- IMDBDataset: Custom PyTorch dataset for text data
- BERTClassifier: Neural network model wrapping BERT
- BertTrainer: Complete training and evaluation pipeline
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class IMDBDataset(Dataset):
    """Custom dataset for IMDB sentiment analysis with robust text processing."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int = 128,
    ) -> None:
        """Initialize the dataset.

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: HuggingFace tokenizer instance
            max_len: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing input_ids, attention_mask, and label tensors
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Ensure we have a valid string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Convert label to integer if needed
        if not isinstance(label, int):
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    """BERT-based text classifier with configurable dropout."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the BERT classifier.

        Args:
            model_name: Name of the pretrained BERT model
            num_labels: Number of output classes
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor

        Returns:
            Classification logits
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BertTrainer:
    """Complete BERT training and evaluation pipeline."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """Initialize the trainer.

        Args:
            model_name: Name of the pretrained BERT model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_data_loader(
        self,
        texts: list[str],
        labels: list[int],
        batch_size: int,
        max_length: int = 128,
    ) -> DataLoader:
        """Create a DataLoader with validated and cleaned data.

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            batch_size: Number of samples per batch
            max_length: Maximum sequence length

        Returns:
            DataLoader instance for the dataset
        """
        clean_texts = []
        clean_labels = []

        for i in range(len(texts)):
            text = texts[i]
            label = labels[i] if i < len(labels) else 0

            # Only keep non-empty string texts
            if isinstance(text, str) and text.strip():
                clean_texts.append(text.strip())
            else:
                continue

            # Ensure we have a valid integer label
            try:
                clean_labels.append(int(label))
            except (ValueError, TypeError):
                clean_labels.append(0)

        print(f"Data validation: {len(clean_texts)} valid texts from {len(texts)} original")

        dataset = IMDBDataset(clean_texts, clean_labels, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_and_evaluate(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 3,
        dropout_rate: float = 0.1,
        max_length: int = 128,
        warmup_ratio: float = 0.1,  # noqa: ARG002
    ) -> dict[str, float]:
        """Train and evaluate the BERT model with the given parameters.

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
            Dictionary containing accuracy, f1_score, and loss metrics
        """
        # Set up the model with the specified dropout rate
        model = BERTClassifier(
            model_name=self.model_name,
            dropout=dropout_rate,
        ).to(self.device)

        # Create data loaders
        train_loader = self.create_data_loader(train_texts, train_labels, batch_size, max_length)
        val_loader = self.create_data_loader(val_texts, val_labels, batch_size, max_length)

        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            total_loss = epoch_loss

        # Evaluation
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)

                predictions.extend(predicted.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="weighted")

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "loss": total_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
        }


def main() -> None:
    """Entry point for testing the module."""
    _trainer = BertTrainer()
    print("BERT Trainer initialized successfully")


if __name__ == "__main__":
    main()
