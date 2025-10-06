#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
from mlp_lm import NPLM, preprocess_data, train, evaluate, sample_from_mlp_lm
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

def create_synthetic_data():
    """Create synthetic text data for training"""
    synthetic_text = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence for language modeling.",
        "Machine learning is a fascinating field of artificial intelligence. Neural networks can learn complex patterns.",
        "Natural language processing helps computers understand human language. Tokenization is an important preprocessing step.",
        "Deep learning models require large amounts of data for training. The transformer architecture revolutionized NLP.",
        "Language models predict the next word in a sequence. They are trained on vast amounts of text data.",
        "Attention mechanisms allow models to focus on relevant parts of the input. This improves performance significantly.",
        "The BERT model uses bidirectional context for better understanding. GPT models generate text autoregressively.",
        "Subword tokenization helps handle out-of-vocabulary words. Byte-pair encoding is a popular method.",
        "Perplexity measures how well a language model predicts text. Lower perplexity indicates better performance.",
        "Cross-entropy loss is commonly used for training language models. It measures the difference between predicted and true distributions.",
        "The best perks of living on the east coast include beautiful beaches and vibrant cities.",
        "East coast cities offer diverse cultural experiences and excellent job opportunities.",
        "Living on the east coast provides access to historical landmarks and modern amenities.",
        "The east coast lifestyle combines urban sophistication with natural beauty.",
        "East coast residents enjoy four distinct seasons and varied landscapes."
    ]
    return {'text': synthetic_text}

def main():
    print("Creating synthetic training data...")
    train_data = create_synthetic_data()
    dev_data = create_synthetic_data()  # Use same data for simplicity
    
    print("Setting up tokenizer and splitter...")
    splitter = SentenceSplitter(language='en')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    print("Preprocessing data...")
    train_dataset = preprocess_data(train_data, local_window_size=6, splitter=splitter, tokenizer=tokenizer)
    dev_dataset = preprocess_data(dev_data, local_window_size=6, splitter=splitter, tokenizer=tokenizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = NPLM(tokenizer.vocab_size, embed_dim=128, local_window_size=6, hidden_dim=256, num_blocks=2, dropout_p=0.2)
    
    # Training configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.NLLLoss()
    
    print("Starting training...")
    epoch_train_losses, epoch_train_ppls, epoch_dev_losses, epoch_dev_ppls = train(
        model, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, 
        num_epochs=5, save_path='pretrained_fixed_window_lm.dat', print_every=5
    )
    
    print("Training completed! Now running sampling...")
    
    # Create config for sampling
    config = EasyDict({
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_blocks': 2,
        'dropout_p': 0.2,
        'local_window_size': 6,
        'save_path': 'pretrained_fixed_window_lm.dat',
        'batch_size': 8,
    })
    
    # Run sampling
    sample_from_mlp_lm(config, dev_data)

if __name__ == "__main__":
    main()


