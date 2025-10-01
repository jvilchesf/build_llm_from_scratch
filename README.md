# Build LLM From Scratch

A complete GPT-style language model implementation built from scratch using PyTorch, inspired by Sebastian Raschka's "Build a Large Language Model (From Scratch)" book. This project demonstrates the full transformer architecture pipeline from data processing to text generation.

## Overview

This implementation follows the complete LLM development pipeline:

1. **Data Processing** (`/services/model/dataset/`) - Text tokenization using GPT-2's tiktoken tokenizer, creating training sequences with sliding window approach
2. **Model Architecture** (`/services/model/`) - Full transformer implementation with multi-head self-attention, feed-forward networks, and layer normalization
3. **Training Pipeline** (`/services/train/`) - Complete training loop with loss calculation, evaluation metrics, and text generation during training
4. **Weight Management** (`/services/load_weights/`) - Integration with pre-trained GPT-2 weights for transfer learning or comparison
5. **Interactive Interface** - Chainlit web application for real-time text generation and model interaction

The architecture follows the original transformer design with causal masking for autoregressive text generation, implementing all components from scratch including attention mechanisms, positional embeddings, and GELU activations.

## Installation

```bash
uv sync
# or
pip install -e .
```

## Project Structure

### `/services/config.py`
Configuration settings using Pydantic. Defines model hyperparameters (embedding dimensions, attention heads, layers), training parameters (batch size, epochs), and dataset parameters (context length, stride).

### `/services/model/`
Core model implementation:

- **`gpt_model.py`** - Main GPT backbone class with token/positional embeddings, transformer blocks, and output head
- **`transformer/transformer.py`** - Single transformer block with layer normalization, multi-head attention, and feed-forward layers
- **`transformer/multihead_attention.py`** - Multi-head self-attention mechanism with causal masking
- **`utils/`** - Helper components:
  - `feedforward.py` - Feed-forward network with GELU activation
  - `gelu.py` - GELU activation function implementation
  - `layernorm.py` - Layer normalization

### `/services/model/dataset/`
Data handling:

- **`create_dataset.py`** - Dataset class for text tokenization and sequence creation
- **`create_data_loader.py`** - DataLoader creation for training and validation splits

### `/services/train/`
Training pipeline:

- **`main.py`** - Training script that combines model, optimizer, and data loaders
- **`train_iteration/train.py`** - Training loop implementation
- **`loss_calculation/loss.py`** - Loss computation utilities
- **`evaluation/evaluate_model.py`** - Model evaluation functions
- **`generate_text/generate.py`** - Text generation utilities with top-k sampling

### `/services/load_weights/`
Model weight management:

- **`gpt_download.py`** - Downloads pre-trained GPT-2 weights
- **`model_weights.py`** - Loads and initializes model with weights
- **`main.py`** - Chainlit web interface for interactive text generation

## Usage

### Train a model:
```bash
python services/train/main.py
```

### Run model inference:
```bash
python services/model/main.py
```

### Interactive web interface:
```bash
chainlit run services/load_weights/main.py
```

## Dependencies

- **PyTorch 2.8.0** - Deep learning framework
- **tiktoken** - GPT tokenizer
- **Chainlit** - Web interface for model interaction
- **Pydantic** - Configuration management