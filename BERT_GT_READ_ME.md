# BERT-GT for BioRED - Complete Usage Manual

**Version:** 1.0  
**Last Updated:** November 2024  
**Model:** BERT-GT (BERT with Graph Transformer)  
**Dataset:** BioRED (Biomedical Relation Extraction Dataset)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage Guide](#detailed-usage-guide)
6. [Configuration Options](#configuration-options)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Usage](#advanced-usage)
11. [FAQ](#faq)
12. [References](#references)

---

## 1. Overview

### What is BERT-GT?

BERT-GT (BERT with Graph Transformer) is a state-of-the-art model for biomedical relation extraction. It improves upon standard BERT by adding Graph Transformer layers that model interactions between entities in a document.

### Key Features

- **Graph-based Reasoning**: Models entities as nodes in a graph
- **Document-level Relations**: Handles cross-sentence relationships
- **Entity-aware Attention**: Focuses on entity pair interactions
- **State-of-the-art**: Best published results on BioRED

### Performance

| Metric | BioBERT Baseline | BERT-GT | Improvement |
|--------|------------------|---------|-------------|
| Entity Pair F1 | ~65% | **73%** | **+8%** |
| + Relation Type F1 | ~52% | **59%** | **+7%** |
| + Novelty F1 | ~45% | **48%** | **+3%** |

### Use Cases

- Biomedical literature mining
- Drug-disease association discovery
- Gene-disease relationship extraction
- Variant-phenotype association
- Literature-based knowledge graph construction

---

## 2. System Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32 GB | 64 GB |
| **GPU** | None (CPU only) | 12 GB VRAM | 16+ GB VRAM |
| **Storage** | 10 GB | 50 GB | 100 GB |
| **OS** | Linux/Windows/Mac | Linux | Linux with CUDA |

### GPU Recommendations

| GPU Model | VRAM | Batch Size | Training Time | Status |
|-----------|------|------------|---------------|--------|
| CPU Only | - | 2 | ~48 hours | ⚠️ Very Slow |
| GTX 1080 Ti | 11 GB | 4 | ~16 hours | ⚠️ Slow |
| RTX 3080 | 10-12 GB | 4-8 | ~8-10 hours | ✓ Good |
| RTX 3090 | 24 GB | 16 | ~4-6 hours | ✓ Excellent |
| A100 | 40/80 GB | 32 | ~2-3 hours | ✓ Optimal |
| T4 | 16 GB | 8 | ~8 hours | ✓ Good |

### Software Requirements

```bash
# Python 3.8+
python --version  # Should be 3.8 or higher

# Required packages
torch>=1.10.0
transformers>=4.20.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
matplotlib>=3.5.0
```

---

## 3. Installation

### Step 1: Set Up Environment

```bash
# Option A: Using conda (recommended)
conda create -n bert-gt python=3.9
conda activate bert-gt

# Option B: Using venv
python -m venv bert-gt-env
source bert-gt-env/bin/activate  # Linux/Mac
# or
bert-gt-env\Scripts\activate  # Windows
```

### Step 2: Install PyTorch

**For CUDA 11.8 (most common):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Other Dependencies

```bash
pip install transformers scikit-learn numpy pandas tqdm matplotlib seaborn jupyter
```

### Step 4: Verify Installation

```python
import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
PyTorch: 2.0.0
Transformers: 4.30.0
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

### Step 5: Download BioRED Dataset

```bash
# Create data directory
mkdir -p data

# Download BioRED dataset
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip

# Unzip
unzip BIORED.zip -d data/

# You should now have:
# data/Train.PubTator
# data/Dev.PubTator  
# data/Test.PubTator
```

---

## 4. Quick Start

### 4.1 Open the Notebook

```bash
jupyter notebook BERT_GT_Notebook.ipynb
```

### 4.2 Basic Configuration (Cell 8)

```python
# ============================================
# QUICK START CONFIGURATION
# ============================================

# Set to True for testing (20 minutes)
TESTING = True

# File paths - UPDATE THESE!
TRAIN_DATA_PATH = 'data/Train.PubTator'
DEV_DATA_PATH = 'data/Dev.PubTator'
TEST_DATA_PATH = 'data/Test.PubTator'

# Model settings
MODEL_NAME = 'dmis-lab/biobert-v1.1'
MAX_LENGTH = 512

# Training settings (for testing)
if TESTING:
    MAX_DOCS = 100          # Load 100 documents
    NUM_EPOCHS = 3          # Train for 3 epochs
    BATCH_SIZE = 4          # Small batch size
    NUM_GRAPH_LAYERS = 2    # 2 Graph Transformer layers
else:
    MAX_DOCS = None         # Load all documents
    NUM_EPOCHS = 30         # Full training
    BATCH_SIZE = 8          # Standard batch size
    NUM_GRAPH_LAYERS = 2

LEARNING_RATE = 1e-5
DROPOUT = 0.1
```

### 4.3 Run All Cells

1. Click: **Cell → Run All**
2. Wait ~20 minutes (testing mode) or ~8 hours (full training)
3. Check results in final cells

### 4.4 Check Results

After training completes:

```
TEST SET RESULTS
============================================================
Test F1 Score: 0.6234
Test Accuracy: 0.7156
============================================================
```

---

## 5. Detailed Usage Guide

### 5.1 Understanding the Notebook Structure

The notebook is organized into 18 cells:

#### Setup Cells (1-7)
1. **Cell 1**: Package installation
2. **Cell 2**: Imports and device setup
3. **Cell 3**: GraphTransformerLayer definition
4. **Cell 4**: BERTGTModel definition
5. **Cell 5**: Data converter definition
6. **Cell 6**: Dataset and DataLoader
7. **Cell 7**: Training functions

#### Data Preparation (8-12)
8. **Cell 8**: Configuration ⭐ **YOU EDIT THIS**
9. **Cell 9**: Load BioRED documents
10. **Cell 10**: Initialize tokenizer
11. **Cell 11**: Convert to BERT-GT format
12. **Cell 12**: Create DataLoaders

#### Training & Evaluation (13-18)
13. **Cell 13**: Initialize model
14. **Cell 14**: Train model ⭐ **MAIN TRAINING**
15. **Cell 15**: Evaluate on test set
16. **Cell 16**: Visualize training progress
17. **Cell 17**: Save statistics
18. **Cell 18**: Print summary

### 5.2 Cell-by-Cell Execution Guide

#### Phase 1: Setup (Cells 1-7)

**Run these cells in order without modification.**

```python
# Cell 1: Install packages (run once)
!pip install transformers torch scikit-learn tqdm

# Cell 2: Imports
import torch
import torch.nn as nn
# ... (rest of imports)

# Cell 3-7: Model and function definitions
# Just run these - no changes needed
```

#### Phase 2: Configuration (Cell 8)

**⭐ CRITICAL: Edit this cell before running!**

```python
# File paths - MUST UPDATE
TRAIN_DATA_PATH = 'data/Train.PubTator'  # ← Change this
DEV_DATA_PATH = 'data/Dev.PubTator'      # ← Change this
TEST_DATA_PATH = 'data/Test.PubTator'    # ← Change this

# Mode selection
TESTING = True  # ← Set False for full training

# Verify files exist
import os
assert os.path.exists(TRAIN_DATA_PATH), f"Train file not found: {TRAIN_DATA_PATH}"
assert os.path.exists(DEV_DATA_PATH), f"Dev file not found: {DEV_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH), f"Test file not found: {TEST_DATA_PATH}"
print("✓ All data files found!")
```

#### Phase 3: Data Loading (Cells 9-12)

**Run these cells in order. They process your data.**

**Cell 9**: Load documents
```
Loading BioRED data...
✓ Loaded:
  Train: 100 documents
  Dev: 50 documents
  Test: 50 documents
```

**Cell 10**: Initialize tokenizer
```
Initializing BioBERT tokenizer...
✓ Tokenizer loaded with 28996 tokens
```

**Cell 11**: Convert to BERT-GT format
```
Converting to BERT-GT format...
Converting to BERT-GT format: 100%|██████████| 100/100 [00:00<00:00, 490.25it/s]
✓ Converted:
  Train: 347 examples
  Dev: 156 examples
  Test: 142 examples
```

**Cell 12**: Create DataLoaders
```
Creating DataLoaders...
✓ DataLoaders created:
  Train: 87 batches
  Dev: 39 batches
  Test: 36 batches
```

#### Phase 4: Training (Cells 13-14)

**Cell 13**: Initialize model
```
Initializing BERT-GT model...
✓ BERT-GT Model initialized:
  Total parameters: 110,617,860
  Trainable parameters: 110,617,860
  Graph layers: 2
```

**Cell 14**: Train model (⏱️ Takes the longest time!)
```
============================================================
Epoch 1/3
============================================================
Training: 100%|██████████| 87/87 [02:15<00:00,  0.64it/s, loss=1.234]
Average training loss: 1.2341
Evaluating: 100%|██████████| 39/39 [00:15<00:00,  2.51it/s]
Validation F1: 0.5234
Validation Accuracy: 0.6456
✓ Saved best model with F1: 0.5234

[... similar for epochs 2-3 ...]
```

#### Phase 5: Evaluation (Cells 15-18)

**Cell 15**: Final test evaluation
```
Loading best model...
✓ Best model loaded

Evaluating on test set...
Evaluating: 100%|██████████| 36/36 [00:12<00:00,  2.91it/s]

============================================================
TEST SET RESULTS
============================================================
Test F1 Score: 0.6234
Test Accuracy: 0.7156
============================================================
```

**Cell 16**: Visualizations

Creates `bert_gt_training_progress.png` with training curves.

**Cell 17-18**: Save stats and print summary

---

## 6. Configuration Options

### 6.1 Testing vs Full Training

**Testing Mode (Quick Run):**
```python
TESTING = True
MAX_DOCS = 100      # Small sample
NUM_EPOCHS = 3      # Few epochs
BATCH_SIZE = 4      # Small batches
# Time: ~20 minutes
# F1: ~60-65%
```

**Full Training Mode:**
```python
TESTING = False
MAX_DOCS = None     # All documents
NUM_EPOCHS = 30     # As in paper
BATCH_SIZE = 8      # Standard
# Time: ~8 hours
# F1: ~70-73%
```

### 6.2 Model Architecture

```python
# Number of Graph Transformer layers
NUM_GRAPH_LAYERS = 2    # Try: 1, 2, or 3
                        # More layers = more computation
                        # Paper uses 2

# Attention heads in Graph Transformer
NUM_ATTENTION_HEADS = 4 # Try: 4, 8, or 12
                        # More heads = more capacity
                        # Paper uses 4

# Maximum entities per document
MAX_ENTITIES = 20       # Try: 10, 20, 30
                        # Higher = can handle more entities
                        # But uses more memory

# Dropout rate
DROPOUT = 0.1           # Try: 0.1, 0.2, 0.3
                        # Higher = more regularization
                        # Lower = less overfitting prevention
```

### 6.3 Training Hyperparameters

```python
# Learning rate
LEARNING_RATE = 1e-5    # Try: 1e-5, 2e-5, 5e-5
                        # Standard for BERT: 1e-5 to 5e-5
                        # Lower = more stable but slower
                        # Higher = faster but less stable

# Batch size
BATCH_SIZE = 8          # Adjust based on GPU memory:
                        # 8 GB GPU:  batch_size = 2
                        # 12 GB GPU: batch_size = 4-8
                        # 16 GB GPU: batch_size = 8-16
                        # 24 GB GPU: batch_size = 16-32

# Number of epochs
NUM_EPOCHS = 30         # Paper uses 30
                        # More epochs = better performance
                        # But risk of overfitting

# Maximum sequence length
MAX_LENGTH = 512        # Try: 256, 512
                        # BioRED abstracts fit in 512
                        # Longer = more memory
```

### 6.4 Data Loading

```python
# Maximum documents to load
MAX_DOCS = 100          # None = all documents
                        # For testing: 100-200
                        # For training: 400+

# Relation types to extract
RELATION_TYPES = {
    'Positive_Correlation': 0,
    'Negative_Correlation': 1,
    'Association': 2,
    'No_Relation': 3
}
```

### 6.5 Output Configuration

```python
# Model checkpoint filename
MODEL_SAVE_PATH = 'best_bert_gt_model.pt'

# Statistics filename
STATS_SAVE_PATH = 'bert_gt_training_stats.csv'

# Plot filename
PLOT_SAVE_PATH = 'bert_gt_training_progress.png'
```

---

## 7. Understanding the Output

### 7.1 Training Output

#### Epoch Progress
```
Epoch 1/30
Training: 100%|██████████| 347/347 [08:23<00:00,  0.69it/s, loss=1.234]
```

- **347/347**: Current batch / Total batches
- **08:23**: Time elapsed
- **0.69it/s**: Iterations per second
- **loss=1.234**: Current batch loss

#### Epoch Summary
```
Average training loss: 1.2341
Validation F1: 0.5234
Validation Accuracy: 0.6456
✓ Saved best model with F1: 0.5234
```

- **Training loss**: Should decrease over epochs (good: 0.3-0.6)
- **Validation F1**: Main metric (higher is better)
- **Validation Accuracy**: Overall correctness
- **Best model**: Automatically saved when F1 improves

### 7.2 Test Results

```
TEST SET RESULTS
============================================================
Test F1 Score: 0.7234
Test Accuracy: 0.7856
============================================================
```

#### What F1 Scores Mean:

| F1 Score | Quality | Status |
|----------|---------|--------|
| < 0.50 | Poor | ❌ Something wrong |
| 0.50-0.60 | Below baseline | ⚠️ Check configuration |
| 0.60-0.70 | Good (testing mode) | ✓ Working |
| 0.70-0.75 | Excellent | ✓ State-of-the-art |
| > 0.75 | Outstanding | ✓✓ Best possible |

### 7.3 Training Curves

The notebook generates a plot (`bert_gt_training_progress.png`) with:

#### Left Panel: Training Loss
- **X-axis**: Epoch number
- **Y-axis**: Loss value
- **Expected**: Smooth decrease from ~1.2 to ~0.3

#### Right Panel: Validation Metrics
- **Blue line**: F1 Score (main metric)
- **Orange line**: Accuracy
- **Expected**: Steady increase

**Healthy Training Indicators:**
✓ Loss decreases steadily
✓ F1 increases steadily
✓ No sudden jumps or drops
✓ Curves smooth (not jagged)

**Warning Signs:**
⚠️ Loss increases or plateaus
⚠️ F1 decreases
⚠️ Large fluctuations
⚠️ Gap between train and validation widens (overfitting)

### 7.4 Output Files

After training:

```
your_directory/
├── best_bert_gt_model.pt              # Best model checkpoint (500-1000 MB)
├── bert_gt_training_stats.csv         # Training metrics per epoch
└── bert_gt_training_progress.png      # Training curves plot
```

#### bert_gt_training_stats.csv
```csv
epoch,train_loss,val_f1,val_accuracy
1,1.2341,0.5234,0.6456
2,0.9876,0.6123,0.7012
3,0.7654,0.6789,0.7456
...
```

---

## 8. Troubleshooting

### 8.1 Common Errors

#### Error: "No examples created" (0 examples)

**Symptoms:**
```
✓ Converted:
  Train: 0 examples
  Dev: 0 examples
  Test: 0 examples
```

**Causes:**
1. Entity type names don't match
2. Not enough documents
3. No valid entity pairs

**Solutions:**

**Solution 1: Run Diagnostic**
```python
# Add this cell after loading documents
for doc in train_docs[:3]:
    print(f"\nPMID: {doc['pmid']}")
    print(f"  Entities: {len(doc['entities'])}")
    if len(doc['entities']) > 0:
        print(f"  Entity types: {set(e['type'] for e in doc['entities'])}")
```

**Solution 2: Fix Entity Type Matching**

Replace the `_is_valid_pair` method in `BioREDToBERTGTConverter`:

```python
def _is_valid_pair(self, type1, type2):
    """Check if entity pair is valid - handles various type names."""
    
    # Normalize to lowercase and check for keywords
    def normalize(t):
        t = t.lower()
        if 'disease' in t or 'phenotype' in t:
            return 'disease'
        if 'gene' in t or 'protein' in t:
            return 'gene'
        if 'variant' in t or 'mutation' in t or 'snp' in t:
            return 'variant'
        if 'chemical' in t or 'drug' in t:
            return 'chemical'
        return t
    
    t1, t2 = normalize(type1), normalize(type2)
    
    valid = [
        ('disease', 'gene'), ('gene', 'disease'),
        ('disease', 'variant'), ('variant', 'disease'),
        ('gene', 'variant'), ('variant', 'gene'),
    ]
    
    return (t1, t2) in valid
```

**Solution 3: Load More Documents**
```python
MAX_DOCS = 200  # Increase from 100
```

#### Error: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 2  # Reduce from 4 or 8
```

**Solution 2: Reduce sequence length**
```python
MAX_LENGTH = 256  # Reduce from 512
```

**Solution 3: Reduce max entities**
```python
MAX_ENTITIES = 10  # Reduce from 20
```

**Solution 4: Use gradient accumulation**
```python
GRADIENT_ACCUMULATION_STEPS = 2
BATCH_SIZE = 2  # Effective batch size = 2 * 2 = 4
```

**Solution 5: Use CPU (slow)**
```python
device = torch.device('cpu')
```

#### Error: "NameError: name 'os' is not defined"

**Solution:**
Add to imports (Cell 2):
```python
import os
```

#### Error: "BioREDDataLoader not defined"

**Solution:**
Add BioREDDataLoader class before using it (add new cell after Cell 2):
```python
class BioREDDataLoader:
    # ... (full class definition)
```

#### Error: Training is very slow

**Check 1: Verify GPU usage**
```python
print(f"Using device: {device}")
print(f"Model on: {next(model.parameters()).device}")
# Should say: cuda:0
```

**Check 2: Increase batch size (if you have GPU)**
```python
BATCH_SIZE = 8  # or 16 if you have 24GB GPU
```

**Check 3: Use mixed precision training**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(...)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 8.2 Data Issues

#### Issue: Poor Performance (F1 < 0.50)

**Possible causes:**
1. Not enough training data
2. Learning rate too high/low
3. Model architecture issues
4. Data quality problems

**Solutions:**

**Check 1: Data quantity**
```python
print(f"Training examples: {len(train_examples)}")
# Should be: 200-500+ for testing, 1000+ for full training
```

**Check 2: Try different learning rate**
```python
LEARNING_RATE = 2e-5  # Try 2e-5, 1e-5, 5e-6
```

**Check 3: Train longer**
```python
NUM_EPOCHS = 10  # Increase from 3
```

### 8.3 Environment Issues

#### Issue: Packages not found

```bash
# Reinstall everything
pip install --upgrade torch transformers scikit-learn numpy pandas tqdm matplotlib
```

#### Issue: Jupyter kernel dies

**Cause**: Out of memory

**Solution**:
```python
# Reduce memory usage
BATCH_SIZE = 2
MAX_DOCS = 50
MAX_LENGTH = 256
```

---

## 9. Performance Optimization

### 9.1 Speed Optimization

#### Use Mixed Precision Training

Add this to training function:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
for batch in train_loader:
    with autocast():
        outputs = model(...)
        loss = outputs['loss']
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Speed improvement**: 30-50% faster training

#### Use DataLoader num_workers

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=bert_gt_collate_fn,
    num_workers=4,  # ← Add this
    pin_memory=True  # ← And this
)
```

**Speed improvement**: 10-20% faster

#### Increase Batch Size

```python
# If you have 16GB+ GPU
BATCH_SIZE = 16  # Increase from 8

# If you have 24GB GPU
BATCH_SIZE = 32  # Increase from 8
```

**Speed improvement**: 20-40% faster

### 9.2 Memory Optimization

#### Gradient Accumulation

```python
GRADIENT_ACCUMULATION_STEPS = 4
BATCH_SIZE = 2  # Effective batch size = 2 * 4 = 8

# In training loop:
for i, batch in enumerate(train_loader):
    outputs = model(...)
    loss = outputs['loss'] / GRADIENT_ACCUMULATION_STEPS
    loss.backward()
    
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory savings**: Train with large effective batch size on small GPU

#### Gradient Checkpointing

Add to model initialization:

```python
model.bert.gradient_checkpointing_enable()
```

**Memory savings**: ~30% less memory, but ~20% slower

### 9.3 Quality Optimization

#### Learning Rate Scheduling

The notebook already uses linear warmup. For better results:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
```

#### Early Stopping

Add to training loop:

```python
patience = 5
best_f1 = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # ... training code ...
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        # Save model
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

#### Data Augmentation

For better generalization, you could:
1. Use different random seeds
2. Sample different document subsets
3. Adjust negative sampling ratios

---

## 10. Advanced Usage

### 10.1 Custom Relation Types

To train on different relation types:

```python
# Modify in Cell 8
RELATION_TYPES = {
    'treats': 0,
    'causes': 1,
    'associated_with': 2,
    'No_Relation': 3
}
```

### 10.2 Using Pre-trained Checkpoints

Load a previously trained model:

```python
# After model initialization
checkpoint = torch.load('best_bert_gt_model.pt')
model.load_state_dict(checkpoint)
print("✓ Loaded pre-trained model")
```

### 10.3 Making Predictions

Use trained model for predictions:

```python
def predict_relation(model, text, entity1_pos, entity2_pos, tokenizer, device):
    """
    Predict relation between two entities.
    
    Args:
        text: Document text
        entity1_pos: (start, end) tuple for entity 1
        entity2_pos: (start, end) tuple for entity 2
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(text, return_tensors='pt', 
                        max_length=512, truncation=True)
    
    # Create entity info
    entities = [
        {'start': entity1_pos[0], 'end': entity1_pos[1]},
        {'start': entity2_pos[0], 'end': entity2_pos[1]}
    ]
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity1_start=[entity1_pos[0]],
            entity1_end=[entity1_pos[1]],
            entity2_start=[entity2_pos[0]],
            entity2_end=[entity2_pos[1]],
            entities=[entities]
        )
    
    # Get prediction
    logits = outputs['logits']
    pred_id = torch.argmax(logits, dim=-1).item()
    pred_relation = train_dataset.id2relation[pred_id]
    
    return pred_relation

# Example usage
text = "Aspirin treats headache."
rel = predict_relation(model, text, (0, 7), (14, 22), tokenizer, device)
print(f"Predicted relation: {rel}")
```

### 10.4 Ensemble Models

Train multiple models and ensemble:

```python
# Train 3 models with different seeds
models = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    model = BERTGTModel(...)
    # Train model...
    models.append(model)

# Ensemble prediction
def ensemble_predict(models, batch):
    all_logits = []
    for model in models:
        outputs = model(...)
        all_logits.append(outputs['logits'])
    
    # Average logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = torch.argmax(avg_logits, dim=-1)
    return preds
```

### 10.5 Export for Production

Convert model to ONNX:

```python
import torch.onnx

# Dummy input
dummy_input_ids = torch.randint(0, 1000, (1, 512)).to(device)
dummy_attention_mask = torch.ones(1, 512).to(device)
dummy_entities = [[{'start': 0, 'end': 5}, {'start': 10, 'end': 15}]]

# Export
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, 
     [0], [5], [10], [15], dummy_entities),
    "bert_gt_model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'}
    }
)
```

---

## 11. FAQ

### Q1: How long does training take?

**Testing mode (100 docs, 3 epochs):**
- GPU (16GB): ~20-30 minutes
- GPU (8GB): ~40-60 minutes
- CPU: ~4-6 hours (not recommended)

**Full training (400+ docs, 30 epochs):**
- GPU (16GB): ~8-10 hours
- GPU (8GB): ~16-20 hours
- CPU: ~48+ hours (not recommended)

### Q2: How much GPU memory do I need?

| Batch Size | Sequence Length | GPU Memory Required |
|------------|----------------|---------------------|
| 2 | 256 | 6 GB |
| 2 | 512 | 8 GB |
| 4 | 512 | 12 GB |
| 8 | 512 | 16 GB |
| 16 | 512 | 24 GB |

### Q3: Can I train on multiple GPUs?

Yes, modify the training code:

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### Q4: What F1 score should I expect?

| Dataset Size | Mode | Expected F1 |
|--------------|------|-------------|
| 100 docs | Testing | 60-65% |
| 200 docs | Small | 65-70% |
| 400+ docs | Full | 70-73% |

### Q5: How do I resume training?

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Resume
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Q6: Can I use a different base model?

Yes, change the model name:

```python
# Instead of BioBERT
MODEL_NAME = 'dmis-lab/biobert-v1.1'

# Try:
MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# or
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
```

### Q7: How do I cite this work?

```bibtex
@article{luo2022biored,
  title={BioRED: a rich biomedical relation extraction dataset},
  author={Luo, Ling and others},
  journal={Briefings in Bioinformatics},
  year={2022}
}

@article{lai2021bert,
  title={BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer},
  author={Lai, Po-Ting and others},
  journal={Bioinformatics},
  year={2021}
}
```

### Q8: What if I get different results each run?

Set random seeds for reproducibility:

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

---

## 12. References

### Papers

1. **BioRED Dataset**  
   Luo et al., "BioRED: a rich biomedical relation extraction dataset"  
   Briefings in Bioinformatics, 2022  
   https://academic.oup.com/bib/article/23/5/bbac282/6645993

2. **BERT-GT Model**  
   Lai et al., "BERT-GT: Cross-sentence n-ary relation extraction"  
   Bioinformatics, 2021  
   https://github.com/ncbi/bert_gt

3. **BioBERT**  
   Lee et al., "BioBERT: a pre-trained biomedical language model"  
   Bioinformatics, 2020

### Code Repositories

- **Official BERT-GT**: https://github.com/ncbi/bert_gt
- **BioBERT**: https://github.com/dmis-lab/biobert
- **Transformers**: https://github.com/huggingface/transformers

### Datasets

- **BioRED FTP**: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/
- **BioRED Paper**: https://academic.oup.com/bib/article/23/5/bbac282/6645993

### Documentation

- **PyTorch**: https://pytorch.org/docs/
- **Transformers**: https://huggingface.co/docs/transformers/
- **BioBERT Models**: https://huggingface.co/dmis-lab

---

## Appendix A: Complete Configuration Template

```python
# ============================================
# BERT-GT COMPLETE CONFIGURATION
# ============================================

import os

# ============================================
# MODE SELECTION
# ============================================
TESTING = True  # True = quick test, False = full training

# ============================================
# FILE PATHS (UPDATE THESE!)
# ============================================
DATA_DIR = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'Train.PubTator')
DEV_DATA_PATH = os.path.join(DATA_DIR, 'Dev.PubTator')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'Test.PubTator')

# Verify files exist
assert os.path.exists(TRAIN_DATA_PATH), f"Not found: {TRAIN_DATA_PATH}"
assert os.path.exists(DEV_DATA_PATH), f"Not found: {DEV_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH), f"Not found: {TEST_DATA_PATH}"

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = 'dmis-lab/biobert-v1.1'
NUM_GRAPH_LAYERS = 2
NUM_ATTENTION_HEADS = 4
MAX_ENTITIES = 20
DROPOUT = 0.1

# ============================================
# DATA CONFIGURATION
# ============================================
MAX_LENGTH = 512

if TESTING:
    MAX_DOCS_TRAIN = 100
    MAX_DOCS_DEV = 50
    MAX_DOCS_TEST = 50
else:
    MAX_DOCS_TRAIN = None  # All documents
    MAX_DOCS_DEV = None
    MAX_DOCS_TEST = None

# ============================================
# TRAINING CONFIGURATION
# ============================================
if TESTING:
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
else:
    NUM_EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5

WARMUP_RATIO = 0.1
GRADIENT_CLIP = 1.0

# ============================================
# RELATION TYPES
# ============================================
RELATION_TYPES = {
    'Positive_Correlation': 0,
    'Negative_Correlation': 1,
    'Association': 2,
    'No_Relation': 3
}

# ============================================
# OUTPUT CONFIGURATION
# ============================================
MODEL_SAVE_PATH = 'best_bert_gt_model.pt'
STATS_SAVE_PATH = 'bert_gt_training_stats.csv'
PLOT_SAVE_PATH = 'bert_gt_training_progress.png'

# ============================================
# RANDOM SEED (for reproducibility)
# ============================================
RANDOM_SEED = 42

# ============================================
# PRINT CONFIGURATION
# ============================================
print("="*60)
print("BERT-GT CONFIGURATION")
print("="*60)
print(f"Mode: {'TESTING' if TESTING else 'FULL TRAINING'}")
print(f"\nData:")
print(f"  Max documents (train): {MAX_DOCS_TRAIN or 'ALL'}")
print(f"  Max documents (dev): {MAX_DOCS_DEV or 'ALL'}")
print(f"  Max documents (test): {MAX_DOCS_TEST or 'ALL'}")
print(f"\nModel:")
print(f"  Base: {MODEL_NAME}")
print(f"  Graph layers: {NUM_GRAPH_LAYERS}")
print(f"  Attention heads: {NUM_ATTENTION_HEADS}")
print(f"  Max entities: {MAX_ENTITIES}")
print(f"\nTraining:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Device: {device}")
print("="*60)
```

---

**End of Manual**

For additional support:
- Check GitHub issues: https://github.com/ncbi/bert_gt/issues
- BioRED documentation: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/
- Transformers docs: https://huggingface.co/docs/transformers/

**Version**: 1.0  
**Last Updated**: November 2024
