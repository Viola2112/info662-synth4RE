# BERT-GT Notebook - Complete Usage Manual

**Version:** 1.2 FINAL  
**Last Updated:** December 2025  
**Model:** BERT-GT (BERT with Graph Transformer)  
**Dataset:** BioRED (Biomedical Relation Extraction Dataset)  
**PyTorch:** 2.6.0+cu124  
**CUDA:** 12.4  
**Notebook:** 36 cells (23 code, 13 markdown)

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Notebook Structure](#5-notebook-structure)
6. [Configuration Guide](#6-configuration-guide)
7. [Running the Notebook](#7-running-the-notebook)
8. [Understanding the Output](#8-understanding-the-output)
9. [Checkpoint System](#9-checkpoint-system)
10. [Troubleshooting](#10-troubleshooting)
11. [Advanced Usage](#11-advanced-usage)
12. [FAQ](#12-faq)

---

## 1. Overview

### What is BERT-GT?

BERT-GT (BERT with Graph Transformer) is a state-of-the-art model for biomedical relation extraction that improves upon standard BERT by adding Graph Transformer layers to model entity interactions.

### Key Features

- ✅ **Graph Transformer Layers**: Models entities as nodes in a graph
- ✅ **Document-level Reasoning**: Handles cross-sentence relationships
- ✅ **Entity-aware Attention**: Focuses on entity pair interactions
- ✅ **Checkpoint System**: Auto-save and auto-resume training
- ✅ **Comprehensive Metrics**: Precision, recall, F1 for all relation types

### Performance

| Model | Entity Pair F1 | Improvement |
|-------|----------------|-------------|
| BioBERT Baseline | ~65% | - |
| **BERT-GT** | **~73%** | **+8%** |

### This Manual

This manual is based on the **final approved version** of `BERT_GT_Notebook.ipynb` and provides complete instructions for:
- Installation and setup
- Running training and evaluation
- Using the checkpoint system
- Troubleshooting common issues
- Understanding results

---

## 2. System Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32 GB | 64 GB |
| **GPU** | 8 GB VRAM | 12 GB VRAM | 16+ GB VRAM |
| **Storage** | 20 GB | 50 GB | 100 GB |
| **OS** | Linux/Windows | Linux | Linux with CUDA 12.4 |

### GPU Recommendations

| GPU Model | VRAM | Batch Size | Training Time (30 epochs) | Status |
|-----------|------|------------|---------------------------|--------|
| CPU Only | - | 2 | ~100+ hours | ⚠️ Not Recommended |
| GTX 1080 Ti | 11 GB | 4 | ~40-50 hours | ⚠️ Slow |
| RTX 3080 | 10-12 GB | 4-8 | **~30-40 hours** | ✓ Good |
| RTX 3090 | 24 GB | 8-16 | ~20-25 hours | ✓ Excellent |
| A100 | 40/80 GB | 16-32 | ~10-15 hours | ✓ Optimal |
| T4 | 16 GB | 8 | ~30-35 hours | ✓ Good |

**Note:** Training times are for full training (30 epochs, 400+ documents, single GPU).

### Software Requirements

```bash
Python: 3.8+
PyTorch: 2.6.0+cu124
CUDA: 12.4
Transformers: 4.30.0+
Scikit-learn: 1.0.0+
NumPy: 1.21.0+
Pandas: 1.3.0+
tqdm: 4.62.0+
Matplotlib: 3.5.0+
```

---

## 3. Installation

### Step 1: Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n bert-gt python=3.9
conda activate bert-gt
```

**Option B: Using venv**
```bash
python -m venv bert-gt-env
source bert-gt-env/bin/activate  # Linux/Mac
# or
bert-gt-env\Scripts\activate  # Windows
```

### Step 2: Install PyTorch 2.6.0 with CUDA 12.4

**Cell 3 in Notebook:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only (not recommended for training):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Other Dependencies

**Cell 4 in Notebook:**
```bash
pip install transformers torch scikit-learn tqdm
```

**Additional packages for visualization:**
```bash
pip install matplotlib seaborn pandas jupyter
```

### Step 4: Verify Installation

**Cell 6 in Notebook:**
```python
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
```

**Expected Output:**
```
PyTorch version: 2.6.0+cu124
Transformers version: 4.30.0+
CUDA available: True
CUDA version in PyTorch: 12.4
GPU Name: NVIDIA GeForce RTX 3080
GPU Count: 1
```

### Step 5: Download BioRED Dataset

```bash
# Create data directory
mkdir -p data

# Download BioRED dataset
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip

# Unzip
unzip BIORED.zip -d data/

# Verify files exist
ls data/
# Should show: Train.PubTator  Dev.PubTator  Test.PubTator
```

### Step 6: Open Notebook

```bash
jupyter notebook BERT_GT_Notebook.ipynb
```

---

## 4. Quick Start

### 4.1 Three-Step Quick Start

1. **Open Notebook**
   ```bash
   jupyter notebook BERT_GT_Notebook.ipynb
   ```

2. **Update File Paths (Cell 19)**
   ```python
   TRAIN_DATA_PATH = './data/Train.PubTator'  # ← Update these!
   DEV_DATA_PATH = './data/Dev.PubTator'
   TEST_DATA_PATH = './data/Test.PubTator'
   ```

3. **Run All Cells**
   - Click: **Cell → Run All**
   - Testing mode: ~20-30 minutes
   - Full training: ~30-40 hours (1 GPU)

### 4.2 Testing Mode (Quick Verification)

Keep `TESTING = False` in Cell 19 to start:
```python
TESTING = False  # Change to True for quick test
```

For quick testing (20-30 minutes):
```python
TESTING = True
MAX_DOCS = 100    # Load 100 documents
NUM_EPOCHS = 3    # Quick training
BATCH_SIZE = 4    # Small batch
```

### 4.3 Expected Results

**Testing Mode (100 docs, 3 epochs):**
- F1 Score: ~60-65%
- Accuracy: ~70-75%
- Training time: 20-30 minutes

**Full Training (400+ docs, 30 epochs):**
- F1 Score: ~70-73%
- Accuracy: ~78-82%
- Training time: 30-40 hours (RTX 3080)

---

## 5. Notebook Structure

The notebook contains **36 cells** organized into **12 sections**:

### Section 1: Installation and Imports (Cells 1-7)

| Cell | Type | Purpose |
|------|------|---------|
| 1 | Markdown | Title and introduction |
| 2 | Markdown | Section header |
| 3 | Code | Install PyTorch 2.6.0 + CUDA 12.4 |
| 4 | Code | Install other dependencies |
| 5 | Code | Check GPU with nvidia-smi |
| 6 | Code | Import libraries |
| 7 | Code | Setup device and check versions |

**What it does:**
- Installs all required packages
- Imports necessary libraries
- Verifies GPU and CUDA availability
- Sets up device (cuda/cpu)

### Section 2: Graph Transformer Layer (Cells 8-9)

| Cell | Type | Purpose |
|------|------|---------|
| 8 | Markdown | Section header |
| 9 | Code | GraphTransformerLayer class definition |

**What it does:**
- Defines the Graph Transformer layer
- Implements multi-head attention for entity graphs
- Key component that makes BERT-GT better than standard BERT

### Section 3: BERT-GT Model (Cells 10-11)

| Cell | Type | Purpose |
|------|------|---------|
| 10 | Markdown | Section header |
| 11 | Code | BERTGTModel class definition |

**What it does:**
- Defines complete BERT-GT model
- Combines BERT encoder with Graph Transformer layers
- Implements entity-aware attention mechanism

### Section 4: Data Converter (Cells 12-13)

| Cell | Type | Purpose |
|------|------|---------|
| 12 | Markdown | Section header |
| 13 | Code | BioREDToBERTGTConverter class |

**What it does:**
- Converts BioRED data format to BERT-GT format
- Handles entity extraction and relation pairing
- Creates training examples for the model

### Section 5: Dataset and DataLoader (Cells 14-15)

| Cell | Type | Purpose |
|------|------|---------|
| 14 | Markdown | Section header |
| 15 | Code | BERTGTDataset class |

**What it does:**
- Defines PyTorch Dataset for BERT-GT
- Handles tokenization and batching
- Prepares data for training

### Section 6: Training and Evaluation Functions (Cells 16-17)

| Cell | Type | Purpose |
|------|------|---------|
| 16 | Markdown | Section header |
| 17 | Code | Training functions with checkpoint support |

**What it does:**
- `get_latest_checkpoint()`: Finds latest checkpoint automatically
- `evaluate_bert_gt()`: Evaluates model with precision/recall/F1
- `train_bert_gt_with_checkpoints()`: Trains with auto-save/resume

**Key Features:**
- ✅ Auto-saves checkpoint every epoch
- ✅ Auto-resumes from latest checkpoint
- ✅ Preserves optimizer and scheduler state
- ✅ Comprehensive metrics (precision, recall, F1)

### Section 7: Load and Prepare Data (Cells 18-24)

| Cell | Type | Purpose |
|------|------|---------|
| 18 | Markdown | Section header |
| 19 | Code | **Configuration** (IMPORTANT!) |
| 20 | Code | BioREDDataLoader class |
| 21 | Code | Load data from files |
| 22 | Code | Initialize tokenizer |
| 23 | Code | Convert data to BERT-GT format |
| 24 | Code | Create PyTorch datasets |

**What it does:**
- Loads BioRED data from PubTator files
- Initializes BioBERT tokenizer
- Converts data to BERT-GT format
- Creates train/dev/test datasets

**Cell 19 (Configuration) - MUST UPDATE:**
```python
# File paths (UPDATE THESE!)
TRAIN_DATA_PATH = './data/Train.PubTator'
DEV_DATA_PATH = './data/Dev.PubTator'
TEST_DATA_PATH = './data/Test.PubTator'

# Training mode
TESTING = False  # Set to True for quick test

# Model settings
MODEL_NAME = 'dmis-lab/biobert-v1.1'
NUM_GRAPH_LAYERS = 2
NUM_EPOCHS = 30  # or 3 for testing
BATCH_SIZE = 8   # or 4 for testing
```

### Section 8: Initialize and Train Model (Cells 25-27)

| Cell | Type | Purpose |
|------|------|---------|
| 25 | Markdown | Section header |
| 26 | Code | Initialize BERT-GT model |
| 27 | Code | **Train model with checkpoint support** |

**What it does:**
- Initializes BERT-GT model with BioBERT
- Trains model with automatic checkpoint saving
- Three training modes available:
  - **AUTO-RESUME** (recommended): Auto-resumes from latest
  - **FRESH START**: Ignores checkpoints, starts from scratch
  - **SPECIFIC CHECKPOINT**: Resume from specific checkpoint

**Cell 27 (Training):**
```python
# AUTO-RESUME (RECOMMENDED)
training_stats = train_bert_gt_with_checkpoints(
    model=model,
    train_loader=train_loader,
    val_loader=dev_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=device,
    checkpoint_dir='checkpoints',
    resume='auto'  # ← Automatically finds and resumes
)
```

### Section 9: Evaluate on Test Set (Cells 28-29)

| Cell | Type | Purpose |
|------|------|---------|
| 28 | Markdown | Section header |
| 29 | Code | Load best model and evaluate |

**What it does:**
- Loads best saved model
- Evaluates on test set
- Reports comprehensive metrics

### Section 10: Visualize Training Progress (Cells 30-31)

| Cell | Type | Purpose |
|------|------|---------|
| 30 | Markdown | Section header |
| 31 | Code | Plot training curves |

**What it does:**
- Creates training loss plot
- Creates validation F1 score plot
- Saves plots as PNG files

### Section 11: Save Training Statistics (Cells 32-33)

| Cell | Type | Purpose |
|------|------|---------|
| 32 | Markdown | Section header |
| 33 | Code | Export statistics to CSV |

**What it does:**
- Converts training stats to DataFrame
- Saves to CSV file
- Displays summary statistics

### Section 12: Model Summary (Cells 34-36)

| Cell | Type | Purpose |
|------|------|---------|
| 34 | Markdown | Section header |
| 35 | Code | Calculate detailed metrics |
| 36 | Code | Display model summary and metrics |

**What it does:**
- Calculates per-class precision, recall, F1
- Creates confusion matrix
- Identifies best/worst performing classes
- Saves detailed metrics to JSON
- Displays comprehensive model summary

---

## 6. Configuration Guide

### 6.1 Cell 19: Main Configuration

This is the **most important cell** - you MUST update file paths here!

```python
# ============================================
# CONFIGURATION (Cell 19)
# ============================================

# Set this to True for testing, False for full training
TESTING = False

# File paths (UPDATE THESE!)
TRAIN_DATA_PATH = './data/Train.PubTator'  # ← Change these!
DEV_DATA_PATH = './data/Dev.PubTator'      # ← to your paths
TEST_DATA_PATH = './data/Test.PubTator'    # ← 

# Model settings
MODEL_NAME = 'dmis-lab/biobert-v1.1'  # BioBERT base model
MAX_LENGTH = 512                       # Maximum sequence length
NUM_GRAPH_LAYERS = 2                   # Number of Graph Transformer layers
NUM_ATTENTION_HEADS = 4                # Attention heads in Graph Transformer
MAX_ENTITIES = 20                      # Max entities per document
DROPOUT = 0.1                          # Dropout rate

# Training settings
if TESTING:
    print("⚠️  TESTING MODE")
    MAX_DOCS = 100        # Load 100 docs for testing
    NUM_EPOCHS = 3        # Quick training
    BATCH_SIZE = 4        # Small batch
    LEARNING_RATE = 1e-5
else:
    print("✓ FULL TRAINING MODE")
    MAX_DOCS = None       # Load all documents
    NUM_EPOCHS = 30       # Full training (as in paper)
    BATCH_SIZE = 8        # Standard batch size
    LEARNING_RATE = 1e-5

# Relation types
RELATION_TYPES = {
    'Positive_Correlation': 0,
    'Negative_Correlation': 1,
    'Association': 2,
    'No_Relation': 3
}
```

### 6.2 Configuration Options Explained

#### Testing vs Full Training

**Testing Mode (`TESTING = True`):**
- Purpose: Quick verification (20-30 minutes)
- Documents: 100
- Epochs: 3
- Batch size: 4
- Expected F1: ~60-65%
- Use when: First time running, testing changes, debugging

**Full Training Mode (`TESTING = False`):**
- Purpose: Best performance (30-40 hours)
- Documents: All (~400+)
- Epochs: 30
- Batch size: 8
- Expected F1: ~70-73%
- Use when: Final training, reproducing paper results

#### Model Architecture Settings

**NUM_GRAPH_LAYERS:**
- Default: 2 (paper setting)
- Options: 1, 2, or 3
- Impact: More layers = better performance but slower training
- Recommendation: Keep at 2

**NUM_ATTENTION_HEADS:**
- Default: 4
- Options: 4 or 8
- Impact: More heads = more computation
- Recommendation: Keep at 4

**MAX_ENTITIES:**
- Default: 20
- Options: 15-30
- Impact: Maximum entities to consider per document
- Recommendation: Keep at 20

**DROPOUT:**
- Default: 0.1
- Options: 0.1, 0.2, or 0.3
- Impact: Higher = more regularization, may prevent overfitting
- Recommendation: Start with 0.1

#### Training Settings

**NUM_EPOCHS:**
- Testing: 3
- Full: 30 (paper setting)
- Impact: More epochs = better convergence

**BATCH_SIZE:**
- Testing: 4
- Full: 8
- GPU Memory: 
  - 8GB: Use 2-4
  - 12GB: Use 4-8
  - 16GB+: Use 8-16

**LEARNING_RATE:**
- Default: 1e-5 (standard for BERT fine-tuning)
- Options: 5e-6 to 2e-5
- Recommendation: Keep at 1e-5

### 6.3 File Paths

**UPDATE THESE IN CELL 19!**

```python
# Your actual file locations
TRAIN_DATA_PATH = '/path/to/your/data/Train.PubTator'
DEV_DATA_PATH = '/path/to/your/data/Dev.PubTator'
TEST_DATA_PATH = '/path/to/your/data/Test.PubTator'
```

**Common Path Formats:**

**Linux/Mac:**
```python
TRAIN_DATA_PATH = '/home/username/data/Train.PubTator'
# or relative
TRAIN_DATA_PATH = './data/Train.PubTator'
# or absolute
TRAIN_DATA_PATH = '/mnt/project/BioRED/Train.PubTator'
```

**Windows:**
```python
TRAIN_DATA_PATH = 'C:/Users/username/data/Train.PubTator'
# or
TRAIN_DATA_PATH = 'C:\\Users\\username\\data\\Train.PubTator'
```

---

## 7. Running the Notebook

### 7.1 First Time Setup

**Step 1: Install Everything (Cells 1-7)**

Run cells 1-7 in order:
1. Cell 1: Read title
2. Cell 2: Read section header
3. Cell 3: Install PyTorch (takes ~2-3 minutes)
4. Cell 4: Install other packages (takes ~1-2 minutes)
5. Cell 5: Check GPU (should show your GPU)
6. Cell 6: Import libraries (takes ~10 seconds)
7. Cell 7: Verify setup (should show PyTorch 2.6.0+cu124)

**Step 2: Review Architecture (Cells 8-17)**

These cells define the model. You can:
- **Just run them**: If you trust the code
- **Read and understand**: If you want to learn how BERT-GT works

Run cells 8-17 in order (takes ~5 seconds total)

**Step 3: Configure (Cell 19) ⭐ CRITICAL**

```python
# UPDATE THESE FILE PATHS!
TRAIN_DATA_PATH = './data/Train.PubTator'  # ← Your path here
DEV_DATA_PATH = './data/Dev.PubTator'      # ← Your path here
TEST_DATA_PATH = './data/Test.PubTator'    # ← Your path here

# For first run, use testing mode
TESTING = True  # ← Set to True for quick test
```

Run cell 19

**Step 4: Load Data (Cells 20-24)**

Run cells 20-24 in order:
- Takes ~2-5 minutes for testing mode
- Takes ~5-10 minutes for full training mode

**Step 5: Initialize Model (Cell 26)**

Run cell 26:
- Takes ~30 seconds to download and initialize BioBERT

**Step 6: Train Model (Cell 27)**

Run cell 27:
- Testing mode: ~20-30 minutes
- Full training: ~30-40 hours (1 GPU)

### 7.2 Run All Cells at Once

**Quick Method:**
1. Update file paths in Cell 19
2. Set `TESTING = True` (for first run)
3. Click: **Cell → Run All**
4. Wait ~30 minutes (testing) or ~30-40 hours (full)

### 7.3 Resume Interrupted Training

If training was interrupted:

**Option 1: Just Rerun Cell 27 (Recommended) ⭐**
```python
# Cell 27 automatically resumes from latest checkpoint!
# Just run the cell again
```

**Option 2: Check Checkpoints First**
```python
# Run this in a new cell to see checkpoints
import os
if os.path.exists('checkpoints'):
    print("Checkpoints found:")
    for f in sorted(os.listdir('checkpoints')):
        if f.endswith('.pt'):
            print(f"  {f}")
else:
    print("No checkpoints found")
```

**Option 3: Resume from Specific Checkpoint**
```python
# In Cell 27, uncomment and modify:
training_stats = train_bert_gt_with_checkpoints(
    ...,
    resume='checkpoints/checkpoint_epoch_15.pt'  # ← Specific checkpoint
)
```

### 7.4 Training Workflow

```
1. First Run (Testing)
   ├─ Set TESTING = True
   ├─ Run All Cells
   ├─ Wait ~30 minutes
   └─ Verify F1 ~60-65%

2. Full Training
   ├─ Set TESTING = False
   ├─ Run Cell 19 (config)
   ├─ Run Cells 20-24 (data loading)
   ├─ Run Cell 26 (model init)
   ├─ Run Cell 27 (training)
   └─ Wait ~30-40 hours

3. If Interrupted
   ├─ Just rerun Cell 27
   └─ Auto-resumes from latest checkpoint

4. Evaluation
   ├─ Run Cell 29 (test evaluation)
   ├─ Run Cell 31 (visualizations)
   ├─ Run Cell 33 (save stats)
   └─ Run Cells 35-36 (detailed metrics)
```

---

## 8. Understanding the Output

### 8.1 Training Output (Cell 27)

**Initial Output:**
```
============================================================
TRAINING WITH AUTO-RESUME
============================================================

NO CHECKPOINT FOUND: Starting fresh training
✓ Starting fresh training
✓ Checkpoints will be saved to: checkpoints/
============================================================
```

Or if resuming:
```
AUTO-RESUME: Found checkpoint
✓ Loaded checkpoint: checkpoint_epoch_15.pt
✓ Resuming from epoch 16
✓ Best validation F1 so far: 0.6891
✓ Training history: 15 epochs
============================================================
```

**During Training (Each Epoch):**
```
============================================================
Epoch 1/30
============================================================
Training: 100%|██████████| 10405/10405 [15:23<00:00, 11.27it/s, loss=0.423]

Average training loss: 0.4567

Validation Metrics:
  F1 (weighted): 0.6234
  Precision (weighted): 0.6189
  Recall (weighted): 0.6289
  Accuracy: 0.6289

✓ Saved checkpoint: checkpoint_epoch_1.pt
✓ Saved best model with F1: 0.6234
```

**After Training:**
```
============================================================
TRAINING COMPLETE!
Best validation F1: 0.7123
Total epochs trained: 30
============================================================

✓ Training complete!
✓ Trained 30 epochs
```

### 8.2 Test Evaluation Output (Cell 29)

```
Loading best model...
✓ Best model loaded

Evaluating on test set...
Evaluating: 100%|██████████| 2500/2500 [02:15<00:00, 18.42it/s]

======================================================================
TEST SET RESULTS
======================================================================

📊 Test Set Performance:
  Accuracy:            0.7234 (72.34%)

  F1 Scores:
    Macro:               0.6891 (68.91%)
    Weighted:            0.7123 (71.23%)
    Micro:               0.7234 (72.34%)

  Precision:
    Macro:               0.6745 (67.45%)
    Weighted:            0.7089 (70.89%)
    Micro:               0.7234 (72.34%)

  Recall:
    Macro:               0.7012 (70.12%)
    Weighted:            0.7234 (72.34%)
    Micro:               0.7234 (72.34%)
```

### 8.3 Per-Class Metrics (Cells 35-36)

```
  Per-Class Performance:
  ----------------------------------------------------------------------
  Relation                  Precision     Recall   F1-Score  Support
  ----------------------------------------------------------------------
  Positive_Correlation         0.7523     0.7145     0.7329       84
  Negative_Correlation         0.6234     0.6891     0.6545       45
  Association                  0.7012     0.6523     0.6759       62
  No_Relation                  0.7189     0.7389     0.7288      123
  ----------------------------------------------------------------------

  📈 Best performing class:
     No_Relation
     F1: 0.7288, Precision: 0.7189, Recall: 0.7389

  📉 Worst performing class:
     Negative_Correlation
     F1: 0.6545, Precision: 0.6234, Recall: 0.6891
```

### 8.4 Confusion Matrix (Cell 36)

```
  🔢 Confusion Matrix:
  ----------------------------------------------------------------------
  Predicted →           Positive_C Negative_C Associatio No_Relatio 
  True ↓              ----------------------------------------------------
  Positive_Correlation          60          8          6         10 
  Negative_Correlation           7         31          4          3 
  Association                    5          6         40         11 
  No_Relation                   10          4         18         91 
```

### 8.5 Files Created

**During Training:**
```
checkpoints/
├── checkpoint_epoch_1.pt     # Full checkpoint (~600 MB)
├── checkpoint_epoch_2.pt
├── ...
└── checkpoint_epoch_30.pt

best_bert_gt_model.pt         # Best model weights (~420 MB)
```

**After Evaluation:**
```
bert_gt_training_stats.csv    # Training history
bert_gt_training_progress.png # Training plots
bert_gt_detailed_metrics.json # Complete metrics
```

---

## 9. Checkpoint System

### 9.1 How Checkpoints Work

The notebook uses an **automatic checkpoint system** that:
- ✅ Saves checkpoint after every epoch
- ✅ Auto-resumes from latest checkpoint
- ✅ Preserves model, optimizer, scheduler state
- ✅ Tracks training history

### 9.2 Checkpoint Files

**Checkpoint File (`checkpoint_epoch_N.pt`):**
Contains:
- Model weights
- Optimizer state (momentum, etc.)
- Scheduler state (learning rate)
- Epoch number
- Training loss
- Validation F1
- Best F1 so far
- Complete training history

**Best Model File (`best_bert_gt_model.pt`):**
Contains:
- Only model weights
- Saved whenever validation F1 improves

### 9.3 Three Resume Modes

**Mode 1: Auto-Resume (Default, Recommended) ⭐**

```python
# Cell 27 - default behavior
training_stats = train_bert_gt_with_checkpoints(
    ...,
    resume='auto'  # ← Automatically finds latest
)
```

Behavior:
- First run: Starts fresh training
- Subsequent runs: Auto-resumes from latest checkpoint
- No manual intervention needed

**Mode 2: Fresh Start**

```python
# Cell 27 - uncomment this section
training_stats = train_bert_gt_with_checkpoints(
    ...,
    resume='fresh'  # ← Ignores checkpoints
)
```

Use when:
- Want to retrain from scratch
- Testing different configurations
- Have bad checkpoints

**Mode 3: Specific Checkpoint**

```python
# Cell 27 - uncomment and modify
training_stats = train_bert_gt_with_checkpoints(
    ...,
    resume='checkpoints/checkpoint_epoch_15.pt'
)
```

Use when:
- Want to resume from specific epoch
- Testing checkpoint loading
- Comparing different training stages

### 9.4 Checkpoint Management

**Check Available Checkpoints:**
```python
# Run in new cell
import os
if os.path.exists('checkpoints'):
    checkpoints = sorted(os.listdir('checkpoints'))
    print(f"Found {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        size_mb = os.path.getsize(f'checkpoints/{ckpt}') / (1024**2)
        print(f"  {ckpt}: {size_mb:.1f} MB")
```

**Delete Old Checkpoints (Save Disk Space):**
```python
# Keep only latest 5 checkpoints
import glob
import os

checkpoints = sorted(glob.glob('checkpoints/checkpoint_epoch_*.pt'))
if len(checkpoints) > 5:
    for ckpt in checkpoints[:-5]:  # Delete all but latest 5
        os.remove(ckpt)
        print(f"Deleted: {ckpt}")
```

**Load Checkpoint Manually:**
```python
# Inspect checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation F1: {checkpoint['val_f1']:.4f}")
print(f"Training loss: {checkpoint['train_loss']:.4f}")
print(f"Best F1 so far: {checkpoint['best_val_f1']:.4f}")
```

### 9.5 Checkpoint Best Practices

1. **Always use `resume='auto'`** (default in Cell 27)
2. **Don't delete checkpoints during training**
3. **Keep latest 5-10 checkpoints** (delete older ones)
4. **Backup best model** (`best_bert_gt_model.pt`)
5. **Check disk space** (each checkpoint ~600 MB)

---

## 10. Troubleshooting

### 10.1 Installation Issues

**Error: "Could not find a version that satisfies the requirement torch==2.6.0"**

**Solution:**
```bash
# Verify Python version (need 3.8+)
python --version

# Try explicit CUDA version
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Or CPU version if no GPU
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Error: "No module named 'transformers'"**

**Solution:**
```bash
pip install transformers scikit-learn tqdm matplotlib
```

### 10.2 GPU Issues

**Error: "CUDA out of memory"**

**Solution 1: Reduce batch size**
```python
# In Cell 19
BATCH_SIZE = 2  # or 4 instead of 8
```

**Solution 2: Reduce max entities**
```python
# In Cell 19
MAX_ENTITIES = 15  # instead of 20
```

**Solution 3: Use gradient accumulation**
```python
# Modify training loop (advanced)
GRADIENT_ACCUMULATION_STEPS = 2
```

**Error: "CUDA not available" (Cell 7 shows False)**

**Check 1: CUDA installation**
```bash
nvidia-smi
# Should show GPU
```

**Check 2: PyTorch CUDA version**
```python
import torch
print(torch.version.cuda)
# Should show 12.4
```

**Check 3: Reinstall PyTorch**
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 10.3 Data Loading Issues

**Error: "FileNotFoundError: [Errno 2] No such file or directory: './data/Train.PubTator'"**

**Solution:**
```python
# In Cell 19, update to actual file paths
TRAIN_DATA_PATH = '/absolute/path/to/Train.PubTator'

# Or check current directory
import os
print(os.getcwd())
print(os.listdir('.'))
```

**Error: "No RE examples created during conversion"**

**Cause:** Not enough documents loaded (need minimum 100)

**Solution:**
```python
# In Cell 19, increase MAX_DOCS
MAX_DOCS = 100  # minimum
# or
MAX_DOCS = None  # load all
```

### 10.4 Training Issues

**Error: "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0"**

**Cause:** Old version of notebook without device fixes

**Solution:** 
- Make sure using latest notebook version (v1.2)
- Cell 17 should have device fixes for entity positions

**Error: Training very slow**

**Check 1: GPU utilization**
```bash
# In terminal
watch -n 1 nvidia-smi
# GPU utilization should be >90%
```

**Check 2: Batch size**
```python
# Increase if GPU not fully utilized
BATCH_SIZE = 16  # if you have 24GB+ VRAM
```

**Check 3: DataLoader workers**
```python
# In dataset creation code (Cell 24)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4  # ← Add this if not present
)
```

**Training stopped/hanged**

**Solution 1: Just rerun Cell 27**
- Automatically resumes from latest checkpoint

**Solution 2: Check checkpoint**
```python
import os
print(os.listdir('checkpoints'))
# Should show checkpoint files
```

### 10.5 Memory Issues

**Error: "RuntimeError: CUDA out of memory"**

**Immediate fixes:**
```python
# 1. Reduce batch size
BATCH_SIZE = 2

# 2. Reduce max sequence length
MAX_LENGTH = 256

# 3. Reduce max entities
MAX_ENTITIES = 10

# 4. Use fewer graph layers
NUM_GRAPH_LAYERS = 1
```

**Advanced fix: Gradient checkpointing**
```python
# In Cell 26, after model initialization
model.bert.gradient_checkpointing_enable()
```

### 10.6 Results Issues

**F1 score too low (<50%)**

**Possible causes:**
1. Not enough training epochs
2. Too small dataset
3. Learning rate too high/low
4. Model not converging

**Solutions:**
```python
# 1. Train longer
NUM_EPOCHS = 30  # instead of 3

# 2. Use full dataset
MAX_DOCS = None  # instead of 100

# 3. Adjust learning rate
LEARNING_RATE = 5e-6  # try different values

# 4. Check training curves (Cell 31)
# Loss should decrease, F1 should increase
```

**Model not improving**

**Check:**
```python
# Plot training stats (Cell 31)
# Look for:
# - Decreasing training loss
# - Increasing validation F1
# - No overfitting (train F1 >> val F1)
```

---

## 11. Advanced Usage

### 11.1 Custom Configuration

**Experiment with different settings:**

```python
# Cell 19 - try different configurations

# Configuration 1: More layers
NUM_GRAPH_LAYERS = 3
NUM_ATTENTION_HEADS = 8

# Configuration 2: Stronger regularization
DROPOUT = 0.3

# Configuration 3: Different learning rate
LEARNING_RATE = 2e-5

# Configuration 4: Larger batch with gradient accumulation
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
```

### 11.2 Different Base Models

**Try different BERT variants:**

```python
# Cell 19 - change MODEL_NAME

# Option 1: PubMedBERT
MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

# Option 2: SciBERT
MODEL_NAME = 'allenai/scibert_scivocab_uncased'

# Option 3: BioBERT v1.2
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.2'

# Option 4: Standard BERT (baseline)
MODEL_NAME = 'bert-base-uncased'
```

### 11.3 Multiple GPU Training

**Use DataParallel:**

```python
# Cell 26 - after model initialization
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

**Use DistributedDataParallel (better):**

```python
# Requires more setup - see PyTorch DDP documentation
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

### 11.4 Custom Data

**Use your own dataset:**

1. Format data like BioRED PubTator format
2. Update paths in Cell 19
3. Run notebook

**PubTator format:**
```
12345678|t|Document title
12345678|a|Document abstract text
12345678	0	5	Gene	Gene	D001
12345678	10	15	Disease	Disease	D002
12345678	CID	D001	D002
```

### 11.5 Export for Production

**Save model for deployment:**

```python
# After training (add to Cell 29)
# Save model architecture + weights
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'num_labels': 4,
        'num_graph_layers': NUM_GRAPH_LAYERS,
        'num_attention_heads': NUM_ATTENTION_HEADS,
        'dropout': DROPOUT
    }
}, 'bert_gt_production.pt')

# Save tokenizer
tokenizer.save_pretrained('./bert_gt_tokenizer/')
```

**Load for inference:**

```python
# In production code
config = torch.load('bert_gt_production.pt')['config']
model = BERTGTModel(
    model_name=MODEL_NAME,
    num_labels=config['num_labels'],
    num_graph_layers=config['num_graph_layers'],
    num_attention_heads=config['num_attention_heads'],
    dropout=config['dropout']
)
model.load_state_dict(config['model_state_dict'])
model.eval()
```

---

## 12. FAQ

### Q1: How long does training take?

**Testing Mode (100 docs, 3 epochs):**
- RTX 3080 (12GB): ~20-30 minutes
- RTX 3090 (24GB): ~15-20 minutes
- A100 (40GB+): ~10-15 minutes

**Full Training (400+ docs, 30 epochs):**
- RTX 3080 (12GB): **~30-40 hours**
- RTX 3090 (24GB): ~20-25 hours
- A100 (40GB+): ~10-15 hours

### Q2: How much GPU memory do I need?

| Batch Size | GPU Memory | Example GPUs |
|------------|------------|--------------|
| 2-4 | 8 GB | GTX 1080 Ti |
| 4-8 | 12 GB | RTX 3080 |
| 8-16 | 16-24 GB | RTX 3090, RTX 4090 |
| 16-32 | 40+ GB | A100, H100 |

**Recommended:** 12GB+ (RTX 3080 or better)

### Q3: Can I train on CPU?

**Yes, but not recommended:**
- Very slow (~100+ hours)
- Set `BATCH_SIZE = 2`
- Consider cloud GPU instead (Google Colab, AWS, etc.)

### Q4: How do I resume training?

**Just rerun Cell 27!**

The notebook automatically:
1. Finds latest checkpoint
2. Loads model, optimizer, scheduler
3. Resumes from next epoch

No manual steps needed if using `resume='auto'` (default)

### Q5: What if I change configuration mid-training?

**Safe to change:**
- File paths (if re-running data cells)
- Visualization settings

**NOT safe to change (will cause errors):**
- NUM_GRAPH_LAYERS
- NUM_ATTENTION_HEADS
- MODEL_NAME
- NUM_LABELS

**If you must change architecture:**
```python
# Use fresh start
training_stats = train_bert_gt_with_checkpoints(
    ...,
    resume='fresh'  # ← Ignores checkpoints
)
```

### Q6: Can I use my own data?

Yes! Format your data in BioRED PubTator format:
- Document title and abstract
- Entity annotations
- Relation annotations

See Section 11.4 for details.

### Q7: How do I know training is working?

**Good signs:**
- Training loss decreases steadily
- Validation F1 increases
- No "CUDA out of memory" errors
- GPU utilization >90%

**Check with:**
- Cell 31: Training plots
- Cell 33: Training statistics
- `nvidia-smi`: GPU usage

### Q8: What's the difference between testing and full training?

| Aspect | Testing | Full Training |
|--------|---------|---------------|
| **Purpose** | Quick verification | Best performance |
| **Documents** | 100 | 400+ (all) |
| **Epochs** | 3 | 30 |
| **Time** | 20-30 min | 30-40 hours |
| **F1 Score** | ~60-65% | ~70-73% |
| **When to use** | First run, debugging | Final training, paper reproduction |

### Q9: How do checkpoints work?

**Automatic process:**
1. After each epoch, saves complete state to `checkpoints/`
2. If training interrupted, just rerun Cell 27
3. Automatically finds and loads latest checkpoint
4. Resumes from next epoch

**Files created:**
- `checkpoint_epoch_N.pt`: Full checkpoint (~600 MB each)
- `best_bert_gt_model.pt`: Best model only (~420 MB)

### Q10: Can I compare with BioBERT baseline?

Yes! The baseline notebook is included:
- `biored_relation_extraction.ipynb` (v1)
- `biored_relation_extraction_v2.ipynb` (v2)

Expected performance:
- BioBERT: ~65% F1
- BERT-GT: ~73% F1
- Improvement: +8% F1

### Q11: What if I get "No RE examples created"?

**Cause:** Not enough documents loaded

**Solution:**
```python
# In Cell 19
MAX_DOCS = 100  # minimum
# or
MAX_DOCS = None  # load all documents
```

Must have at least 100 documents to create enough relation examples.

### Q12: How do I save/share my trained model?

**Option 1: Share checkpoint (recommended)**
```bash
# Zip the checkpoint directory
tar -czf bert_gt_checkpoints.tar.gz checkpoints/
# Share: bert_gt_checkpoints.tar.gz (~18 GB for 30 epochs)
```

**Option 2: Share best model only**
```bash
# Share just the best model
# Share: best_bert_gt_model.pt (~420 MB)
```

**Option 3: Export for production**
```python
# See Section 11.5 for full export code
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {...}
}, 'bert_gt_production.pt')
```

---


### References

**Claude.Ai Chatlog:**
- https://claude.ai/share/0111ec41-3c6b-4ea2-8928-e80da01faf9d

**BERT-GT Paper:**
- Lai et al. (2021) - "Biomedical Relation Extraction with Graph Transformer"

**BioRED Dataset:**
- Luo et al. (2022) - "BioRED: A Rich Biomedical Relation Extraction Dataset"

**BioBERT:**
- Lee et al. (2020) - "BioBERT: a pre-trained biomedical language representation model"

### Community

**GitHub Issues:**
- Original BERT-GT: github.com/ncbi/BioRED

**Hugging Face:**
- BioBERT model: dmis-lab/biobert-v1.1
- 

---

## 🎉 Conclusion

You now have everything needed to:
- ✅ Install and set up BERT-GT
- ✅ Train on BioRED dataset
- ✅ Use checkpoint system
- ✅ Evaluate and visualize results
- ✅ Troubleshoot common issues
- ✅ Customize for your needs

**Quick Recap:**

1. **Install** PyTorch 2.6.0 + CUDA 12.4 (Cells 3-4)
2. **Configure** file paths (Cell 19)
3. **Test** with `TESTING = True` (20-30 min)
4. **Train** with `TESTING = False` (30-40 hours)
5. **Evaluate** and get ~70-73% F1 (Cells 29, 35-36)

**Good luck with your biomedical relation extraction! 🚀**

---

**Manual Version:** 1.2 FINAL  
**Last Updated:** December 2024  
**Based on:** BERT_GT_Notebook.ipynb (36 cells)  
**Status:** Production Ready ✓
