# Child Stunting Classification using Machine Learning

## Author

**Kumi Yunis Konda**  
Course: Introduction to Machine Learning  
Facilitator: Dirac Murairi  
Date: February 21, 2026

## Overview

This project uses machine learning and deep learning to predict child stunting. I compare 7 different models to find the best one for identifying children at risk of malnutrition.

I did this project for my Introduction to Machine Learning course. The goal is to help medical staff in clinics (especially in refugee camps like Kakuma) quickly identify children who need help.

---

## My Mission

I want to create tools that help medical workers in under-staffed clinics. In Kakuma refugee camp, there are over 300,000 people but only 6 clinics. When clinics are overwhelmed, children with malnutrition can be missed. Machine learning can help screen children faster and flag those at risk.

---

## Dataset

**Dataset Name:** Stunting and Wasting Dataset (Synthetic)

**Source:** [Kaggle](https://www.kaggle.com/datasets/jabirmuktabir/stunting-wasting-dataset)

**What is this data?**
- 100,000 records of Indonesian children
- The data is SYNTHETIC (created by computer, not from real hospitals)
- It follows WHO growth standards to look realistic

**Columns in the dataset:**
| Column Name | Meaning | Example |
|-------------|---------|---------|
| Jenis Kelamin | Gender (Laki-laki = Male, Perempuan = Female) | Laki-laki |
| Umur (bulan) | Age in months | 24 months |
| Tinggi Badan (cm) | Height in centimeters | 85 cm |
| Berat Badan (kg) | Weight in kilograms | 12 kg |
| Stunting | Target class for my project | Normal, Stunted, Severely Stunted, Tall |
| Wasting | Another malnutrition measure (I didn't use this) | - |

**Class Distribution (Stunting):**
| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 72,312 | 72.3% |
| Stunted | 16,160 | 16.2% |
| Severely Stunted | 5,819 | 5.8% |
| Tall | 5,709 | 5.7% |

**The challenge:** The data is imbalanced. The "Severely Stunted" and "Tall" classes have very few samples. Good models must handle this well.

---

## What is Stunting?

Stunting means a child is **too short for their age**. It happens from long-term malnutrition. Unlike wasting (being too thin), stunting is permanent and affects brain development. That's why prevention is important - once a child is stunted, it cannot be fixed.

---

## What I Did (7 Experiments)

I ran 7 experiments to compare different approaches:

### Experiment 1: Logistic Regression
- **Type:** Traditional Machine Learning (baseline)
- **Goal:** Simple model to see how well basic approach works
- **Accuracy:** 80.93%
- **Problem:** Only caught 10% of severely stunted children (missed 90 out of 100!)

### Experiment 2: Random Forest
- **Type:** Traditional Machine Learning
- **Goal:** See if ensemble method performs better
- **Accuracy:** 100% (SUSPICIOUS!)
- **Note:** Perfect accuracy on health data is impossible. This suggests either data leakage or the synthetic data is too perfect. I don't trust this result.

### Experiment 3: Sequential Neural Network
- **Type:** Deep Learning (Sequential API)
- **Architecture:** 64 neurons → Dropout → 32 neurons → Output (4 classes)
- **Accuracy:** 95.97%
- **Improvement:** Severely stunted recall improved to 92%

### Experiment 4: Functional API Neural Network
- **Type:** Deep Learning (Functional API)
- **Goal:** Try more flexible architecture
- **Accuracy:** 96.03%
- **Note:** Similar performance to Experiment 3

### Experiment 5: tf.data API Neural Network
- **Type:** Deep Learning with TensorFlow data pipeline
- **Goal:** Test if data pipeline improves performance
- **Accuracy:** 96.73%
- **Benefit:** Training was faster due to prefetching

### Experiment 6: Neural Network with Callbacks (BEST MODEL) 
- **Type:** Deep Learning with training optimizations
- **Callbacks used:**
  - **EarlyStopping:** Stops training when no improvement (patience=5)
  - **ReduceLROnPlateau:** Cuts learning rate in half when progress stalls
- **Accuracy:** 98.96%
- **Severely Stunted Recall:** 99% (huge improvement from 10%!)
- **Why it's best:** Excellent on ALL classes, not just majority

### Experiment 7: Neural Network with Adamax + Batch Size 64
- **Type:** Deep Learning with different optimizer
- **Changes:** Used Adamax instead of Adam, batch size 32 → 64
- **Accuracy:** 97.91%
- **Note:** Very good, but Experiment 6 still better

---

## Results Summary

| Experiment | Model | Accuracy | Best Feature |
|------------|-------|----------|--------------|
| Exp 1 | Logistic Regression | 80.93% | Baseline |
| Exp 2 | Random Forest | 100% (unreliable) | - |
| Exp 3 | Sequential NN | 95.97% | First deep learning model |
| Exp 4 | Functional NN | 96.03% | Flexible architecture |
| Exp 5 | tf.data NN | 96.73% | Faster training |
| **Exp 6** | **NN + Callbacks** | **98.96%** | **Best overall** |
| Exp 7 | Adamax + Batch 64 | 97.91% | Good alternative |

### Per-Class Performance (Best Model - Exp 6)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.99 | 1.00 | 0.99 |
| Severely Stunted | 0.97 | 0.99 | 0.98 |
| Stunted | 0.98 | 0.98 | 0.98 |
| Tall | 1.00 | 0.93 | 0.97 |

**Key Achievement:** Severely stunted children went from 10% recall (Exp 1) to 99% recall (Exp 6). This means the model now catches nearly all at-risk children.

---

## Visualizations Included

In the notebook, you will find:
- Class distribution bar charts
- Learning curves (training vs validation)
- Confusion matrices for all models
- ROC curves with AUC scores
- Model comparison charts

---

## Files in This Repository

| File Name | What It Is |
|-----------|------------|
| `Child_Stunting_Classification.ipynb` | Main notebook with all code and outputs |
| `README.md` | This file - project explanation |
| `requirements.txt` | List of Python packages needed |
| `.gitignore` | Files to ignore when uploading to GitHub |
| `results_summary.csv` | Experiment results in table format |

---

## How to Run This Project

### Option 1: Run in Google Colab (Easiest)

1. Click this button: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yunis-Konda001/child-stunting-classification/blob/main/Child_Stunting_Classification.ipynb)
   

2. Once opened, click **Runtime → Run all**

### Option 2: Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/Yunis-Konda001/child-stunting-classification.git
