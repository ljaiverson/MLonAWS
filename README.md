# Machine Learning at Scale on AWS

A distributed machine learning project using Apache Spark on AWS to predict song popularity from the Million Song Dataset.

## Overview

This project uses Apache Spark on AWS EMR to build binary classification models that predict whether a song will be popular based on audio features and metadata. A song is classified as "popular" if its hotness score exceeds the dataset mean.

**Dataset**: Million Song Dataset (~580K songs initially, 117K after cleaning)
**Task**: Binary classification (popular vs. not popular)
**Platform**: Apache Spark on AWS EMR

## Features

**Numeric Features (16)**: artist familiarity/hotness, geographic coordinates, duration, tempo, loudness, key, mode, time signature, year

**Text Features**: song title (TF-IDF), artist genre tags (Bag-of-Words)

**Final Feature Vector**: 31 dimensions (16 numeric + 5 TF-IDF + 10 BoW)

## Pipeline

1. **Data Loading**: Load processed CSV files from S3 with defined schema
2. **EDA**: Histogram and scatter plot analysis to understand feature distributions
3. **Data Cleaning**:
   - Drop all-zero features (danceability, energy)
   - Filter invalid years (≤1920)
   - Remove records with missing location data
   - Result: 20% retention rate
4. **Feature Engineering**:
   - StandardScaler normalization
   - TF-IDF on song titles
   - Bag-of-Words on artist terms
5. **Modeling**: Train/test split (80/20), evaluate with AUC metric
6. **Hyperparameter Tuning**: Grid search on Random Forest parameters

## Results

| Model | Features | Train AUC | Test AUC |
|-------|----------|-----------|----------|
| Logistic Regression | Baseline | 0.764 | 0.762 |
| Random Forest | Baseline | 0.770 | 0.769 |
| Logistic Regression | +TF-IDF/BoW | 0.777 | 0.773 |
| Random Forest | +TF-IDF/BoW | 0.777 | 0.771 |

Text features improved model performance by ~1-2% AUC.

## Setup

1. Launch AWS EMR cluster with Spark
2. Install dependencies on master node:
   ```bash
   pip install matplotlib pandas numpy
   ```
3. Configure S3 bucket name in notebook
4. Run `MLonAWS.ipynb` with PySpark kernel

## Key Findings

- Text features (titles, artist tags) provide meaningful predictive signal
- Geographic location correlates with song popularity
- Aggressive data cleaning improved model quality despite reduced dataset size
- Random Forest slightly outperformed Logistic Regression
- Dataset is relatively balanced (55.4% positive class)
