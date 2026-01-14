# 🎵 Music Clustering & Playlist Recommendation System
## 📌 Overview

With millions of tracks available on music streaming platforms, manual song categorization is no longer scalable.
This project applies unsupervised machine learning techniques to automatically group songs based on their audio features such as tempo, energy, danceability, and valence.

The resulting clusters enable efficient playlist curation, music discovery, and personalized recommendations.

## 🎯 Objective

Automatically cluster songs based on audio characteristics

Identify meaningful patterns in music data

Support playlist generation and recommendation use cases

## 🧠 Skills Gained

Data exploration, preprocessing, and cleaning

Feature engineering and normalization

K-Means clustering and optimal cluster selection

Cluster evaluation using Silhouette Score and Davies–Bouldin Index

Dimensionality reduction with PCA

Cluster visualization and interpretation

Python-based machine learning workflows

Interactive dashboard development using Streamlit

## 🛠️ Tech Stack
### Programming Language

Python 3.10+

### Libraries & Tools

pandas

numpy

matplotlib

seaborn

scikit-learn

streamlit

joblib

### Development Environment

Visual Studio Code

### Visualization

Streamlit Dashboard

## 📂 Project Workflow
1️⃣ Data Loading & Exploration

Imported the Amazon Music dataset

Analyzed audio attributes such as energy, tempo, and valence using pandas and visualizations

2️⃣ Data Cleaning & Feature Engineering

Removed non-essential columns

Dataset contained no missing values

3️⃣ Feature Scaling

Standardized numerical features using StandardScaler to ensure fair clustering

4️⃣ Dimensionality Reduction

Applied PCA to reduce feature dimensions for visualization and pattern discovery

5️⃣ Clustering

Implemented K-Means clustering to group similar songs

6️⃣ Model Evaluation

Evaluated clustering performance using:

Silhouette Score

Davies–Bouldin Index

7️⃣ Visualization & Interpretation

PCA scatter plots for cluster separation

Heatmaps for feature distribution across clusters

8️⃣ Dashboard Development

Built an interactive Streamlit application to:

Explore clusters dynamically

Compare feature distributions

View top 10 songs per cluster

## 📊 Key Insights

Each cluster represents a distinct musical profile aligned with specific moods or listening contexts

Feature averages help clearly differentiate clusters

Interactive visualization improves interpretability and usability
