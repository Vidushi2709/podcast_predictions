# 🎧 Podcast Listening Time Prediction

This project aims to predict the **listening time** of podcast episodes using various machine learning techniques. The evaluation metric is **Root Mean Squared Error (RMSE)**.

---

## 📈 Problem Statement

Given a dataset of podcast episodes with metadata and engagement stats, your goal is to predict how long a user listens to an episode.  
**Metric**:  
\[
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 }
\]

---

## 🧰 Tech Stack & Tools

- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- LightGBM
- CatBoost
- Optuna for hyperparameter optimization
- KFold Cross-Validation

---

## ⚙️ Approach

### 🔍 Data Exploration
- Visualized distributions, correlations, and missing values.
- Analyzed feature importance and outliers.

### 🧹 Preprocessing
- **Categorical Columns** → Label Encoding
- **`podcast_name` Column** → TF-IDF Vectorization
- **Numerical Columns** → Normalized using StandardScaler

### 🔄 Modeling
- Used K-Fold Cross-Validation (k=5)
- Trained with:
  - LightGBM
  - CatBoost

### 🧪 Optimization
- Tuned hyperparameters using **Optuna**

---

## 🏁 Results

- 📉 **Best RMSE**: `12.09`
- ✅ Best performing model: **LightGBM + TF-IDF on podcast_name + normalized numerics**

---
