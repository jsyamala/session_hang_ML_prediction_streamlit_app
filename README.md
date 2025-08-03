# session_hang_ML_prediction_streamlit_app
Forecast session hangs in 5G core networks (PCRF–PGW) using Python and ML—includes synthetic data, feature engineering, Random Forest &amp; LSTM models, and Streamlit dashboard.
# Network Session Hang Predictor

A Python project that combines custom data structures and machine learning to predict network session hangs using real-world telco-like data. Built with Streamlit for interactive exploration.

## 🚀 Project Overview

This project explores how engineered features derived from network sessions can help predict session-level failures like unexpected hangups. We:

- Simulate a telco-like session log dataset
- Extract advanced features (duration stats, error codes, jitter)
- Apply and compare multiple ML models (Logistic Regression, Random Forest, DNN)
- Present a Streamlit dashboard for real-time prediction and feature insights

## 📁 Structure

- data/: synthetic dataset CSVs
- models/: saved model files (joblib or HDF5)
- notebooks/: EDA and training notebooks
- app.py: Streamlit app
- train_model.py: model training pipeline

## 🔍 Demo Screenshots

<img width="1226" height="887" alt="image" src="https://github.com/user-attachments/assets/23f9961f-3356-4531-9aec-5da5a0e004e4" />

<img width="1141" height="697" alt="image" src="https://github.com/user-attachments/assets/2ae568d4-b996-4b3f-9ea7-4958a3466748" />



## ⚙️ How to Run

1. Clone the repo  
```bash
git clone https://github.com/jsyamala/net-session-hang-predict.git
cd net-session-hang-predict
```

2. Create environment and install dependencies  
```bash
conda create -n netpredict-env python=3.9
conda activate netpredict-env
pip install -r requirements.txt
```

3. Train model (optional)  
```bash
python train_model.py
```

4. Run Streamlit app  
```bash
streamlit run app.py
```

## ✅ Results Summary

- Random Forest achieved 93.2% accuracy on test set with high recall on hang sessions.
- Deep Neural Network (2 hidden layers) matched RF but required more tuning.
- Session duration variability, error frequency, and upstream jitter emerged as top predictive features.

## 📦 Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn, tensorflow (if using DNN)
- streamlit

## 🧠 Contributions

Built with 💻 and ☕ by [jsyamala](https://github.com/jsyamala)

Pull requests welcome!

## 🔗 GitHub Repo

https://github.com/jsyamala/session_hang_ML_prediction_streamlit_app
