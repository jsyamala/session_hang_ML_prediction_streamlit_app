import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.title("Session Hang Prediction in PCRF–PGW (N28)")

# Generate synthetic session data
def generate_data(n=2000):
    df = pd.DataFrame({
        'ccr_i': np.random.uniform(0, 1, n),
        'first_ul': np.random.uniform(0.1, 2, n),
        'first_dl': np.random.uniform(0.2, 5, n),
        'ccr_u': np.random.uniform(1, 3, n),
        'ccr_t': np.random.uniform(3, 6, n),
        'arp': np.random.choice([1, 2, 3, 4], n)
    })
    df['hang'] = (df['first_dl'] > 4).astype(int)
    return df

data = generate_data()

st.subheader("Synthetic Data Sample")
st.write(data.head())

# Train Random Forest Model
features = ['ccr_i', 'first_ul', 'first_dl', 'ccr_u', 'ccr_t', 'arp']
X = data[features]
y = data['hang']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.subheader("Random Forest Results")
st.text("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Train a simple LSTM on reshaped data
# X_seq = X.values.reshape(-1, 6, 1)
# model = Sequential([LSTM(32, input_shape=(6, 1)), Dense(1, activation='sigmoid')])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_seq, y, epochs=5, batch_size=32, verbose=0)

# loss, acc = model.evaluate(X_seq, y, verbose=0)
# st.subheader("LSTM Results")
# st.write(f"Accuracy: {acc:.2f}")

# KDE Plot of DL Latency
st.subheader("KDE Plot: First DL Packet Delay")
fig, ax = plt.subplots()
data[data['hang'] == 1]['first_dl'].plot.kde(label='Hangs', ax=ax)
data[data['hang'] == 0]['first_dl'].plot.kde(label='No Hang', ax=ax)
ax.set_xlabel("First DL Delay (s)")
ax.set_title("Distribution of First DL Delays")
ax.legend()
st.pyplot(fig)
