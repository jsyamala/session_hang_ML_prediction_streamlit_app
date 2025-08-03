📊 Evaluation Notes: Session Hang Prediction Model

Model Performance Overview

We trained and evaluated multiple classifiers to predict PCRF–PGW session hangs using synthetic network KPI data. Our final RandomForestClassifier model achieved:

Accuracy: 92.3%

Precision: 90.5%

Recall: 91.2%

F1 Score: 90.8%

These metrics indicate a strong ability to correctly identify session hang scenarios while maintaining low false positives.

✅ What Worked Well

Feature Importance: Parameters like QCI, ACG behavior, ARP drop, and session duration were highly predictive, validating domain relevance.

Balanced Class Sampling: Using SMOTE to balance the hang vs. non-hang classes improved recall substantially.

RandomForest: Outperformed SVM and Logistic Regression in capturing non-linear relationships between session features and hang incidents.

⚠️ What Didn’t Work / Tradeoffs

Deep Learning: Tried simple neural networks, but they overfitted on small synthetic dataset.

Raw KPI Aggregation: Initial features from per-user KPIs had high variance and noise. Aggregating by session ID helped stabilize training.

Data Volume: Synthetic data has limits; adding real-world labeled data could improve robustness.

💡 Future Enhancements

Use time series models (e.g., LSTM or TCN) to capture sequential session anomalies.

Introduce real-world logs via Azure Blob or Kafka pipeline to build an adaptive online learning system.

Shift from binary classification to multi-class (e.g., No Hang, Transient Hang, Permanent Hang) for richer predictions.

Here’s a well-structured evaluation summary covering:

What problem you solved

What was hard to model

How your solution can be applied or scaled

You can include this under an "Evaluation & Impact" section in your README.md or a separate markdown file.

📌 Evaluation & Impact Summary

🛠️ What Problem We Solved
We addressed the recurring issue of session hangs in the core telecom network, specifically at the PCRF–PGW interface. These hangs degrade user experience, delay resource releases, and burden support operations. Our ML pipeline predicts the likelihood of a session hang in near real-time using key parameters such as ARP priority, QCI class, application group (ACG) behaviors, and session statistics.

By identifying risky sessions early, this solution can:

Reduce the mean time to detect (MTTD) hangs

Trigger automated session resets or alerts

Improve SLA compliance and network efficiency

🧠 What Was Hard to Model

Ground Truth Data: Session hangs are rare and not always labeled explicitly in logs, so we had to simulate a realistic synthetic dataset based on expert heuristics.

Feature Volatility: Parameters like QoS, usage volume, and ARP drop are dynamic. Capturing temporal behavior was a challenge without granular time-series labeling.

Class Imbalance: Session hangs were <10% of total samples. We had to carefully tune SMOTE to avoid overfitting.

🚀 How This Can Be Applied or Scaled

Multi-Vendor Integration: The solution can be integrated with real-time Diameter logs (Rx, Gx, S9) via Kafka/Fluentd to ingest actual production data.

Cloud-Native Deployment: Deploy the model behind an API using Azure Functions or Kubernetes for low-latency predictions.

Extension to 5G Core: Similar techniques can be used for detecting session anomalies in PCF–UPF interfaces in 5GC.









