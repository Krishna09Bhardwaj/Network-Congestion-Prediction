# Network Congestion Analysis Project

## Overview
This project provides comprehensive data analysis and machine learning modeling for network congestion prediction using a variety of performance metrics. The system analyzes patterns in network traffic to identify factors contributing to congestion and develops models to predict congestion levels.

## Author
**Krishna Bhardwaj**

## Dataset
The Network Congestion Dataset contains detailed metrics collected from network nodes across various regions. The dataset includes:

- **Packet Loss Rate**: The percentage of packets that fail to reach their destination
- **Average Latency**: Time delay in milliseconds for packet transmission
- **Node Betweenness Centrality**: A measure of a node's importance in network topology
- **Traffic Volume**: Data throughput in MBps
- **Link Stability Score**: Reliability metric for network connections
- **Regional Information**: Geographic classification of network nodes
- **Administrative Contact**: Management information
- **Temporal Data**: Timestamps for trend analysis

### Citation
K. Bhardwaj, "Network Congestion Prediction," Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/krishna09bhardwaj/network-congestion-prediction

## Project Components

### Data Preprocessing
- Outlier detection and handling using IQR method
- Feature engineering including temporal feature extraction
- Target variable creation based on packet loss and latency metrics
- Categorical data encoding

### Exploratory Data Analysis
- Distribution analysis of key network metrics
- Correlation analysis between performance indicators
- Temporal pattern identification (hourly and daily trends)
- Regional congestion analysis
- Connection pair performance evaluation

### Machine Learning Models
The project implements multiple classification algorithms to predict network congestion levels:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

### Model Evaluation
Models are evaluated using comprehensive metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### Feature Importance Analysis
- Identifies key factors contributing to network congestion
- Permutation importance analysis
- Visual representation of feature significance

# Output
## Snapshots of Jupyter notebook 
<img width="1725" alt="SCR-20250429-leed" src="https://github.com/user-attachments/assets/01d43305-a8f8-4f83-82d0-95ed1096c1eb" />
<img width="1227" alt="SCR-20250429-ldyy" src="https://github.com/user-attachments/assets/1b0f7934-5419-4879-9120-5ff3ff7f5e0d" />



## Technical Implementation
- Language: Python 3.11
- Key Libraries:
  - Pandas & NumPy for data manipulation
  - Scikit-learn for machine learning algorithms
  - XGBoost for gradient boosting implementations
  - Matplotlib, Seaborn & Plotly for data visualization
  - Streamlit for interactive web application

## Application Usage
The Streamlit web application allows users to:
1. Explore the dataset with interactive visualizations
2. View model performance comparisons
3. Analyze feature importance
4. Understand temporal and regional congestion patterns
5. Identify high-congestion network connections

## Business Value
This analysis provides network administrators with:
- Identification of congestion patterns and bottlenecks
- Predictive capabilities for proactive network management
- Insights into key factors affecting network performance
- Data-driven decision support for infrastructure planning

## Future Enhancements
- Real-time data integration
- Network topology visualization
- Advanced time-series forecasting
- Anomaly detection for network incidents
- Capacity planning recommendations

---

© 2025 Krishna Bhardwaj. All Rights Reserved.
