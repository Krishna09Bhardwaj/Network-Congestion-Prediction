"""
Network Congestion Analysis
---------------------------
A Python script for network congestion analysis using machine learning
with comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st

# For preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

# For evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For feature importance
from sklearn.inspection import permutation_importance

# Set random seed for reproducibility
np.random.seed(42)

# Set up Streamlit page
st.set_page_config(page_title="Network Congestion Analysis", 
                   page_icon="🌐", 
                   layout="wide")

st.title("Network Congestion Analysis with Machine Learning")
st.markdown("""
This application analyzes network congestion data using various machine learning techniques.
The workflow includes:
- Data Loading and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Model Training (Multiple Algorithms)
- Model Evaluation
- Results Visualization
- Feature Importance Analysis
""")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('attached_assets/Network Congestion Dataset.csv')
    return data

# Display loading spinner while loading data
with st.spinner('Loading data...'):
    data = load_data()

# Data Overview Section
st.header("1. Data Overview")
st.write(f"Dataset Shape: {data.shape}")
st.write("First 5 rows:")
st.dataframe(data.head())

# Display basic info using expanders
with st.expander("Data Information"):
    buffer = pd.DataFrame({
        'Column': data.columns,
        'Type': data.dtypes,
        'Non-Null Count': data.count(),
        'Null Count': data.isnull().sum()
    })
    st.dataframe(buffer)

with st.expander("Statistical Summary"):
    st.dataframe(data.describe())

# Data Preprocessing Section
st.header("2. Data Preprocessing and Feature Engineering")

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract additional time-based features
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create connection pairs for analysis
data['Connection'] = data['Source_Node'] + ' → ' + data['Destination_Node']

st.write("Updated dataset with new features:")
st.dataframe(data.head())

# Handle categorical data
categorical_cols = ['Source_Node', 'Destination_Node', 'Admin_Contact', 'Region_Code']
st.subheader("Categorical Features")
for col in categorical_cols:
    st.write(f"Unique values in {col}:")
    st.write(data[col].value_counts())

# Check for outliers
numerical_cols = ['Packet_Loss_Rate', 'Average_Latency_ms', 'Node_Betweenness_Centrality', 
                 'Traffic_Volume_MBps', 'Link_Stability_Score']

# Function to identify outliers using IQR method
def identify_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound, len(outliers)

# Function to handle outliers using capping
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    return data

st.subheader("Outlier Analysis")
with st.expander("Show Outlier Analysis Details"):
    for col in numerical_cols:
        outliers, lower, upper, count = identify_outliers(data, col)
        st.write(f"Outliers in {col}:")
        st.write(f"Lower bound: {lower:.4f}, Upper bound: {upper:.4f}")
        st.write(f"Number of outliers: {count} ({(count/len(data))*100:.2f}% of data)")

# Handle outliers for each numerical column
data_processed = data.copy()
for col in numerical_cols:
    data_processed = cap_outliers(data_processed, col)

# Create a congestion level target variable
# Normalize the factors for weighted scoring
loss_norm = (data_processed['Packet_Loss_Rate'] - data_processed['Packet_Loss_Rate'].min()) / \
           (data_processed['Packet_Loss_Rate'].max() - data_processed['Packet_Loss_Rate'].min())

latency_norm = (data_processed['Average_Latency_ms'] - data_processed['Average_Latency_ms'].min()) / \
              (data_processed['Average_Latency_ms'].max() - data_processed['Average_Latency_ms'].min())

# Combined congestion score (weighted average)
data_processed['Congestion_Score'] = 0.6 * loss_norm + 0.4 * latency_norm

# Create congestion level categories (Low, Medium, High)
data_processed['Congestion_Level'] = pd.qcut(data_processed['Congestion_Score'], 
                                           q=[0, 0.33, 0.67, 1.0], 
                                           labels=['Low', 'Medium', 'High'])

st.subheader("Target Variable Creation")
st.write("We created a 'Congestion_Level' target variable based on 'Packet_Loss_Rate' and 'Average_Latency_ms'.")
st.dataframe(data_processed[['Packet_Loss_Rate', 'Average_Latency_ms', 'Congestion_Score', 'Congestion_Level']].head())

# Plot the distribution of congestion levels
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Congestion_Level', data=data_processed, ax=ax)
plt.title('Distribution of Network Congestion Levels')
plt.xlabel('Congestion Level')
plt.ylabel('Count')
st.pyplot(fig)

# Exploratory Data Analysis Section
st.header("3. Exploratory Data Analysis (EDA)")

# Distribution of numerical features
st.subheader("Distribution of Numerical Features")
fig = plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(data_processed[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
st.pyplot(fig)

# Correlation matrix for numerical features
st.subheader("Correlation Analysis")
numerical_data = data_processed[numerical_cols + ['Congestion_Score']]
correlation_matrix = numerical_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
st.pyplot(fig)

# Interactive correlation heatmap with Plotly
st.subheader("Interactive Correlation Heatmap")
fig = px.imshow(correlation_matrix, 
               text_auto='.2f',
               color_continuous_scale='RdBu_r',
               title='Correlation Matrix of Numerical Features')
fig.update_layout(width=800, height=800)
st.plotly_chart(fig)

# Hourly patterns in congestion
st.subheader("Temporal Patterns in Network Congestion")
hourly_congestion = data_processed.groupby('Hour')['Congestion_Score'].mean().reset_index()

fig = plt.figure(figsize=(12, 6))
plt.plot(hourly_congestion['Hour'], hourly_congestion['Congestion_Score'], marker='o', linestyle='-')
plt.title('Average Congestion Score by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Congestion Score')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
st.pyplot(fig)

# Interactive hourly pattern with Plotly
fig = px.line(hourly_congestion, x='Hour', y='Congestion_Score', markers=True,
             title='Average Congestion Score by Hour of Day')
fig.update_layout(xaxis_title='Hour of Day', 
                 yaxis_title='Average Congestion Score',
                 xaxis=dict(tickmode='linear', tick0=0, dtick=1))
st.plotly_chart(fig)

# Analyze congestion by day of week
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
              4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
data_processed['DayName'] = data_processed['DayOfWeek'].map(day_mapping)

daily_congestion = data_processed.groupby('DayName')['Congestion_Score'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='DayName', y='Congestion_Score', data=daily_congestion, ax=ax)
plt.title('Average Congestion Score by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Congestion Score')
plt.grid(True, alpha=0.3)
st.pyplot(fig)

# Regional analysis
st.subheader("Regional Analysis")
region_congestion = data_processed.groupby('Region_Code')['Congestion_Score'].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Region_Code', y='Congestion_Score', data=region_congestion, ax=ax)
plt.title('Average Congestion Score by Region')
plt.xlabel('Region')
plt.ylabel('Average Congestion Score')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Connection pair analysis
st.subheader("Connection Pair Analysis")
connection_congestion = data_processed.groupby('Connection')['Congestion_Score'].mean().sort_values(ascending=False).reset_index()
top_10_congested = connection_congestion.head(10)

fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(x='Congestion_Score', y='Connection', data=top_10_congested, ax=ax)
plt.title('Top 10 Most Congested Connection Pairs')
plt.xlabel('Average Congestion Score')
plt.ylabel('Connection Pair')
plt.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# Scatter plots to examine relationships
st.subheader("Relationship between Key Variables")
fig = plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(x='Packet_Loss_Rate', y='Average_Latency_ms', hue='Congestion_Level', data=data_processed)
plt.title('Packet Loss Rate vs. Average Latency')

plt.subplot(2, 2, 2)
sns.scatterplot(x='Traffic_Volume_MBps', y='Packet_Loss_Rate', hue='Congestion_Level', data=data_processed)
plt.title('Traffic Volume vs. Packet Loss Rate')

plt.subplot(2, 2, 3)
sns.scatterplot(x='Node_Betweenness_Centrality', y='Packet_Loss_Rate', hue='Congestion_Level', data=data_processed)
plt.title('Node Betweenness Centrality vs. Packet Loss Rate')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Link_Stability_Score', y='Packet_Loss_Rate', hue='Congestion_Level', data=data_processed)
plt.title('Link Stability Score vs. Packet Loss Rate')

plt.tight_layout()
st.pyplot(fig)

# Interactive scatter plot with Plotly
st.subheader("Interactive Scatter Plot")
features = st.multiselect('Select features for interactive plot:', 
                        numerical_cols, 
                        default=['Packet_Loss_Rate', 'Average_Latency_ms'])

if len(features) >= 2:
    fig = px.scatter(data_processed, 
                    x=features[0], 
                    y=features[1], 
                    color='Congestion_Level',
                    hover_data=['Connection', 'Region_Code', 'Admin_Contact'],
                    title=f'{features[0]} vs {features[1]} by Congestion Level')
    st.plotly_chart(fig)
else:
    st.write("Please select at least two features for the scatter plot.")

# Machine Learning Section
st.header("4. Machine Learning Models")

# Prepare data for machine learning
st.subheader("Data Preparation for ML")

# Select features and target
X = data_processed.drop(['Packet_ID', 'Timestamp', 'Congestion_Level', 'Congestion_Score', 
                        'Connection', 'DayName'], axis=1)
y = data_processed['Congestion_Level']

# Get categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

st.write("Features:")
st.write(f"Numerical features: {num_cols}")
st.write(f"Categorical features: {cat_cols}")
st.write(f"Target variable: Congestion_Level")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Testing set size: {X_test.shape[0]} samples")

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Model Training and Evaluation
st.subheader("Model Training and Evaluation")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
results = {}
feature_importances = {}

# Add progress bar
progress_bar = st.progress(0)
status_text = st.empty()

for i, (name, model) in enumerate(models.items()):
    status_text.text(f"Training {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': pipeline,
        'predictions': y_pred
    }
    
    # Update progress bar
    progress_bar.progress((i + 1) / len(models))

status_text.text("Model training complete!")

# Display results
st.subheader("Model Performance Comparison")
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'Precision': [results[model]['precision'] for model in results],
    'Recall': [results[model]['recall'] for model in results],
    'F1 Score': [results[model]['f1'] for model in results]
})
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
st.dataframe(results_df)

# Plot model comparison
st.subheader("Model Performance Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df['Model']))
width = 0.2
multiplier = 0

for attribute, measurement in results_df.drop('Model', axis=1).items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width, results_df['Model'], rotation=45)
plt.legend(loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# Interactive plot
fig = go.Figure()
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df[metric],
        name=metric
    ))

fig.update_layout(
    title='Model Performance Comparison',
    xaxis_title='Model',
    yaxis_title='Score',
    barmode='group',
    width=800,
    height=500
)
st.plotly_chart(fig)

# Feature Importance Analysis
st.header("5. Feature Importance Analysis")

# Get the best model based on F1 score
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

# Interpret feature importance (method depends on the model type)
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
    st.subheader(f"Feature Importance for {best_model_name}")
    
    # Extract feature importances if the model supports it
    model_step = best_model.named_steps['model']
    
    # Get the feature names after preprocessing
    preprocessor = best_model.named_steps['preprocessor']
    feature_names = []
    
    # Get numerical feature names
    if num_cols:
        feature_names.extend(num_cols)
    
    # Get one-hot encoded feature names for categorical features
    if cat_cols:
        # Get the OneHotEncoder
        onehotencoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        # Get the encoded feature names
        categorical_features = onehotencoder.get_feature_names_out(cat_cols)
        feature_names.extend(categorical_features)
    
    # Extract feature importances
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        
        # Create a DataFrame for the feature importances
        if len(importances) == len(feature_names):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display the feature importances
            st.dataframe(importance_df)
            
            # Plot the feature importances
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
            plt.title(f'Top 15 Feature Importances - {best_model_name}')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write(f"Feature importance extraction not supported for this configuration of {best_model_name}")
    else:
        st.write(f"Feature importance extraction not supported for {best_model_name}")
    
    # For permutation importance (works with any model)
    with st.spinner("Calculating permutation importance (this may take a while)..."):
        # Extract preprocessor and model from pipeline
        processor = best_model.named_steps['preprocessor']
        model = best_model.named_steps['model']
        
        # Process the data
        X_test_processed = processor.transform(X_test)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test_processed, y_test, 
                                            n_repeats=5, random_state=42)
        
        # If we have feature names
        if len(perm_importance.importances_mean) == len(feature_names):
            perm_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Permutation Feature Importance")
            st.dataframe(perm_importance_df)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(15), ax=ax)
            plt.title('Top 15 Permutation Feature Importances')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Permutation importance could not be calculated with feature names")

# Confusion Matrix for the best model
st.subheader("Confusion Matrix")
best_preds = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, best_preds)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=y.unique(), yticklabels=y.unique(), ax=ax)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, best_preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Conclusion Section
st.header("6. Conclusion")
st.markdown(f"""
### Model Performance Summary
The best performing model is **{best_model_name}** with:
- Accuracy: {results[best_model_name]['accuracy']:.4f}
- Precision: {results[best_model_name]['precision']:.4f}
- Recall: {results[best_model_name]['recall']:.4f}
- F1 Score: {results[best_model_name]['f1']:.4f}

### Key Findings from Analysis
1. The temporal patterns show variations in congestion levels throughout the day.
2. Some regions and connection pairs consistently show higher congestion levels.
3. There are clear correlations between certain network metrics and congestion.
4. The most important features for predicting congestion are displayed in the feature importance analysis.

### Recommendations
Based on the analysis, network administrators should:
1. Focus resources on the most congested connection pairs
2. Implement traffic management during peak hours
3. Consider capacity upgrades for regions with consistently high congestion
4. Monitor the key metrics identified in the feature importance analysis
""")

st.markdown("""
---
### Thank you for using the Network Congestion Analysis Tool
""")