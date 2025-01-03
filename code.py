import shap   
import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
import pkg_resources

# Read data from the repository's CSV file
data = pd.read_csv("Data.csv", encoding='ISO-8859-1')


# Bin the "Age" column into 5-year intervals
bins = range(0, data['Age'].max() + 5, 5)
data['AgeGroup'] = pd.cut(data['Age'], bins=bins)

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")


st.title("Anomaly Detection Analysis of Enerjisa Feeder Outage Data Using Isolation Forest")
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""

## Project Overview
This project analyzes feeder outage data collected during an internship at Enerjisa. Due to confidentiality agreements,
certain sensitive information has been redacted from the shared version of the project.

## Data Processing & Feature Engineering
* Created specialized features to analyze outage patterns:
    * **Inventory_Blackout_count**: Comparison metric for individual feeder outages vs inventory average
    * **Inventory_Blackout_mean**: Average outage rate for feeders within same inventory group
    * **Diff**: Delta analysis showing deviation of individual feeder performance from inventory average

## Key Components
### 1. Anomaly Detection
* Implements Isolation Forest algorithm to identify unusual outage patterns
* Contamination factor set to 0.1 for optimal anomaly detection

### 2. Visualization
* Age distribution analysis using 5-year intervals
* Blackout frequency distribution
* SHAP value analysis for feature importance
* Interactive anomaly score visualization

### 3. Maintenance System
* Real-time feeder maintenance tracking
* Interactive maintenance list management
* Historical maintenance record keeping

## Technical Features
* Streamlit-based interactive dashboard
* SHAP (SHapley Additive exPlanations) for model interpretability
* Seaborn/Matplotlib for statistical visualizations
* Pandas for data manipulation and analysis
* Scikit-learn for machine learning implementations

---
*Note: While this version uses modified data to comply with NDAs, the analytical methodology 
and system architecture remain true to the original implementation.*
""")


st.write("# Age Distribution Analysis")
st.write("""
This chart shows the distribution of feeder ages in 5-year intervals. Understanding age distribution helps identify potential maintenance needs and lifecycle patterns in our infrastructure.
""")
# Create a bar chart for Age
age_group_counts = data['AgeGroup'].value_counts().sort_index()
plt.figure(figsize=(12, 8))
sns.barplot(x=age_group_counts.index.astype(str), y=age_group_counts.values, palette="coolwarm")
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Age Distribution in 5-Year Intervals', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)


st.write("# Blackout Frequency Distribution")
st.write("""
This graph shows the blackout frequency distribution across feeders. 
Each bar represents the number of feeders experiencing a specific count of blackouts. 
For better visualization, all blackout counts above 50 are grouped together. 
This helps identify patterns in outage frequency and highlight feeders that may require attention.
""")


# Create a modified blackout count series
modified_blackouts = data["Blackout_count"].copy()
modified_blackouts[modified_blackouts > 50] = 51  # Group all counts > 50 into 51

plt.figure(figsize=(12, 8))
blackout_counts = modified_blackouts.value_counts().sort_index()
sns.barplot(x=blackout_counts.index.astype(str), y=blackout_counts.values, palette="coolwarm")
plt.xlabel('Blackout Count', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Blackout Count (>50 grouped)', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)



# Perform anomaly detection using Isolation Forest with numeric features only
numeric_features = data.select_dtypes(include=[np.number])
iso_forest = IsolationForest(contamination=0.1)
data['Anomaly_Score'] = iso_forest.fit_predict(numeric_features)
data['Anomaly_Score'] = iso_forest.decision_function(numeric_features)

# Feature Importance Analysis
iso_forest = IsolationForest(contamination=0.1)
data['Anomaly_Score'] = iso_forest.fit_predict(numeric_features)
data['Anomaly_Score'] = iso_forest.decision_function(numeric_features)
st.write("# Isolation Forest Model Analysis")
st.write("""
The Isolation Forest algorithm is an unsupervised learning method specifically designed for anomaly detection. 
In this analysis:
- Contamination parameter is set to 0.1 (10% of data points are assumed to be anomalies)
- Negative scores indicate anomalies, while positive scores indicate normal data points
- Features used include numerical data like age, blackout counts, and inventory metrics
""")

# Display the code
st.code("""
# Isolation Forest implementation
iso_forest = IsolationForest(contamination=0.1)
data['Anomaly_Score'] = iso_forest.fit_predict(numeric_features)
data['Anomaly_Score'] = iso_forest.decision_function(numeric_features)

# Calculate SHAP values
explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X_sample)
""", language='python')


# Calculate SHAP values more efficiently by using a subset of data if the dataset is large
sample_size = min(1000, len(numeric_features))
X_sample = numeric_features.sample(n=sample_size, random_state=28)

# Initialize TreeExplainer which is more efficient for tree-based models
explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X_sample)

# Create SHAP summary plot
st.header("Feature Importance Analysis (SHAP)")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X_sample,
    plot_type="bar",
    show=False,
    max_display=10  # Show only top 10 features
)
plt.tight_layout()
st.pyplot(fig)


st.write("""
### Key Observations from SHAP Analysis:

1. **Primary Factors:**
    - Age is the most influential feature (highest SHAP value ~0.45)
    - Customer count ranks second in importance
    - Inventoru blackout count shows significant impact as the third most important feature

2. **Secondary Factors:**
    - Blackout count and Diff demonstrate moderate but meaningful influence
    - Inventory_Blackout_mean show lower but still relevant impact

## Conclusions:
- Temporal and customer-related variables are the strongest predictors
- Infrastructure age and comparative metrics provide supporting insights
- The model successfully identifies key operational indicators for anomaly detection

This analysis helps prioritize which factors to monitor most closely for preventive maintenance.
""")


st.write("# Anomaly Score Visualization")

# Plot the anomaly scores for better visualization
plt.figure(figsize=(12,8))
scatter = sns.scatterplot(x=data.index, y='Anomaly_Score', data=data, hue='Anomaly_Score', palette='coolwarm', alpha=0.7)
scatter.legend(loc='lower right')
plt.title('Anomaly Scores', fontsize=16)
plt.tight_layout()
st.pyplot(plt)

st.write("""
This visualization shows the anomaly scores for each feeder in our dataset. 
Key points about this plot:
- Each point represents a feeder
- The y-axis shows the anomaly score (negative values indicate potential anomalies)
- Color gradient helps identify severity (darker red indicates more anomalous behavior)
- Points closer to -1 are more likely to be anomalies
- Points closer to 0 represent normal behavior

This helps us quickly identify feeders that may require immediate attention based on their unusual behavior patterns.
""")


st.write("""
## Feeder Anomaly Rankings and Details

This section provides a comprehensive view of the anomaly scores for all feeders, sorted from most anomalous to least anomalous. 
Negative scores indicate potential anomalies that may require attention.

Below, you can:
- View the complete ranked list of feeders and their anomaly scores
- Select a specific feeder to examine its detailed characteristics
- Use this information to prioritize maintenance and inspections
""")
# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Display the anomaly scores in the first column
with col1:
    st.write("Anomaly Scores:")
    st.write(data[['Anomaly_Score', 'Feeder']].sort_values(by='Anomaly_Score', ascending=True))

# Display the feeder details in the second column
with col2:
    # Add a selectbox for feeder selection
    unique_feeders = data['Feeder'].unique()
    selected_feeder = st.selectbox("Select a Feeder to Review:", unique_feeders)
    # Display all columns except AgeGroup
    columns_to_show = [col for col in data.columns if col != 'AgeGroup']
    st.write(data[data['Feeder'] == selected_feeder][columns_to_show])



class Feeder:
    def __init__(self, name):
        self.name = name
        self.needs_maintenance = True

class MaintenanceSystem:
    def __init__(self):
        self.feeders = []
        # DataFrame to track maintenance done
        self.maintenance_done = pd.DataFrame(columns=["Feeder", "Timestamp"])

    def add_feeder(self, feeder):
        self.feeders.append(feeder)

    def list_maintenance_required(self):
        return [f for f in self.feeders if f.needs_maintenance]

def perform_maintenance(self, feeder_name):
    feeder = next((f for f in self.feeders if f.name == feeder_name and f.needs_maintenance), None)
    if feeder:
        feeder.needs_maintenance = False
        self.feeders.remove(feeder)
        # Add to maintenance_done with timestamp
        new_data = pd.DataFrame([{
            "Feeder": feeder_name
        }])
        self.maintenance_done = pd.concat([self.maintenance_done, new_data], ignore_index=True)


# Create DataFrame from your actual feeder data
data = pd.DataFrame({
    "Feeder": data['Feeder'].unique()  # Using the feeders from your CSV file
})

# Initialize system with feeders from data
if "system" not in st.session_state:
    st.session_state.system = MaintenanceSystem()
    unique_feeders = data['Feeder'].unique()
    for feeder_name in unique_feeders:
        st.session_state.system.add_feeder(Feeder(feeder_name))

# Feeder selection and maintenance
feeders_required = st.session_state.system.list_maintenance_required()
feeder_names = [feeder.name for feeder in feeders_required]


st.header("Feeder Maintenance Management System")
st.write("""
This system allows you to manage feeder maintenance tasks and view the maintenance history.
Select feeders that need maintenance and track completed maintenance activities.
""")

# Sütunları oluşturma
col1, col2 = st.columns(2)

# 1. sütunda, Feeder listesi gösteriliyor
with col1:
    st.subheader('Feeder List')
    feeders = data["Feeder"].tolist()
    selected_feed = st.multiselect("Select Feeders", feeders)
    if st.button('Add To List'):
        st.session_state.selected_feed = selected_feed

# 2. sütunda Maintenance List gösteriliyor
with col2:
    st.subheader('Maintenance List')
    if 'selected_feed' in st.session_state and len(st.session_state.selected_feed) > 0:
        # Seçilen feederları bir tablo şeklinde göster
        maintenance_list = pd.DataFrame(st.session_state.selected_feed, columns=["Feeder"])
        st.table(maintenance_list)
    else:
        st.write("No Feeder in Maintenance List.")

# List of specific packages to show in the display
core_packages = [
    "shap",
    "numpy",
    "pandas",
    "streamlit",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pkg_resources"
]

# Get versions for display
display_requirements = [
    f"{dist.key}=={dist.version}" 
    for dist in pkg_resources.working_set 
    if dist.key in core_packages
]

# Display only core packages
st.code('\n'.join(display_requirements), language='text')




st.markdown("""
<div style='text-align: center; padding: 20px; color: #666; font-size: 14px; font-style: italic;'>
    Please reach out to yusufff.poyraz@gmail.com if you have any questions or feedback.
</div>
""", unsafe_allow_html=True)
