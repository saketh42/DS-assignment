# python libraries
import io               
import pandas as pd
import numpy as np

# data generation
from faker import Faker
from mimesis import Generic

# data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fake = Faker()
generic = Generic("en")


def generate_dataset(n_records=5000):
    ''' Power_Plant_ID: Unique ID for each power plant
        Plant_Location: City where the power plant is located
        Operating_Capacity_MW: Power generation capacity in megawatts
        Fuel_Type: Type of fuel used (Coal, Gas, Nuclear, Renewable)
        Emission_Level_CO2_tonnes: CO2 emissions in tonnes
        Operational_Years: Number of years the plant has been operational
        Maintenance_Cost_MUSD: Annual maintenance cost in million USD
        Temperature_C: Average temperature near the plant in Celsius
        Humidity_percent: Average humidity percentage near the plant
        Wind_Speed_kmh: Average wind speed near the plant in km/h
        Soil_Moisture_percent: Percentage of soil moisture near the plant
        Dam_Height_m: Height of the associated dam in meters
        Reservoir_Capacity_MCM: Reservoir capacity in million cubic meters
        Sedimentation_Rate_mpy: Rate of sediment accumulation in meters per year
        Efficiency_percent: Operational efficiency of the plant as a percentage '''

    data = {
        "Power_Plant_ID": [fake.unique.uuid4() for _ in range(n_records)],
        "Plant_Location": [fake.city() for _ in range(n_records)],
        "Operating_Capacity_MW": [generic.random.uniform(50, 1000) for _ in range(n_records)],
        "Fuel_Type": np.random.choice(["Coal", "Gas", "Nuclear", "Renewable"], 
                                    size=n_records, p=[0.3, 0.3, 0.2, 0.2]),
        "Emission_Level_CO2_tonnes": [generic.random.uniform(1000, 50000) for _ in range(n_records)],
        "Operational_Years": [generic.random.randint(1, 60) for _ in range(n_records)],
        "Maintenance_Cost_MUSD": [generic.random.uniform(0.5, 20) for _ in range(n_records)],
        "Temperature_C": [generic.random.uniform(15, 45) for _ in range(n_records)],
        "Humidity_percent": [generic.random.uniform(20, 90) for _ in range(n_records)],
        "Wind_Speed_kmh": [generic.random.uniform(0, 100) for _ in range(n_records)],
        "Soil_Moisture_percent": [generic.random.uniform(5, 40) for _ in range(n_records)],
        "Dam_Height_m": [generic.random.uniform(30, 300) for _ in range(n_records)],
        "Reservoir_Capacity_MCM": [generic.random.uniform(10, 5000) for _ in range(n_records)],
        "Sedimentation_Rate_mpy": [generic.random.uniform(0.1, 1) for _ in range(n_records)],
        "Efficiency_percent": [generic.random.uniform(60, 95) for _ in range(n_records)]}

    df = pd.DataFrame(data)
    
    # Add missing values
    for col in ["Operating_Capacity_MW", "Emission_Level_CO2_tonnes", "Efficiency_percent"]:
        indices = np.random.choice(n_records, size=100, replace=False)
        df.loc[indices, col] = np.nan

    # Add outliers
    outlier_indices = np.random.choice(n_records, size=100, replace=False)
    df.loc[outlier_indices, "Emission_Level_CO2_tonnes"] *= 10

    # Add duplicates
    duplicate_indices = np.random.choice(n_records, size=100, replace=False)
    duplicate_rows = df.iloc[duplicate_indices]
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Add noise
    noise_indices = np.random.choice(n_records, size=10, replace=False)
    df.loc[noise_indices, "Operating_Capacity_MW"] += np.random.uniform(-50, 50)
    
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    df = df.copy()
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Handle outliers using IQR
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(series, lower_bound, upper_bound)
    
    df["Emission_Level_CO2_tonnes"] = cap_outliers(df["Emission_Level_CO2_tonnes"])
    df["Operating_Capacity_MW"] = cap_outliers(df["Operating_Capacity_MW"])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Fuel_Type_encoded'] = le.fit_transform(df['Fuel_Type'])
    
    return df

# Streamlit App
st.title("Power Plant Data Analysis Dashboard")

# Generate data
n_records = 5000

if 'raw_df' not in st.session_state:
    st.session_state.n_records = n_records
    st.session_state.raw_df = generate_dataset(n_records)
    st.session_state.processed_df = preprocess_data(st.session_state.raw_df)


# Sidebar navigation
page = st.sidebar.radio("Navigation", 
    ["Dataset Overview", "Data Quality Analysis", "Model Comparison", "Correlations & Distributions"])

if page == "Dataset Overview":
    st.header("Dataset Overview")
    
    # Toggle between raw and processed data
    data_type = st.radio("Select Data Type:", ["Raw Data", "Processed Data"])
    df_to_show = st.session_state.raw_df if data_type == "Raw Data" else st.session_state.processed_df
    
    # Show basic information
    st.subheader("Data Sample")
    st.dataframe(df_to_show.head())
    
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df_to_show.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.subheader("Statistical Summary")
    st.write(df_to_show.describe())

elif page == "Data Quality Analysis":
    st.header("Data Quality Analysis")
    
    # Missing values analysis
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Raw Data': st.session_state.raw_df.isnull().sum(),
        'Processed Data': st.session_state.processed_df.isnull().sum()
    })
    st.bar_chart(missing_df)
    
    # Duplicates analysis
    st.subheader("Duplicate Rows")
    dupes_data = {
        'Raw Data': st.session_state.raw_df.duplicated().sum(),
        'Processed Data': st.session_state.processed_df.duplicated().sum()
    }
    st.bar_chart(dupes_data)
    
    # Outliers visualization
    st.subheader("Outlier Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.raw_df[['Operating_Capacity_MW', 'Emission_Level_CO2_tonnes']])
        plt.xticks(rotation=45)
        plt.title("Raw Data Outliers")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.processed_df[['Operating_Capacity_MW', 'Emission_Level_CO2_tonnes']])
        plt.xticks(rotation=45)
        plt.title("Processed Data Outliers")
        st.pyplot(fig)

elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    # Train models
    features = ['Operating_Capacity_MW', 'Emission_Level_CO2_tonnes', 'Operational_Years',
                'Maintenance_Cost_MUSD', 'Temperature_C', 'Humidity_percent', 'Wind_Speed_kmh',
                'Soil_Moisture_percent', 'Dam_Height_m', 'Reservoir_Capacity_MCM',
                'Sedimentation_Rate_mpy']
    
    # Raw data model
    X_raw = st.session_state.raw_df[features].fillna(0)
    y_raw = st.session_state.raw_df['Efficiency_percent'].fillna(0)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2)
    
    model_raw = RandomForestRegressor(random_state=42)
    model_raw.fit(X_train_raw, y_train_raw)
    y_pred_raw = model_raw.predict(X_test_raw)
    
    # Processed data model
    X_proc = st.session_state.processed_df[features]
    y_proc = st.session_state.processed_df['Efficiency_percent']
    X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(X_proc, y_proc, test_size=0.2)
    
    model_proc = RandomForestRegressor(random_state=42)
    model_proc.fit(X_train_proc, y_train_proc)
    y_pred_proc = model_proc.predict(X_test_proc)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Model")
        st.write(f"MAE: {mean_absolute_error(y_test_raw, y_pred_raw):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test_raw, y_pred_raw):.4f}")

    with col2:
        st.subheader("Processed Data Model")
        st.write(f"MAE: {mean_absolute_error(y_test_proc, y_pred_proc):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test_proc, y_pred_proc):.4f}")
    # Feature importance plot
    st.subheader("Feature Importance Comparison")
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Raw Model': model_raw.feature_importances_,
        'Processed Model': model_proc.feature_importances_
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.plot(x='Feature', y=['Raw Model', 'Processed Model'], kind='bar', ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Correlations & Distributions":
    st.header("Feature Correlations and Distributions")
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = st.session_state.processed_df.select_dtypes(include=[np.number]).columns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(st.session_state.processed_df[numeric_cols].corr(), 
                annot=True, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature:", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=st.session_state.raw_df, x=selected_feature, kde=True)
        plt.title("Raw Data Distribution")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(data=st.session_state.processed_df, x=selected_feature, kde=True)
        plt.title("Processed Data Distribution")
        st.pyplot(fig)
    
    # Scatter plot
    st.subheader("Feature Relationships")
    feature1 = st.selectbox("Select First Feature:", numeric_cols, index=0)
    feature2 = st.selectbox("Select Second Feature:", numeric_cols, index=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=st.session_state.raw_df, x=feature1, y=feature2)
        plt.title("Raw Data Scatter Plot")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=st.session_state.processed_df, x=feature1, y=feature2)
        plt.title("Processed Data Scatter Plot")
        st.pyplot(fig)

