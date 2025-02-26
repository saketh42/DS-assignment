# Power Plant Data Analysis and Prediction

## Overview
This project involves generating a synthetic dataset for power plants, preprocessing the data, handling missing values, outliers, and noise, and finally training a Support Vector Regression (SVR) model to predict plant efficiency.

## Dataset Generation
The dataset consists of 1,000 records of power plants with features such as:
- **Power_Plant_ID**: Unique identifier
- **Plant_Location**: City where the plant is located
- **Operating_Capacity_MW**: Capacity in megawatts
- **Fuel_Type**: Type of fuel used (Coal, Gas, Nuclear, Renewable)
- **Emission_Level_CO2_tonnes**: CO2 emissions in tonnes
- **Operational_Years**: Age of the plant
- **Maintenance_Cost_MUSD**: Maintenance cost in million USD
- **Environmental Conditions**: Temperature, Humidity, Wind Speed, Soil Moisture
- **Reservoir Details**: Dam height, reservoir capacity, sedimentation rate
- **Efficiency_percent**: Efficiency of the plant

### Additional Features:
- **Missing Values**: Introduced in some numerical columns.
- **Outliers**: Extreme CO2 emission values added.
- **Duplicates**: Some rows were duplicated.
- **Noise**: Small modifications in numerical values.

## Data Preprocessing
The dataset is preprocessed with the following steps:
1. **Visualization**
   - Missing values visualization using `missingno`
   - Pair plots for numerical features using `seaborn`
2. **Handling Missing Values**
   - Imputed missing numerical values using the mean strategy.
3. **Handling Duplicates**
   - Removed duplicate rows.
4. **Handling Outliers**
   - Used Z-score method to filter out extreme outliers.
5. **Handling Noise**
   - Applied a rolling median filter to smooth numerical data.
6. **Feature Scaling**
   - Standardized numerical features using `StandardScaler`.

## Model Training
We trained an **SVR (Support Vector Regression)** model to predict **Efficiency_percent** using:
- **Features**: All numerical columns except `Power_Plant_ID`, `Fuel_Type`, and `Plant_Location`
- **Train-Test Split**: 80% training, 20% testing
- **Model Performance Metrics**:
  - **Mean Squared Error (MSE)**
  - **R-squared (R²) Score**

## Results
After training the SVR model:
- **MSE**: Evaluates the model's error rate.
- **R² Score**: Indicates the proportion of variance explained by the model.

## Usage
### Install Dependencies
```sh
pip install pandas numpy faker mimesis scipy scikit-learn missingno seaborn matplotlib
```

### Run the Project
```sh
python power_plant_analysis.py
```

## Output
- `power_plant_dataset.csv` - Raw dataset
- `power_plant_preprocessed_dataset.csv` - Cleaned dataset
- Model evaluation metrics displayed in the console.

## Future Improvements
- Try different regression models like **Random Forest Regressor** or **Neural Networks**.
- Feature engineering to improve predictive performance.
- Deploy the model as a REST API for real-time predictions.

## License
This project is licensed under the MIT License.
