import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import pickle
import io
import pkg_resources
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pkg_resources
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# Set the page configuration
st.set_page_config(
    page_title="ML Supervised APP",  # This changes the display name in the browser tab
    page_icon="💡",       # Optional: Add a custom favicon (emoji or path to an icon file)
    layout="wide"         # Optional: Use a wide layout
)


# Inject custom CSS to reduce padding and margins
st.markdown("""
    <style>
        .reportview-container {
            padding: 0;
        }
        .main {
            max-width: 100%;
        }
        .block-container {
            padding: 1rem 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables if they do not exist
if "step" not in st.session_state:
    st.session_state.step = 0  # Set the default step to 0 (Home page)

if "data" not in st.session_state:
    st.session_state.data = None  # Data loaded by the user

if "filtered_data1" not in st.session_state:
    st.session_state.filtered_data = None  # Initialize filtered data after preprocessing

if "filtered_data2" not in st.session_state:
    st.session_state.filtered_data1 = None  # For additional processing if needed

# For model training and evaluation
if "X_train" not in st.session_state:
    st.session_state.X_train = None  # Features for training
if "X_test" not in st.session_state:
    st.session_state.X_test = None  # Features for testing
if "y_train" not in st.session_state:
    st.session_state.y_train = None  # Target for training
if "y_test" not in st.session_state:
    st.session_state.y_test = None  # Target for testing

# Model-related
if "model" not in st.session_state:
    st.session_state.model = None  # The best trained model selected
if "target" not in st.session_state:
    st.session_state.target = None  # Target column name
if "features" not in st.session_state:
    st.session_state.features = None  # List of selected features

if "target1" not in st.session_state:
    st.session_state.target1 = None  # Secondary target column name (if applicable)
if "features1" not in st.session_state:
    st.session_state.features1 = None  # Secondary list of features (if applicable)

# Handling for missing values, outliers, scaling, and encoding
if "missing_handling" not in st.session_state:
    st.session_state.missing_handling = {}  # Dictionary to track missing value handling methods
if "outlier_handling" not in st.session_state:
    st.session_state.outlier_handling = {}  # Dictionary to track outlier handling methods
if "scaling_methods" not in st.session_state:
    st.session_state.scaling_methods = {}  # Dictionary to track scaling methods applied
if "encoding_methods" not in st.session_state:
    st.session_state.encoding_methods = {}  # Dictionary to track encoding methods applied

# Hyperparameter tuning and search
if "param_grid" not in st.session_state:
    st.session_state.param_grid = {}  # Parameter grid for tuning
if "best_params" not in st.session_state:
    st.session_state.best_params = None  # Best parameters from tuning
if "grid_search" not in st.session_state:
    st.session_state.grid_search = None  # GridSearchCV object for tuning

# Hyperparameter tuning state variables
if "tuning_method" not in st.session_state:
    st.session_state.tuning_method = None  # Method for tuning (e.g., GridSearch or RandomizedSearch)

# Models selection and best model tracking
if "models" not in st.session_state:
    st.session_state.models = {}  # Dictionary to store trained models
if "best_model" not in st.session_state:
    st.session_state.best_model = None  # Instance or name of the best-performing model


# Function to navigate between steps
def navigate(step):
    st.session_state.step = step

# Steps
steps = [
    "🏠 Home",
    "📂 Upload and Preview Dataset",
    "🎯 Target and Features Selection",
    "🔍 Handling Duplicate Data",
    "🛠️ Handling Missing Data",
    "📊 Handling Outliers",
    "📏 Feature Scaling",
    "🔄 Encoding",
    "✂️ Train-Test Split",
    "🤖 Train the Models",
    "📊 Comparison of Models & Hyperparameter Tuning",
    "🔮 Step 11: Predict Using Selected Model"
]



for idx, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"nav_{idx}"):
        navigate(idx)



# Define the Home Page
def home_page():
    
    
    st.title("🤖 Welcome to ML Supervised APP!")
    st.markdown("""
    **ML Supervised APP** is designed to simplify the machine learning process by automating key steps 
    like data preprocessing, model selection, and evaluation. Whether you're a data scientist, 
    analyst, or enthusiast, this tool empowers you to build robust models with minimal effort.
    """)

    st.subheader("✨ Key Features:")
    st.markdown("""
    - 📂 **Upload Data**: Start by uploading your dataset in CSV or Excel format.
    - 🎯 **Target Selection**: Easily define the target variable and features for analysis.
    - 🔍 **Data Cleaning**: Handle duplicates, missing values, and outliers effortlessly.
    - 📊 **Data Scaling & Encoding**: Automatically prepare data for machine learning models.
    - 🤖 **Model Training**: Train multiple models and compare their performance.
    - 📈 **Visualization**: Generate intuitive visualizations for insights and model performance.
    """)

    st.subheader("⚙️ How It Works:")
    st.markdown("""
    1. **Upload Your Dataset**: Start with your raw data in CSV or Excel format.
    2. **Preprocess the Data**: Clean, scale, and encode data through an interactive interface.
    3. **Choose Models**: Let the system train and evaluate various algorithms.
    4. **Analyze Results**: Compare model metrics and choose the best performer.
    5. **Download Outputs**: Get reports, visualizations, and model files for further use.
    """)

    st.markdown("---")
    st.markdown("🚀 **Get started now by navigating to the first step from the sidebar!**")
    st.balloons()

# Step 0: Home
if st.session_state.step == 0:
    home_page()



# Step 1: Upload and Preview Dataset
if st.session_state.step == 1:
    st.title("📂 Step 1: Upload and Preview Dataset")
    st.write("Upload your dataset (CSV or Excel) and get a quick overview using automated profiling.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format):", type=["csv", "xlsx"], key="upload_file")

    if uploaded_file is not None:
        # Start a timer for estimated loading time
        start_time = time.time()

        # Show a loading spinner while processing the file
        with st.spinner("Processing the file..."):
            try:
                # Check file extension and read the file
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.data = pd.read_csv(uploaded_file)  # Store original dataset
                elif uploaded_file.name.endswith(".xlsx"):
                    st.session_state.data = pd.read_excel(uploaded_file)  # Store original dataset
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    st.stop()

                # Initialize filtered_data as a copy of the original dataset
                st.session_state.filtered_data1 = st.session_state.data.copy()
                st.session_state.filtered_data2 = st.session_state.data.copy()

                

                # Show a preview of the first 5 rows of the dataset
                st.write("### Dataset Preview")
                st.dataframe(st.session_state.data.head())

                # Validate column names and data types
                st.write("### Dataset Columns and Types")
                st.write(st.session_state.data.dtypes)

                # Button to trigger detailed EDA
                if st.button("Show Detailed EDA Report"):
                    # Generate profiling report for in-app display using YData Profiling
                    profile_in_app = ProfileReport(
                        st.session_state.data,
                        title="YData Profiling Report",
                        explorative=True,
                    )

                    # Save a downloadable report to a file (includes navigation)
                    downloadable_report_path = "ydata_profiling_report.html"
                    profile_in_app.to_file(downloadable_report_path)
                    with open(downloadable_report_path, "rb") as f:
                        st.download_button(
                            label="💾 Download Full Profiling Report",
                            data=f,
                            file_name="ydata_profiling_report.html",
                            mime="text/html",
                        )

                    # Display the profiling report in the Streamlit app
                    st_profile_report(profile_in_app)

                    # Display the elapsed time for generating the report
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Report generated in {elapsed_time:.2f} seconds! 🎉")
            except Exception as e:
                st.error(f"🚨 An error occurred while processing the file: {e}")

if st.session_state.step == 2:
    st.title("🎯 Step 2: Target and Features Selection")
    st.write("Select the target variable and the features for your analysis.")
    
    st.subheader("📌 What are Target and Features?")
    st.write("""
    **Target Variable**  
    - The target variable (also called the dependent variable or label) is the column you want to predict or analyze.  
    - For example:
        - In a regression problem, this could be **house prices** or **sales amount**.
        - In a classification problem, this could be **spam/non-spam** or **loan approval status**.

    **Input Features**  
    - Input features (also called independent variables or predictors) are the columns used to make predictions about the target variable.  
    - They provide the model with information to identify patterns and relationships.  
    - For example:
        - In predicting house prices, input features could include **square footage**, **number of bedrooms**, and **location**.
        - For loan approval, input features might be **income**, **credit score**, and **employment history**.

    💡 **Tip**: Select features that are most relevant to your target variable for better model performance.
    """)

    # Ensure that data is available
    if st.session_state.data is not None:
        # Step 2: Select Target and Features
        st.subheader("Step 2: Select Target and Features")
        st.markdown(
    "<h2 style='color: blue;'> Select the target column:</h2>", 
    unsafe_allow_html=True
        )

        # Select the target column
        target_col = st.selectbox(
            "target column:", 
            st.session_state.data.columns, 
            key="target_column", 
            index=st.session_state.data.columns.get_loc(st.session_state.target) if st.session_state.target else 0
        )
        
        # Show all columns for feature selection except the target column
        feature_cols = [col for col in st.session_state.data.columns if col != target_col]
        features = st.multiselect(
            "Select feature columns:", 
            feature_cols, 
            default=st.session_state.features if st.session_state.features else []
        )

        # Show the selected target and features
        if target_col and features:
            st.subheader("Step 1: Define Column Data Types")

            # Create the selected columns list (target + features)
            selected_columns = [target_col] + features

            # Display initial data types
            st.write("### Initial Column Data Types")
            initial_data_types = st.session_state.data.dtypes[selected_columns]
            st.write(initial_data_types)

            # Set data types for the selected columns with default values
            column_data_types = {}
            for column in selected_columns:
                # Get the current data type for the column from the dataset
                current_dtype = str(st.session_state.data[column].dtype)
                st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {column}:</h3>", 
                        unsafe_allow_html=True
                        )

                # Set default data type in the selectbox
                dtype = st.selectbox(
                    f"{column}", 
                    ["int64", "float64", "object"], 
                    index=["int64", "float64", "object"].index(current_dtype) if current_dtype in ["int64", "float64", "object"] else 0,
                    key=f"datatype_{column}"
                )
                column_data_types[column] = dtype

            # Show the defined data types for confirmation
            st.write("### Defined Column Data Types:") 
            st.write(column_data_types)

            # Confirm selection button
            if st.button("Confirm Selection", key="confirm_selection"):
                # Update session state with the target, features, and column data types
                st.session_state.target = target_col
                st.session_state.target1 = target_col  # Save a copy for other steps
                st.session_state.features = features
                st.session_state.features1 = features  # Save a copy for other steps
                st.session_state.column_data_types = column_data_types

                # Apply the data type changes to the dataset
                for column, dtype in column_data_types.items():
                    if dtype == "int64":
                        st.session_state.data[column] = st.session_state.data[column].astype(int)
                        st.session_state.filtered_data1[column] = st.session_state.filtered_data1[column].astype(int)
                        st.session_state.filtered_data2[column] = st.session_state.filtered_data2[column].astype(int)
                    elif dtype == "float64":
                        st.session_state.data[column] = st.session_state.data[column].astype(float)
                        st.session_state.filtered_data1[column] = st.session_state.filtered_data1[column].astype(float)
                        st.session_state.filtered_data2[column] = st.session_state.filtered_data2[column].astype(float)
                    elif dtype == "object":
                        st.session_state.data[column] = st.session_state.data[column].astype(str)
                        st.session_state.filtered_data1[column] = st.session_state.filtered_data1[column].astype(str)
                        st.session_state.filtered_data2[column] = st.session_state.filtered_data2[column].astype(str)
                    

                # Update the filtered datasets to only contain the selected target and features
                st.session_state.filtered_data1 = st.session_state.data[selected_columns].copy()
                st.session_state.filtered_data2 = st.session_state.filtered_data1.copy()  # For Step 11 testing

                # Debugging output: show the selected columns and updated data
                st.write("### Updated Data After Target and Feature Selection")
                st.write(f"Target: {st.session_state.target}")
                st.write(f"Features: {st.session_state.features}")
                st.write("### Updated Dataset (Filtered Data 1)")
                st.dataframe(st.session_state.filtered_data1.head())

                # Show success message
                st.success("✅ Target and features selected successfully, and data types updated! 🎉")
        else:
            st.warning("🚨 Please select both target and feature columns first.")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


if st.session_state.step == 3:
    st.title("🔍 Step 3: Handling Duplicate Data")
    st.write("Duplicate data can lead to biased analysis, inaccurate models, and inefficient processing, so it's important to remove them.")

    # Ensure the target and features are selected from Step 2
    if st.session_state.target and st.session_state.features:
        # Use filtered_data1 from Step 2
        filtered_data = st.session_state.filtered_data1

        # Count the number of duplicate rows before handling
        num_duplicates_before = filtered_data.duplicated().sum()
        st.write("Number of duplicate rows before handling:", num_duplicates_before)

        # Show the current number of rows before removing duplicates
        rows_before = len(filtered_data)
        st.write(f"Total number of rows before removing duplicates: {rows_before}")

        # Display the first few rows of the dataset
        st.write("### Dataset Preview Before Removing Duplicates")
        st.dataframe(filtered_data.head())

        if num_duplicates_before > 0:
            # Option to remove duplicates
            if st.button("Remove Duplicates", key="remove_duplicates"):
                # Start a timer for processing time
                start_time = time.time()

                # Remove duplicates and update session state
                with st.spinner("Removing duplicates, please wait..."):
                    st.session_state.filtered_data1 = filtered_data.drop_duplicates().reset_index(drop=True)

                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                st.success(f"✅ Duplicates removed in {elapsed_time:.2f} seconds! 🎉")

                # Show the updated row count
                rows_after = len(st.session_state.filtered_data1)
                st.write(f"Total number of rows after removing duplicates: {rows_after}")

                # Count the number of duplicate rows after handling
                num_duplicates_after = st.session_state.filtered_data1.duplicated().sum()
                st.write("Number of duplicate rows after handling:", num_duplicates_after)

                # Show the updated dataset
                st.write("### Updated Dataset (after removing duplicates):")
                st.dataframe(st.session_state.filtered_data1.head())

        else:
            st.success("✅ No duplicate rows found in the dataset! 🎉")

    else:
        st.warning("🚨 Please select the target and feature columns in Step 2.")

    # If the dataset is not loaded yet
    if st.session_state.data is None:
        st.warning("🚨 Please upload a dataset in Step 1.")




if st.session_state.step == 4:
    st.title("🛠️ Step 4: Handling Missing Data")
    st.write("""
        Missing data can distort analysis and reduce model accuracy, so handling it properly is crucial. 
        In this step, you will be able to choose strategies for handling missing values in your dataset.
    """)
    
    st.subheader("📌 Use Cases for Handling Missing Data")
    
    st.write("""
    **1. Mean Imputation**  
    - Use when the data is **numerical** and **normally distributed** (no significant skew or outliers).  

    **2. Median Imputation**  
    - Use when the data is **numerical** with **skewed distributions** or contains **outliers**.  

    **3. Mode Imputation**  
    - Use when the data is **categorical** or for numerical columns with a distinct and meaningful **most frequent value**.
    """)


    if st.session_state.data is not None:
        # Create a filtered dataset with only the selected columns (target and features)
        filtered_data = st.session_state.filtered_data1  # Use the filtered_data1 from previous steps

        # Debugging: Show a preview of the data before processing
        st.write("### Preview of Data (Before Handling Missing Data):")
        st.dataframe(filtered_data.head())

        # Ensure all missing values (None, empty string, or NaN) are treated as NaN
        filtered_data = filtered_data.replace(['nan', 'NA', 'N/A', None, ' '], pd.NA)

        # Calculate the number and percentage of missing values for each selected column
        missing_count = filtered_data.isnull().sum()
        missing_percentage = (missing_count / len(filtered_data)) * 100
        missing_info = pd.DataFrame({"Missing Count": missing_count, "Missing Percentage": missing_percentage})

        # Debugging: Show the missing data summary
        st.write("### Missing Data Summary (Before Handling):")
        st.dataframe(missing_info)

        # Provide the option to handle missing data for each selected column (including the target column)
        for col in filtered_data.columns:
            if missing_count[col] > 0:
                st.write(f"#### Column: {col} ({missing_count[col]} missing values)")

                # Treat the column as categorical if it is selected by the user
                if st.session_state.column_data_types.get(col) == "object":
                    st.write(f"Missing data in this categorical column. Option: Fill with Mode.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                    strategy = st.selectbox(
                        f"{col}:",
                        ["None", "Fill with mode", "Drop rows"],
                        key=f"missing_strategy_{col}",
                        index=["None", "Fill with mode", "Drop rows"].index(
                            st.session_state.missing_handling.get(col, "None")
                        ),
                    )
                else:  # Numerical columns
                    st.write(f"Missing data in this numerical column. Options: Fill with Mean, Median.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                    strategy = st.selectbox(
                        f"{col}:",
                        ["None", "Fill with mean", "Fill with median", "Drop rows"],
                        key=f"missing_strategy_{col}",
                        index=["None", "Fill with mean", "Fill with median", "Drop rows"].index(
                            st.session_state.missing_handling.get(col, "None")
                        ),
                    )

                st.session_state.missing_handling[col] = strategy

                # Display mean/median/mode for numerical or categorical columns
                if st.session_state.column_data_types.get(col) != "object":  # Numerical columns
                    st.write(f"**Mean**: {filtered_data[col].mean():.2f}, **Median**: {filtered_data[col].median():.2f}")
                else:  # Categorical columns
                    st.write(f"**Mode**: {filtered_data[col].mode().iloc[0]}")

        # Handle missing data when the button is pressed
        if st.button("Handle Missing Data", key="handle_missing"):
            # Start a timer to calculate the time taken for handling missing data
            start_time = time.time()

            # Show a loading spinner while handling the missing data
            with st.spinner("Handling missing data, please wait..."):
                # Show progress bar while handling missing data
                progress_bar = st.progress(0)
                for i, (col, strategy) in enumerate(st.session_state.missing_handling.items()):
                    if strategy == "Drop rows":
                        filtered_data = filtered_data.dropna(subset=[col])
                    elif strategy == "Fill with mean" and st.session_state.column_data_types.get(col) != "object":
                        filtered_data[col] = filtered_data[col].fillna(filtered_data[col].mean())
                    elif strategy == "Fill with median" and st.session_state.column_data_types.get(col) != "object":
                        filtered_data[col] = filtered_data[col].fillna(filtered_data[col].median())
                    elif strategy == "Fill with mode":
                        filtered_data[col] = filtered_data[col].fillna(filtered_data[col].mode().iloc[0])

                    # Simulate progress for each column
                    time.sleep(0.1)
                    progress_bar.progress(int((i + 1) / len(st.session_state.missing_handling) * 100))

                # Handle missing values in the target column (if applicable)
                target_column = st.session_state.target
                if target_column in filtered_data.columns:
                    target_missing_count = filtered_data[target_column].isnull().sum()
                    if target_missing_count > 0:
                        st.markdown(
                        f"<h3 style='color: blue;'>Select strategy for {target_column}:</h3>", 
                        unsafe_allow_html=True
                        )
                        target_strategy = st.selectbox(
                            f"({target_column}):",
                            ["None", "Fill with mode", "Drop rows"],
                            key=f"missing_strategy_target",
                            index=["None", "Fill with mode", "Drop rows"].index(
                                st.session_state.missing_handling.get(target_column, "None")
                            ),
                        )
                        st.session_state.missing_handling[target_column] = target_strategy

                        if target_strategy == "Drop rows":
                            filtered_data = filtered_data.dropna(subset=[target_column])
                        elif target_strategy == "Fill with mode":
                            filtered_data[target_column] = filtered_data[target_column].fillna(
                                filtered_data[target_column].mode().iloc[0]
                            )

                # Calculate the elapsed time and display the result
                elapsed_time = time.time() - start_time
                st.success(f"✅ Missing data handled in {elapsed_time:.2f} seconds! 🎉")

                # Now, update the main dataset with the modified filtered_data1
                st.session_state.filtered_data1 = filtered_data  # Update filtered data1 in session state

                # Show the updated dataset and row count
                st.write("### Updated Dataset (after handling missing data):")
                st.dataframe(filtered_data)

                # Calculate the number and percentage of missing values after handling
                missing_count_after = filtered_data.isnull().sum()
                missing_percentage_after = (missing_count_after / len(filtered_data)) * 100
                missing_info_after = pd.DataFrame({"Missing Count": missing_count_after, "Missing Percentage": missing_percentage_after})

                # Show a table with missing data summary after handling
                st.write("### Missing Data Summary After Handling")
                st.dataframe(missing_info_after)

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


if st.session_state.step == 5:
    st.title("📊 Step 5: Handling Outliers")
    st.write("Outliers can skew data distributions and impact model performance, so identifying and managing them is essential.")

    st.subheader("📌 Use Cases for Handling Outliers")

    st.write("""
    **1. IQR Method (Interquartile Range)**  
    - Use when you want a **robust statistical approach** to identify and handle outliers.  
    - Particularly useful for **numerical data** and when the data distribution is not heavily skewed.  
    - It removes outliers beyond the lower (Q1 - 1.5 * IQR) and upper (Q3 + 1.5 * IQR) bounds.

    **2. Capping Outliers**  
    - Use when outliers are important but need to be **limited to a maximum or minimum value** for consistency.  
    - Helps in cases where you don't want to lose data but want to **reduce the impact** of extreme values.  
    - For example, capping salaries at a maximum value to reduce skewness in income data.

    **3. Removing Outliers**  
    - Use when the outliers are **erroneous data points** or are **not relevant** to the analysis.  
    - Ideal when the dataset is large, and removing a small percentage of outliers won't affect overall insights.  
    - For instance, removing sensor readings that are clearly invalid or impossible values in real-world data.
    """)

    if st.session_state.data is not None:
        # Ensure outlier_handling is initialized
        if "outlier_handling" not in st.session_state:
            st.session_state.outlier_handling = {}

        # Use selected feature columns instead of all numerical columns
        numerical_cols = [col for col in st.session_state.features if st.session_state.data[col].dtype in ["float", "int"]]

        # Identify columns with outliers using the IQR method
        outlier_columns = []
        for col in numerical_cols:
            Q1 = st.session_state.data[col].quantile(0.25)
            Q3 = st.session_state.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Check if there are any outliers
            if any(st.session_state.data[col] < lower_bound) or any(st.session_state.data[col] > upper_bound):
                outlier_columns.append(col)

        if outlier_columns:
            st.write("Columns with outliers detected:")
            for col in outlier_columns:
                st.write(f"**{col}**")
                # Plotting the boxplot for columns with outliers
                fig, ax = plt.subplots(figsize=(10, 1))  # Create a figure and axis object
                sns.boxplot(x=st.session_state.data[col], ax=ax, color='lightblue', flierprops=dict(marker='o', color='red', markersize=5))
                st.pyplot(fig)  # Pass the figure to st.pyplot()
                

                # Select method for handling outliers
                st.markdown(
                    f"<h3 style='color: blue;'>Select method to handle outliers in {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                method = st.selectbox(f" {col}:",
                                      ["None", "IQR", "Cap outliers", "Remove outliers"],
                                      key=f"outlier_method_{col}",
                                      index=["None", "IQR", "Cap outliers", "Remove outliers"].index(st.session_state.outlier_handling.get(col, "None")))

                st.session_state.outlier_handling[col] = method

            if st.button("Apply Outlier Handling", key="apply_outliers"):
                # Start a timer for estimated time
                start_time = time.time()

                # Show a loading spinner while handling outliers
                with st.spinner("Handling outliers, please wait..."):
                    # Show progress bar while handling outliers
                    progress_bar = st.progress(0)
                    total_columns = len(outlier_columns)

                    # Handle outliers for each column
                    for idx, col in enumerate(outlier_columns):
                        method = st.session_state.outlier_handling[col]

                        Q1 = st.session_state.data[col].quantile(0.25)
                        Q3 = st.session_state.data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        if method == "IQR":
                            st.session_state.filtered_data1 = st.session_state.filtered_data1[(st.session_state.filtered_data1[col] >= lower_bound) & (st.session_state.filtered_data1[col] <= upper_bound)]
                        elif method == "Cap outliers":
                            st.session_state.filtered_data1[col] = np.clip(st.session_state.filtered_data1[col], lower_bound, upper_bound)
                        elif method == "Remove outliers":
                            st.session_state.filtered_data1 = st.session_state.filtered_data1[(st.session_state.filtered_data1[col] >= lower_bound) & (st.session_state.filtered_data1[col] <= upper_bound)]

                        # Update progress bar
                        progress_bar.progress(int((idx + 1) / total_columns * 100))
                        time.sleep(0.5)  # Simulate processing time for each column

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Outliers handled in {elapsed_time:.2f} seconds! 🎉")

                    # Show box plots for the columns again after handling outliers
                    st.write("Updated box plots for columns after handling outliers:")
                    for col in outlier_columns:
                        fig, ax = plt.subplots(figsize=(10, 1))  # Create a figure and axis object
                        sns.boxplot(x=st.session_state.filtered_data1[col], ax=ax, color='lightgreen', flierprops=dict(marker='o', color='red', markersize=5))
                        st.pyplot(fig)  # Pass the figure to st.pyplot()

                    # Show the updated data after handling outliers
                    st.write("### Updated Dataset (after handling outliers):")
                    st.dataframe(st.session_state.filtered_data1)

        else:
            st.write("✅ No outliers detected in the dataset. 🎉")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

if st.session_state.step == 6:
    st.title("📏 Step 6: Feature Scaling")
    st.write("Feature scaling ensures that all features contribute equally to the model by standardizing their ranges, improving model performance and convergence.")

    st.subheader("📌 Use Cases for Scaling Methods")

    st.write("""
    **1. Standard Scaling**  
    - Use when the data follows a **normal distribution** or approximately Gaussian.  
    - Commonly used in algorithms like **logistic regression**, **linear regression**, and **SVMs** where assumptions of normality hold.  

    **2. Min-Max Scaling**  
    - Use when the data needs to be **scaled to a specific range** (e.g., [0, 1] or [-1, 1]).  
    - Ideal for algorithms like **neural networks** and **gradient-based models** that are sensitive to feature magnitude.  
    - Be cautious with outliers, as they can significantly affect the scaling.  

    **3. Robust Scaling**  
    - Use when the dataset contains **outliers**, as this method is robust to extreme values.  
    - Suitable for data that is not normally distributed and has heavy-tailed distributions.  
    - Often used in preprocessing data for models like **tree-based algorithms**.  

    **4. MaxAbs Scaling**  
    - Use when the data is **already centered at zero** or when handling **sparse data**.  
    - Ideal for models like **principal component analysis (PCA)** or **matrix factorization techniques** that require preserving sparsity.  
    """)


    if st.session_state.filtered_data1 is not None:
        # Ensure scaling_methods is initialized
        if "scaling_methods" not in st.session_state:
            st.session_state.scaling_methods = {}

        # Separate numerical and categorical columns based on updated data types
        numerical_cols = []
        categorical_cols = []

        for col in st.session_state.features:
            if st.session_state.filtered_data1[col].dtype in ["float64", "int64"]:
                # Check if the column has strictly increasing or sequential values
                if np.all(np.diff(st.session_state.filtered_data1[col].dropna().sort_values()) == 1):
                    categorical_cols.append(col)  # Treat as categorical
                else:
                    numerical_cols.append(col)  # Treat as numerical
            else:
                categorical_cols.append(col)  # Treat non-numerical columns as categorical

        # Exclude target variable from scaling if it is numerical
        if st.session_state.target in numerical_cols:
            numerical_cols = [col for col in numerical_cols if col != st.session_state.target]
            st.warning(f"The target variable '{st.session_state.target}' will not be scaled. It is excluded from feature scaling.")

        # Show preview of data before scaling
        st.write("### Preview of Data Before Scaling:")
        st.write(st.session_state.filtered_data1[numerical_cols].head())

        # Proceed with scaling if there are numerical columns
        if len(numerical_cols) > 0:
            # Allow the user to select a scaling method for each numerical column
            for col in numerical_cols:
                st.markdown(
                    f"<h3 style='color: blue;'>Select scaling method for {col}</h3>", 
                        unsafe_allow_html=True
                        )
                scaling_method = st.selectbox(f" {col}:",
                                              ["None", "Standard Scaling", "Min-Max Scaling", "Robust Scaling", "MaxAbs Scaling"],
                                              key=f"scaling_method_{col}",
                                              index=["None", "Standard Scaling", "Min-Max Scaling", "Robust Scaling", "MaxAbs Scaling"].index(st.session_state.scaling_methods.get(col, "None")))

                st.session_state.scaling_methods[col] = scaling_method

            if st.button("Apply Scaling", key="scaling_apply"):
                # Start a timer for estimated time
                start_time = time.time()

                # Show a loading spinner while applying scaling
                with st.spinner("Applying feature scaling, please wait..."):
                    # Show progress bar while applying scaling
                    progress_bar = st.progress(0)
                    total_columns = len(numerical_cols)

                    for idx, col in enumerate(numerical_cols):
                        method = st.session_state.scaling_methods[col]

                        if method == "Standard Scaling":
                            scaler = StandardScaler()
                            st.session_state.filtered_data1[col] = scaler.fit_transform(st.session_state.filtered_data1[[col]])
                        elif method == "Min-Max Scaling":
                            scaler = MinMaxScaler()
                            st.session_state.filtered_data1[col] = scaler.fit_transform(st.session_state.filtered_data1[[col]])
                        elif method == "Robust Scaling":
                            scaler = RobustScaler()
                            st.session_state.filtered_data1[col] = scaler.fit_transform(st.session_state.filtered_data1[[col]])
                        elif method == "MaxAbs Scaling":
                            scaler = MaxAbsScaler()
                            st.session_state.filtered_data1[col] = scaler.fit_transform(st.session_state.filtered_data1[[col]])

                        # Update progress bar
                        time.sleep(0.5)  # Simulate processing time for each column
                        progress_bar.progress(int((idx + 1) / total_columns * 100))

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Feature scaling applied in {elapsed_time:.2f} seconds! 🎉")

                    # Show preview of filtered data after scaling
                    st.write("### Preview of Data After Scaling:")
                    st.write(st.session_state.filtered_data1[numerical_cols].head())

        else:
            st.warning("✅ No numerical columns found in the dataset. You can skip this step.")
            st.write("Since there are no numerical columns, feature scaling is not necessary. You can proceed to the next step.")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


if st.session_state.step == 7:
    st.title("🔄 Step 7: Encoding")
    st.write("Encoding converts categorical data into numerical format, enabling machine learning models to process and learn from it effectively.")

    st.subheader("📌 Use Cases for Encoding Methods")

    st.write("""
    **1. Label Encoding**  
    - Use when the data is **ordinal** (categories have a meaningful order, e.g., **Low, Medium, High**).  
    - Suitable for models that can handle the ordinal relationship directly (e.g., **tree-based models**).  
    - Avoid for nominal data to prevent introducing unintended relationships.  

    **2. One-Hot Encoding**  
    - Use when the data is **nominal** (categories have no intrinsic order, e.g., **Red, Blue, Green**).  
    - Ideal for algorithms that work well with sparse data, such as **logistic regression** or **neural networks**.  
    - Can lead to high memory usage for features with many unique categories.  

    **3. Target Encoding**  
    - Use for **high-cardinality categorical features** (e.g., hundreds or thousands of unique values).  
    - Works well for tree-based models like **XGBoost** or **CatBoost**.  
    - Be cautious of **data leakage**; apply this method separately to train and test data.  

    **4. Binary Encoding**  
    - Use when the data contains **high-cardinality categorical features** but you want to reduce dimensionality compared to one-hot encoding.  
    - Suitable for models that handle binary features effectively, such as **logistic regression** or **tree-based models**.  

    **5. Ordinal Encoding**  
    - Use when the data has a **predefined ranking or order** (e.g., **Education Level: High School < Bachelor's < Master's**).  
    - Appropriate for models that benefit from ordinal relationships (e.g., **decision trees**).  
    """)


    if st.session_state.filtered_data1 is not None:
        # Ensure encoding_methods is initialized
        if "encoding_methods" not in st.session_state:
            st.session_state.encoding_methods = {}

        # Use only selected categorical feature columns
        existing_columns = [col for col in st.session_state.features if col in st.session_state.filtered_data1.columns]
        categorical_cols = [col for col in existing_columns if st.session_state.filtered_data1[col].dtype in ["object", "category"]]

        if st.session_state.target in categorical_cols:
            # Automatically apply Label Encoding if the target is categorical
            le = LabelEncoder()
            st.session_state.filtered_data1[st.session_state.target] = le.fit_transform(st.session_state.filtered_data1[st.session_state.target])
            categorical_cols = [col for col in categorical_cols if col != st.session_state.target]  # Exclude target from encoding

            st.warning(f"The target variable '{st.session_state.target}' has been label-encoded.")

        if len(categorical_cols) == 0:
            st.write("There are no categorical columns to encode. You can skip this step.")
        else:
            # Show preview of filtered data before encoding
            st.write("### Preview of Data Before Encoding:")
            st.write(st.session_state.filtered_data1[categorical_cols].head())

            encoding_update_button_pressed = False  # To track whether update buttons were pressed
            for col in categorical_cols:
                st.markdown(
                    f"<h3 style='color: blue;'>Select encoding method for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                encoding_method = st.selectbox(f" {col}:",
                                               ["None", "Label Encoding", "One-Hot Encoding", "Target Encoding", "Binary Encoding", "Ordinal Encoding"],
                                               key=f"encoding_method_{col}",
                                               index=["None", "Label Encoding", "One-Hot Encoding", "Target Encoding", "Binary Encoding", "Ordinal Encoding"].index(
                                                   st.session_state.encoding_methods.get(col, "None") if isinstance(st.session_state.encoding_methods.get(col, "None"), str) else "None"))

                st.session_state.encoding_methods[col] = encoding_method

                # For Ordinal Encoding, ask the user to provide the order
                if encoding_method == "Ordinal Encoding":
                    # Fetch unique categories from the column
                    unique_categories = st.session_state.filtered_data1[col].dropna().unique()
                    unique_categories = sorted(unique_categories)  # Sort categories (if applicable)

                    st.write(f"Categories in {col}: {unique_categories}")

                    # Ask the user to define the order of categories
                    order = st.text_area(f"Enter the order of categories for {col} (comma separated):",
                                         value=', '.join(map(str, unique_categories)),
                                         help="For example: 'low, medium, high'")

                    if order:
                        # Split the order input by commas and strip spaces
                        ordered_categories = [cat.strip() for cat in order.split(',')]
                        
                        if len(ordered_categories) == len(unique_categories):
                            # Map the ordered categories to integers
                            category_mapping = {cat: i for i, cat in enumerate(ordered_categories)}
                            st.session_state.encoding_methods[col] = category_mapping  # Save the mapping

                            # Provide the option to update after the user enters the order
                            encoding_update_button_pressed = True  # Mark as updated
                            st.success(f"Ordinal encoding applied to {col}. The encoded values are: {category_mapping}")
                        else:
                            st.warning("The number of categories in the order must match the number of unique categories in the column.")

            # Button to update the encoding methods
            if st.button("Update Encoding Methods", key="update_encoding"):
                st.success("Encoding methods updated! Click 'Apply Encoding' to finalize.")

            # Button to apply encoding for all columns at once
            if st.button("Apply Encoding", key="encoding_apply"):
                # Start a timer for estimated time
                start_time = time.time()

                # Show loading spinner and progress bar while encoding
                with st.spinner("Applying encoding, please wait..."):
                    progress_bar = st.progress(0)
                    total_columns = len(categorical_cols)

                    for idx, col in enumerate(categorical_cols):
                        method = st.session_state.encoding_methods[col]

                        # Apply selected encoding method
                        if method == "Label Encoding":
                            le = LabelEncoder()
                            st.session_state.filtered_data1[col] = le.fit_transform(st.session_state.filtered_data1[col])
                        elif method == "One-Hot Encoding":
                            st.session_state.filtered_data1 = pd.get_dummies(st.session_state.filtered_data1, columns=[col], drop_first=True)
                        elif method == "Target Encoding":
                            # Target encoding (mean of target per category)
                            target_mean = st.session_state.filtered_data1.groupby(col)[st.session_state.target].mean()
                            st.session_state.filtered_data1[col] = st.session_state.filtered_data1[col].map(target_mean)
                        elif method == "Binary Encoding":
                            # Binary encoding (for categorical variables with high cardinality)
                            binary_encoded = st.session_state.filtered_data1[col].apply(lambda x: format(int(x), 'b'))
                            st.session_state.filtered_data1[col] = binary_encoded
                        elif isinstance(method, dict):  # For Ordinal Encoding
                            # Apply ordinal encoding using the saved category mapping
                            st.session_state.filtered_data1[col] = st.session_state.filtered_data1[col].map(method)

                        # Update progress bar
                        time.sleep(0.5)  # Simulate processing time for each column
                        progress_bar.progress(int((idx + 1) / total_columns * 100))

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Encoding applied in {elapsed_time:.2f} seconds! 🎉")

                    # Show preview of filtered data after encoding
                    st.write("### Preview of Data After Encoding:")
                    st.write(st.session_state.filtered_data1.head())

                    # Update the features list to reflect the new columns after encoding
                    st.session_state.features = [col for col in st.session_state.filtered_data1.columns if col != st.session_state.target]

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")










if st.session_state.step == 8:
    st.title("✂️ Step 8: Train-Test Split")
    st.write("Train-Test Split divides the dataset into training and testing sets. This ensures the model is trained on one part of the data and tested on unseen data, allowing evaluation of its performance on new inputs.")

    st.subheader("📌 Understanding Train and Test Sets")

    st.write("""
    **1. X_train**  
    - Represents the **training features**, i.e., the independent variables used to train the model.  
    - Contains the majority of the dataset to ensure the model learns patterns effectively.  
    - For example: In predicting house prices, `X_train` may include features like square footage, number of bedrooms, and location.

    **2. X_test**  
    - Represents the **testing features**, i.e., the independent variables used to evaluate the model's performance.  
    - This set is kept separate from training to assess how well the model generalizes to unseen data.  
    - For example: If `X_test` contains house features, the model's predictions on this data are compared to the actual prices.  

    **3. y_train**  
    - Represents the **training target variable**, i.e., the dependent variable corresponding to `X_train`.  
    - This is the output the model learns to predict based on the input features.  
    - For example: In predicting house prices, `y_train` contains the actual prices for the corresponding `X_train`.  

    **4. y_test**  
    - Represents the **testing target variable**, i.e., the dependent variable corresponding to `X_test`.  
    - Used to compare the model's predictions with the actual values to evaluate its performance.  
    - For example: In the house price example, `y_test` contains the actual prices to validate predictions.  

    💡 **Where We Use Them:**  
    - **X_train, y_train**: Used to train the machine learning model and fit its parameters.  
    - **X_test, y_test**: Used after training to calculate evaluation metrics (e.g., accuracy, RMSE) and assess how well the model generalizes to unseen data.  
    """)

    if st.session_state.filtered_data1 is not None and st.session_state.target:
        # Extract features and target
        X = st.session_state.filtered_data1.drop(columns=[st.session_state.target])
        y = st.session_state.filtered_data1[st.session_state.target]

        # Show preview of the data before splitting
        st.write("### Preview of Data Before Train-Test Split:")
        st.write("**Features (X) Preview:**")
        st.dataframe(X.head())

        st.write("**Target (y) Preview:**")
        st.dataframe(y.head())

        # Slider for test size selection
        test_size = st.slider("Select test size ratio:", 0.1, 0.5, 0.2, step=0.1, key="test_size")

        if st.button("Split Data", key="split_data"):
            # Start timer for feedback
            start_time = time.time()

            # Perform train-test split
            with st.spinner("Splitting the data, please wait..."):
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            st.success(f"✅ Data split into train and test sets in {elapsed_time:.2f} seconds! 🎉")

            # Display shapes of resulting datasets
            st.write("**Training Set:**", st.session_state.X_train.shape)
            st.write("**Test Set:**", st.session_state.X_test.shape)

            # Display previews of datasets
            st.write("### Preview of Data After Train-Test Split:")

            st.write("**Training Features (X_train) Preview:**")
            st.dataframe(st.session_state.X_train.head())

            st.write("**Training Target (y_train) Preview:**")
            st.dataframe(st.session_state.y_train.head())

            st.write("**Test Features (X_test) Preview:**")
            st.dataframe(st.session_state.X_test.head())

            st.write("**Test Target (y_test) Preview:**")
            st.dataframe(st.session_state.y_test.head())
    else:
        st.warning("🚨 Please complete previous steps.")


if st.session_state.step == 9:
    st.title("🤖 Step 9: Train the Models")
    st.write("In this step, you'll train machine learning models using the training data. This process involves the model learning patterns and relationships within the data to make accurate predictions on unseen inputs.")

    st.subheader("📌 Understanding Task Types")
    st.write("""
    Machine learning problems can be broadly categorized into two task types:

    **1. Classification**  
    - The goal is to predict a **category or class** for the target variable.  
    - Examples:  
      - Determining whether an email is "Spam" or "Not Spam".  
      - Classifying images as "Cat", "Dog", or "Bird".  
      - Predicting customer churn ("Yes" or "No").  
    - Output: A discrete value representing a class label.

    **2. Regression**  
    - The goal is to predict a **continuous value** for the target variable.  
    - Examples:  
      - Predicting house prices based on features like location, size, and age.  
      - Forecasting stock prices or sales figures.  
      - Estimating the temperature for a given day.  
    - Output: A continuous numerical value.

    **Selecting the Task Type:**  
    - The task type depends on the nature of the target variable (`y_train`).  
    - If the target variable is categorical (e.g., "Yes"/"No", "Red"/"Blue"), the task is **Classification**.  
    - If the target variable is numerical (e.g., prices, age, or temperature), the task is **Regression**.
    """)
    # Ensure necessary session state variables exist
    if all(key in st.session_state and st.session_state[key] is not None for key in ["X_train", "y_train", "X_test", "y_test"]):
        # Detect task type based on target variable type
        target_type = st.session_state.y_train.dtypes
        if target_type == "object" or target_type.name == "category":
            suggested_task = "Classification"
        elif target_type in ["float64", "int64"]:
            suggested_task = "Regression"
        else:
            suggested_task = "Classification"

        # Provide a task type selection option (with default suggestion)
        task_type_options = ["Classification", "Regression"]
        st.radio(
            "Select Task Type:",
            task_type_options,
            index=task_type_options.index(suggested_task),
            key="task_type"
        )

        # Display available models based on task type
        if st.session_state.task_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Support Vector Classifier (SVC)": SVC(),
                "Naive Bayes": GaussianNB(),
            }
        elif st.session_state.task_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Voting Regressor": VotingRegressor([("lr", LinearRegression()), ("rf", RandomForestRegressor())]),
                "Stacking Regressor": StackingRegressor(estimators=[("lr", LinearRegression()), ("rf", RandomForestRegressor())], final_estimator=LinearRegression()),
                "XGBoost Regressor": XGBRegressor(),
                "SVR (Support Vector Regressor)": SVR(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
            }

        # Let the user select models to train
        selected_models = st.multiselect("Select models to train:", list(models.keys()), key="selected_models")

        # Show preview of selected models
        if selected_models:
            st.write("### Selected Models:")
            for model_name in selected_models:
                st.write(f"- {model_name}")

        # Train models on selected datasets
        if st.button("Train Selected Models"):
            if selected_models:
                st.session_state.models = {}
                start_time = time.time()
                with st.spinner("Training models, please wait..."):
                    progress_bar = st.progress(0)
                    for idx, model_name in enumerate(selected_models):
                        model = models[model_name]
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.models[model_name] = model

                        # Update progress bar
                        progress_bar.progress(int((idx + 1) / len(selected_models) * 100))
                    elapsed_time = time.time() - start_time
                st.success(f"✅ Models trained successfully in {elapsed_time:.2f} seconds!🎉")

                # Save selected models to session state for later use
                st.session_state.selected_models_list = selected_models
            else:
                st.error("⚠️ Please select at least one model to train.")
    else:
        st.warning("🚨 Please complete Step 8 (Train-Test Split) to prepare the training and testing data before proceeding.")



 # Save selected models


if st.session_state.step == 10:
    st.title("📊 Step 10: Comparison of Models")
    st.write("In this step, you'll compare the performance of different machine learning models. By analyzing metrics like accuracy, precision, recall, and others, you can identify the model best suited for your dataset and task.")

    task_type = st.session_state.get('task_type', 'Classification')

    # Save task_type in session state for Step 11
    st.session_state.task_type = task_type

    # Metric Descriptions
    if task_type == "Classification":
        st.write("""For classification tasks (categorical target variables), we calculate:""")
        st.write("""- **Accuracy**: The proportion of correct predictions.""")
        st.write("""- **Precision**: The ability to predict positive classes correctly""")
        st.write("""- **Recall**: The ability to find all positive classes.""")
        st.write("""- **F1 Score**: A balance between precision and recall.""")
        st.write("""- **Type 1 Error**: False Positive (Incorrectly predicted as positive)""")
        st.write("""- **Type 2 Error**: False Negative (Incorrectly predicted as negative)""")

    elif task_type == "Regression":
        st.write("""For regression tasks (numerical target variables), we calculate:""")
        st.write("""- **Mean Squared Error (MSE)**: Measures the average squared error""")
        st.write("""- **Mean Absolute Error (MAE)**: Measures the average absolute error""")
        st.write("""- **Root Mean Squared Error (RMSE)**: Measures the square root of MSE""")
        st.write("""- **R² Score**: Measures the proportion of variance explained by the model""")

    # Initialize containers for metrics data
    classification_metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Type 1 Error": [],
        "Type 2 Error": []
    }

    regression_metrics = {
        "R² Score": [],
        "MSE": [],
        "MAE": [],
        "RMSE": [],
    }

    # Processing models for comparison
    if st.session_state.models:
        for model_name, model in st.session_state.models.items():
            if task_type == "Classification" and isinstance(model, (LogisticRegression, RandomForestClassifier, SVC, GradientBoostingClassifier, DecisionTreeClassifier, GaussianNB)):
                y_pred = model.predict(st.session_state.X_test)

                # Classification metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = precision_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)

                classification_metrics["Accuracy"].append((model_name, accuracy))
                classification_metrics["Precision"].append((model_name, precision))
                classification_metrics["Recall"].append((model_name, recall))
                classification_metrics["F1 Score"].append((model_name, f1))

                # Type 1 and Type 2 Errors
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                type_1_error = cm[0][1]  # False Positive
                type_2_error = cm[1][0]  # False Negative
                classification_metrics["Type 1 Error"].append((model_name, type_1_error))
                classification_metrics["Type 2 Error"].append((model_name, type_2_error))

            elif task_type == "Regression" and isinstance(model, (LinearRegression, RandomForestRegressor, SVR, GradientBoostingRegressor, DecisionTreeRegressor)):
                y_pred = model.predict(st.session_state.X_test)

                # Regression metrics
                r2 = r2_score(st.session_state.y_test, y_pred)
                mse = mean_squared_error(st.session_state.y_test, y_pred)
                mae = mean_absolute_error(st.session_state.y_test, y_pred)
                rmse = np.sqrt(mse)

                regression_metrics["R² Score"].append((model_name, r2))
                regression_metrics["MSE"].append((model_name, mse))
                regression_metrics["MAE"].append((model_name, mae))
                regression_metrics["RMSE"].append((model_name, rmse))

        # Model comparison section with tabs for Classification and Regression
        st.subheader("Model Comparison")
        if task_type == "Classification":
            # Show Classification Metrics
            with st.expander("Classification Metrics Comparison"):
                st.write("### Accuracy Comparison")
                accuracy_df = pd.DataFrame(classification_metrics["Accuracy"], columns=["Model", "Accuracy"])
                st.dataframe(accuracy_df)

                st.write("### Precision Comparison")
                precision_df = pd.DataFrame(classification_metrics["Precision"], columns=["Model", "Precision"])
                st.dataframe(precision_df)

                st.write("### Recall Comparison")
                recall_df = pd.DataFrame(classification_metrics["Recall"], columns=["Model", "Recall"])
                st.dataframe(recall_df)

                st.write("### F1 Score Comparison")
                f1_df = pd.DataFrame(classification_metrics["F1 Score"], columns=["Model", "F1 Score"])
                st.dataframe(f1_df)

                # Adding the Type 1 and Type 2 Errors table
                st.write("### Type 1 and Type 2 Errors Comparison")
                type_error_df = pd.DataFrame({
                    "Model": [item[0] for item in classification_metrics["Type 1 Error"]],
                    "Type 1 Error (False Positive)": [item[1] for item in classification_metrics["Type 1 Error"]],
                    "Type 2 Error (False Negative)": [item[1] for item in classification_metrics["Type 2 Error"]]
                })
                st.dataframe(type_error_df)

                # Visualization for classification metrics
                st.write("### Visualization of Classification Metrics")
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

                for i, metric in enumerate(metrics):
                    ax = axes[i // 2, i % 2]
                    metric_data = pd.DataFrame(classification_metrics[metric], columns=["Model", metric])

                    if metric_data.empty:
                        st.warning(f"No data available for {metric} metric.")
                        continue

                    sns.barplot(x=metric_data["Model"], y=metric_data[metric], ax=ax, palette='viridis')
                    ax.set_title(f'{metric} Comparison')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                    for p in ax.patches:
                        if p.get_height() > 0:
                            ax.text(
                                p.get_x() + p.get_width() / 2,
                                p.get_height() + 0.01,
                                f'{p.get_height():.2f}',
                                ha='center',
                                va='bottom',
                                fontsize=9
                            )

                plt.subplots_adjust(hspace=1)
                st.pyplot(fig)

            # Handling empty metrics lists before calling max()
            if classification_metrics["Accuracy"]:
                best_accuracy_model = max(classification_metrics["Accuracy"], key=lambda x: x[1])
                st.write(f"**Best Model by Accuracy**: {best_accuracy_model[0]} with Accuracy: {best_accuracy_model[1]:.2f}🎉")
                st.session_state.best_model_for_tuning = best_accuracy_model[0]
                st.session_state.best_model = best_accuracy_model[0]
            else:
                st.warning("🚨 No classification models were trained or evaluated.")

        elif task_type == "Regression":
            # Show Regression Metrics
            with st.expander("Regression Metrics Comparison"):
                st.write("### R² Score Comparison")
                r2_df = pd.DataFrame(regression_metrics["R² Score"], columns=["Model", "R² Score"])
                st.dataframe(r2_df)

                st.write("### MSE Comparison")
                mse_df = pd.DataFrame(regression_metrics["MSE"], columns=["Model", "MSE"])
                st.dataframe(mse_df)

                st.write("### MAE Comparison")
                mae_df = pd.DataFrame(regression_metrics["MAE"], columns=["Model", "MAE"])
                st.dataframe(mae_df)

                st.write("### RMSE Comparison")
                rmse_df = pd.DataFrame(regression_metrics["RMSE"], columns=["Model", "RMSE"])
                st.dataframe(rmse_df)

                # Visualization for regression metrics
                st.write("### Visualization of Regression Metrics")
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                metrics = ['R² Score', 'MSE', 'MAE', 'RMSE']

                for i, metric in enumerate(metrics):
                    ax = axes[i // 2, i % 2]
                    metric_data = pd.DataFrame(regression_metrics[metric], columns=["Model", metric])

                    if metric_data.empty:
                        st.warning(f"No data available for {metric} metric.")
                        continue

                    sns.barplot(x=metric_data["Model"], y=metric_data[metric], ax=ax, palette='viridis')
                    ax.set_title(f'{metric} Comparison')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

                    for p in ax.patches:
                        if p.get_height() > 0:
                            ax.text(
                                p.get_x() + p.get_width() / 2,
                                p.get_height() + 0.02,
                                f'{p.get_height():.2f}',
                                ha='center',
                                va='bottom',
                                fontsize=9
                            )

                plt.subplots_adjust(hspace=1)
                st.pyplot(fig)

            # Handling empty metrics lists before calling max()
            if regression_metrics["R² Score"]:
                best_r2_model = max(regression_metrics["R² Score"], key=lambda x: x[1])
                st.write(f"**Best Model by R² Score**: {best_r2_model[0]} with R² Score: {best_r2_model[1]:.2f}🎉")
                st.session_state.best_model_for_tuning = best_r2_model[0]
                st.session_state.best_model = best_r2_model[0]
            else:
                st.warning("🚨 No regression models were trained or evaluated.")

        # Step: Hyperparameter Tuning
        best_model_name = st.session_state.get("best_model", None)
        best_model_instance = st.session_state.models.get(best_model_name, None)

        # Section to download the original model
        if best_model_instance is not None:
            st.write("### Download Original Model")
            original_model_buffer = io.BytesIO()
            pickle.dump(best_model_instance, original_model_buffer)
            original_model_buffer.seek(0)

            st.download_button(
                label="📥 Download Original Model",
                data=original_model_buffer,
                file_name=f"{best_model_name}_original_model.pkl",
                mime="application/octet-stream",
            )

        # Hyperparameter Tuning Step
        if best_model_instance:
            st.subheader(f"🔧 Hyperparameter Tuning for Best Model: **{best_model_name}**")
            col1, col2 = st.columns(2)  # Create two columns for side-by-side comparison

            # Initialize variables
            accuracy_before = precision_before = recall_before = f1_before = None
            r2_before = mse_before = mae_before = rmse_before = None

            # Text area for parameter grid input
            param_grid_input = st.text_area(
                "Enter your parameter grid as a Python dictionary:",
                value="{\n    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']\n}",
                height=150
            )

            # Button to validate the parameter grid
            if st.button("Validate Parameters"):
                try:
                    param_grid = eval(param_grid_input)
                    if not isinstance(param_grid, dict):
                        st.error("The parameter grid must be a dictionary. Please correct your input.")
                    else:
                        st.success("Parameter grid is valid!")
                        st.session_state.param_grid = param_grid  # Save in session state
                        st.write("### Parsed Parameter Grid:")
                        st.json(param_grid)
                except Exception as e:
                    st.error(f"🚨 Invalid parameter grid. Please ensure it follows Python dictionary syntax.\n\nError: {e}")

            # Check if parameters are validated
            if st.session_state.get("param_grid"):
                st.write("### Final Parameter Grid:")
                st.json(st.session_state.param_grid)

                # Select the tuning method
                st.session_state.tuning_method = st.radio(
                    "Choose the hyperparameter tuning method:",
                    ("Grid Search", "Randomized Search"),
                    key="tuning_method_selection"
                )

                # Hyperparameter tuning section
                if st.button("Start Hyperparameter Tuning"):
                    st.write(f"Starting **{st.session_state.tuning_method}** with the parameter grid provided...")

                    try:
                        # Define the tuner based on the selected method
                        if st.session_state.tuning_method == "Grid Search":
                            tuner = GridSearchCV(
                                estimator=best_model_instance,
                                param_grid=st.session_state.param_grid,
                                scoring="accuracy" if st.session_state.task_type == "Classification" else "r2",
                                cv=5,
                                n_jobs=-1,
                                verbose=2
                            )
                        elif st.session_state.tuning_method == "Randomized Search":
                            tuner = RandomizedSearchCV(
                                estimator=best_model_instance,
                                param_distributions=st.session_state.param_grid,
                                scoring="accuracy" if st.session_state.task_type == "Classification" else "r2",
                                n_iter=50,
                                cv=5,
                                n_jobs=-1,
                                verbose=2,
                                random_state=42
                            )

                        # Fit the tuner on the training data
                        tuner.fit(st.session_state.X_train, st.session_state.y_train)

                        st.success("Hyperparameter tuning completed!")
                        st.write("### Best Parameters:")
                        st.json(tuner.best_params_)

                        st.write(f"### Best {st.session_state.tuning_method} Score:")
                        st.write(f"{tuner.best_score_:.4f}")

                        # Save the tuned model in session state
                        st.session_state.tuned_model = tuner.best_estimator_

                        # Get predictions for both models after tuning
                        if st.session_state.X_test is not None and st.session_state.y_test is not None:
                            # Predictions from the original model
                            y_pred_before = best_model_instance.predict(st.session_state.X_test)

                            # Metrics for the original model
                            if st.session_state.task_type == "Classification":
                                accuracy_before = accuracy_score(st.session_state.y_test, y_pred_before)
                                precision_before = precision_score(st.session_state.y_test, y_pred_before, average='weighted', zero_division=0)
                                recall_before = recall_score(st.session_state.y_test, y_pred_before, average='weighted', zero_division=0)
                                f1_before = f1_score(st.session_state.y_test, y_pred_before, average='weighted', zero_division=0)
                            elif st.session_state.task_type == "Regression":
                                r2_before = r2_score(st.session_state.y_test, y_pred_before)
                                mse_before = mean_squared_error(st.session_state.y_test, y_pred_before)
                                mae_before = mean_absolute_error(st.session_state.y_test, y_pred_before)
                                rmse_before = np.sqrt(mse_before)

                            # Predictions from the tuned model
                            y_pred_after = st.session_state.tuned_model.predict(st.session_state.X_test)

                            # Metrics for the tuned model
                            if st.session_state.task_type == "Classification":
                                accuracy_after = accuracy_score(st.session_state.y_test, y_pred_after)
                                precision_after = precision_score(st.session_state.y_test, y_pred_after, average='weighted', zero_division=0)
                                recall_after = recall_score(st.session_state.y_test, y_pred_after, average='weighted', zero_division=0)
                                f1_after = f1_score(st.session_state.y_test, y_pred_after, average='weighted', zero_division=0)
                            elif st.session_state.task_type == "Regression":
                                r2_after = r2_score(st.session_state.y_test, y_pred_after)
                                mse_after = mean_squared_error(st.session_state.y_test, y_pred_after)
                                mae_after = mean_absolute_error(st.session_state.y_test, y_pred_after)
                                rmse_after = np.sqrt(mse_after)

                        # Display the metrics comparison
                        if st.session_state.task_type == "Classification":
                            st.write(f"### Comparison of Original and Tuned Model Metrics - {best_model_name}")
                            st.write(f"**Original Model:**")
                            st.write(f"- Accuracy: {accuracy_before:.4f}")
                            st.write(f"- Precision: {precision_before:.4f}")
                            st.write(f"- Recall: {recall_before:.4f}")
                            st.write(f"- F1 Score: {f1_before:.4f}")

                            st.write(f"**Tuned Model:**")
                            st.write(f"- Accuracy: {accuracy_after:.4f}")
                            st.write(f"- Precision: {precision_after:.4f}")
                            st.write(f"- Recall: {recall_after:.4f}")
                            st.write(f"- F1 Score: {f1_after:.4f}")

                        elif st.session_state.task_type == "Regression":
                            st.write(f"### Comparison of Original and Tuned Model Metrics - {best_model_name}")
                            st.write(f"**Original Model:**")
                            st.write(f"- R²: {r2_before:.4f}")
                            st.write(f"- MSE: {mse_before:.4f}")
                            st.write(f"- MAE: {mae_before:.4f}")
                            st.write(f"- RMSE: {rmse_before:.4f}")

                            st.write(f"**Tuned Model:**")
                            st.write(f"- R²: {r2_after:.4f}")
                            st.write(f"- MSE: {mse_after:.4f}")
                            st.write(f"- MAE: {mae_after:.4f}")
                            st.write(f"- RMSE: {rmse_after:.4f}")

                        # Section to download the tuned model
                        if st.session_state.tuned_model is not None:
                            st.write("### Download Tuned Model")
                            tuned_model_buffer = io.BytesIO()
                            pickle.dump(st.session_state.tuned_model, tuned_model_buffer)
                            tuned_model_buffer.seek(0)
                            st.download_button(
                                label="📥 Download Tuned Model",
                                data=tuned_model_buffer,
                                file_name=f"{best_model_name}_tuned_model.pkl",
                                mime="application/octet-stream",
                            )

                    except Exception as e:
                        st.error(f"🚨 Error during hyperparameter tuning: {e}")


if st.session_state.step == 11:
    st.title("🚀 Step 11: Prediction")
    st.write("In this final step, you can test your trained model by providing inputs for the features you selected earlier. The model will use these inputs to generate predictions.")

    st.subheader("📌 How to Use the Prediction Form")
    st.write("""
    - **Enter Input Values**: Provide values for each feature based on the columns you selected in Step 2.  
    - **Generate Prediction**: Once you've entered all required inputs, click the "Predict" button to see the model's output.  
    - **Use Cases**:  
      - Test the model with real-world scenarios.  
      - Validate the model's behavior with edge cases or hypothetical data.  
      - Understand how changes in feature values impact predictions.  
    - **Example**: If your model predicts house prices, you might enter details like square footage, number of bedrooms, and location to get a predicted price.
    """)
    # Step 11: Prediction on new data
    if "best_model" in st.session_state and st.session_state.best_model is not None:
        best_model_name = st.session_state.get("best_model", None)
        best_model_instance = st.session_state.models.get(best_model_name, None)

        if best_model_instance:
            # Retrieve the processed data and target
            features = st.session_state.features1  # Features columns
            target = st.session_state.target1      # Target column

            # Description to explain the source of input features
            st.write(f"### Input Data for Prediction using **{best_model_name}**")
            st.write(f"#### Feature Columns:")
            st.write("The features used for prediction are derived from your dataset (preprocessed and cleaned).")
            st.write("Below, you can input the values for the features, which will be used to predict the target variable.")

            # Initialize an empty dictionary to hold the user inputs for the prediction
            user_input_data = {}

            # Loop through each feature column and create input fields
            for feature in features:
                if feature in st.session_state.filtered_data2.columns:
                    # Check if the feature is categorical or numerical
                    if st.session_state.filtered_data2[feature].dtype == 'object':
                        # Categorical column: Show unique categories
                        unique_values = st.session_state.filtered_data2[feature].unique()
                        user_input_data[feature] = st.selectbox(
                            f"Select {feature} value", unique_values
                        )
                    else:
                        # Numerical column: Ask user to input a numerical value
                        user_input_data[feature] = st.number_input(
                            f"Enter value for {feature}", 
                            min_value=float(st.session_state.filtered_data2[feature].min()), 
                            max_value=float(st.session_state.filtered_data2[feature].max()),
                            value=float(st.session_state.filtered_data2[feature].mean())
                        )

            # When the user clicks the predict button
            if st.button("Predict"):
                # Create a DataFrame from user input
                input_data_df = pd.DataFrame([user_input_data])

                # Encoding categorical columns (if any)
                for feature in features:
                    if feature in st.session_state.filtered_data2.columns:
                        if st.session_state.filtered_data2[feature].dtype == 'object':
                            # Check if LabelEncoder is in session state
                            if f"{feature}_encoder" in st.session_state:
                                encoder = st.session_state[f"{feature}_encoder"]
                                input_data_df[feature] = encoder.transform(input_data_df[feature])
                            else:
                                # If no encoder exists, create a new one and fit it
                                encoder = LabelEncoder()
                                encoder.fit(st.session_state.filtered_data2[feature])
                                st.session_state[f"{feature}_encoder"] = encoder
                                input_data_df[feature] = encoder.transform(input_data_df[feature])

                # Scaling numerical features (if any)
                if "scaler" in st.session_state:
                    input_data_scaled = st.session_state.scaler.transform(input_data_df)
                else:
                    # If no scaler exists, create a new one and fit it
                    scaler = StandardScaler()
                    scaler.fit(st.session_state.filtered_data2[features])
                    st.session_state.scaler = scaler
                    input_data_scaled = scaler.transform(input_data_df)

                # Predict using the best model
                try:
                    prediction = best_model_instance.predict(input_data_scaled)
                    st.write(f"### Predicted {target}: {prediction[0]}🎉")
                except Exception as e:
                    st.error(f"🚨 Error during prediction: {e}")
