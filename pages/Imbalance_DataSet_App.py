import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import io
import numpy as np
import time
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.utils import resample  # Import resample for undersampling
from imblearn.over_sampling import SMOTE  # Import SMOTE for synthetic oversampling


# Set the page configuration
st.set_page_config(
    page_title="Handle Imbalance DATASET APP",  # Page title for browser tab
    page_icon="🛠️",                      # Custom favicon
    layout="wide"                        # Use wide layout for the page
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

# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = 0  # Default step to 0 (Home page)
# Dataset management
if "data4" not in st.session_state:
    st.session_state.data4 = None  # Original uploaded dataset
if "balanced_data" not in st.session_state:
    st.session_state.balanced_data = None  # Balanced dataset
# Target variable
if "target" not in st.session_state:
    st.session_state.target = None  # User-selected target variable
# Class distribution and imbalance handling
if "class_distribution" not in st.session_state:
    st.session_state.class_distribution = None  # Class distribution of the target
if "imbalance_method" not in st.session_state:
    st.session_state.imbalance_method = None  # Selected method for handling imbalance

# Function to navigate between steps
def navigate(step):
    st.session_state.step = step

steps = [
    "🏠 Home",
    "📂 Upload and Preview Dataset",
    "🎯 Analyze Class Imbalance",
    "🔍 Handling Duplicate Data",
    "🛠️ Handling Missing Data",
    "⚖️ Apply Balancing Techniques"
    
]



for idx, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"nav_{idx}"):
        navigate(idx)



# Define the Home Page
def home_page():
    

    # App title and introduction
    st.title("⚖️ Welcome to Handle Imbalanced Data App!")
    st.markdown("""
    **Handle Imbalanced Data App** is designed to help you detect and resolve class imbalance in your datasets 
    effectively, ensuring better performance of machine learning models.

    Whether you're a data scientist, analyst, or beginner, this app simplifies the process of balancing datasets 
    using state-of-the-art techniques like SMOTE, undersampling, and more.
    """)

    # Key features
    st.subheader("✨ Key Features:")
    st.markdown("""
    - 📂 **Upload Dataset**: Easily upload your dataset in CSV or Excel format.
    - 📊 **Analyze Class Imbalance**: Visualize the distribution of your target variable.
    - 🔄 **Apply Balancing Techniques**: Use methods like oversampling (e.g., SMOTE), undersampling, or hybrid approaches.
    - 📥 **Download Balanced Dataset**: Export the processed, balanced dataset for further use.
    """)

    # How it works
    st.subheader("⚙️ How It Works:")
    st.markdown("""
    1. **Upload Your Dataset**: Start by uploading your dataset in CSV or Excel format.
    2. **Analyze the Target Variable**: Choose your target variable and visualize its distribution.
    3. **Choose a Balancing Method**: Apply one of the imbalance-handling techniques to balance the dataset.
    4. **Download the Processed Dataset**: Export the balanced dataset for further analysis or model training.
    """)

    # Motivational conclusion
    st.markdown("---")
    st.markdown("🚀 **Ready to balance your data? Navigate to the first step using the sidebar!**")
    st.balloons()

# Step 0: Home
if st.session_state.step == 0:
    home_page()


# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = 1


# Step 1: Upload and Preview Dataset
if st.session_state.step == 1:
    st.title("📂 Step 1: Upload and Preview Dataset")
    st.markdown("Upload your dataset in CSV or Excel format.")
    st.markdown("After uploading, you'll select the target variable.")
    st.markdown("Ensure that the target variable is **categorical** to analyze the imbalanced dataset.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset:", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Load dataset
            if uploaded_file.name.endswith(".csv"):
                st.session_state.data4 = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                st.session_state.data4 = pd.read_excel(uploaded_file)

            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(st.session_state.data4.head())

            # Ask user to select target variable
            st.markdown(
                "<h2 style='color: blue;'> Select Target Variable:</h2>", 
            unsafe_allow_html=True
                )
            target_variable = st.selectbox("Choose the target variable:", options=st.session_state.data4.columns)

            # Check if selected column is categorical or numeric with limited categories
            if target_variable:
                target_dtype = st.session_state.data4[target_variable].dtype

                if target_dtype in ["object", "category"]:
                    st.session_state.target = target_variable
                    st.success(f"Target variable selected: **{target_variable}** (Categorical)")
                elif target_dtype in ["int64", "float64"]:
                    # Check if the numeric column has limited distinct values
                    unique_values = st.session_state.data4[target_variable].nunique()
                    if unique_values <= 10:  # You can adjust this threshold as needed
                        st.session_state.target = target_variable
                        st.success(f"✅ Target variable selected: **{target_variable}** (Categorical-like numeric values)")
                    else:
                        st.error(f"🚨 The selected target variable **{target_variable}** has too many unique values for a categorical task. Please choose a categorical column.")
                else:
                    st.error(f"🚨 The selected target variable **{target_variable}** is not categorical. Please choose a categorical column.")
                    
        except Exception as e:
            st.error(f" 🚨 Error loading dataset: {e}")
    else:
        st.warning("⚠️ Please upload a dataset to proceed.")




# Step 2: Analyze Class Imbalance
if st.session_state.step == 2:
    st.title("🎯 Step 2: Analyze Class Imbalance")
    st.markdown("""
    This step helps you analyze the distribution of classes in your target variable. Understanding the class distribution is crucial because imbalances in the dataset can lead to biased models that perform poorly on underrepresented classes.

    ### Key Points:
    - **Class Imbalance**: Class imbalance occurs when one class in the target variable is significantly underrepresented compared to others. For example, in a dataset of customer churn, if only 5% of customers have churned (target = 1), and 95% have not (target = 0), this is a class imbalance issue.
    - **Impact of Imbalance**: Imbalance can lead to the model being biased towards the majority class, often predicting the majority class for all instances and ignoring the minority class. This can result in poor predictive performance for the underrepresented class.
    - **Visualizing Class Distribution**: It's essential to visualize the distribution of the target variable (e.g., using bar plots or histograms) to detect imbalance. This helps to decide whether you need to apply techniques like **resampling**, **class weight adjustment**, or **synthetic data generation** (SMOTE) to balance the classes.

    **Use Case**: Analyzing the class distribution is vital in classification tasks, especially when the target variable has a skewed distribution. Identifying class imbalance early in the process enables you to apply appropriate techniques to improve the model’s fairness and accuracy across all classes.
    """)


    if st.session_state.data4 is not None and st.session_state.target is not None:
        target_column = st.session_state.target
        class_counts = st.session_state.data4[target_column].value_counts()

        # Display distribution chart
        st.write(f"### Distribution of Target Variable: **{target_column}**")
        st.bar_chart(class_counts)

        # Detailed class distribution
        total_samples = len(st.session_state.data4)
        class_percentages = class_counts / total_samples * 100

        st.markdown("### Class Distribution Details")
        # Combine counts and percentages in a readable format
        class_distribution_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage": class_percentages.values
        })

        # Display the class distribution as a table
        st.dataframe(class_distribution_df)

        # Highlight potential issues with imbalances
        st.markdown("### Insights")
        st.write(f"Total samples: **{total_samples}**")
        st.write("In an ideal classification task, we'd expect a roughly equal distribution of classes. If you notice a significant skew towards one or a few classes, this indicates an imbalance.")
        
        # Identify major imbalance (if any)
        max_class_count = class_counts.max()
        min_class_count = class_counts.min()

        imbalance_ratio = max_class_count / min_class_count

        if imbalance_ratio > 2:  # If imbalance ratio exceeds 2:1, it's considered imbalanced
            st.warning(f"⚠️ There is a significant class imbalance in **{target_column}**. The largest class has more than twice the samples of the smallest class (imbalance ratio: {imbalance_ratio:.2f}).")
        else:
            st.success(f"✅ The class distribution seems relatively balanced (imbalance ratio: {imbalance_ratio:.2f}).")

    else:
        st.error("🚨 Please complete Step 1 to upload and select the target variable.")


# Step 3: Handling Missing Data (Updated Step Number)
if st.session_state.step == 3:
    st.title("🔍 Step 3: Handling Duplicate Data")
    st.markdown("""
    In this step, we will check for any duplicate rows in your dataset and provide you with the option to remove them.
    Duplicate data can negatively impact your model's performance, so it's important to ensure that your dataset is clean.
    """)

    if st.session_state.data4 is not None:
        # Check for duplicate rows
        duplicate_rows = st.session_state.data4.duplicated().sum()
        st.write(f"### Number of Duplicate Rows: {duplicate_rows}")

        if duplicate_rows > 0:
            st.warning("""
            ⚠️ There are duplicate rows in your dataset. Duplicate data can affect the performance of your model.
            You can choose to remove the duplicates to ensure a cleaner dataset.
            """)

            # Button to remove duplicates
            if st.button("Remove Duplicates"):
                # Remove duplicates
                st.session_state.data4 = st.session_state.data4.drop_duplicates()
                st.success("✅ Duplicate rows have been removed successfully.")
                st.write("### Updated Dataset")
                

        else:
            st.success("✅ No duplicate rows found in the dataset.")

    else:
        st.error("🚨 Please complete Step 1 to upload the dataset.")

# Step 4: Handling Missing Data
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
    if st.session_state.data4 is not None:
        # Calculate the number and percentage of missing values for each column
        missing_count = st.session_state.data4.isnull().sum()
        missing_percentage = (missing_count / len(st.session_state.data4)) * 100
        missing_info = pd.DataFrame({"Missing Count": missing_count, "Missing Percentage": missing_percentage})
        
        st.write("### Missing Data Summary")
        st.dataframe(missing_info)

        # Initialize or update session state for missing data handling strategies
        if 'missing_handling' not in st.session_state:
            st.session_state.missing_handling = {}

        # Provide the option to handle missing data for each column
        for col in st.session_state.data4.columns:
            if missing_count[col] > 0:
                st.write(f"#### Column: {col}")

                # Show strategies based on column type
                if st.session_state.data4[col].dtype in [np.float64, np.int64]:  # Numerical columns
                    st.write(f"Missing data in this numerical column. Options: Fill with Mean, Median.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )

                    strategy = st.selectbox(f"{col}:", 
                                            ["None", "Fill with mean", "Fill with median", "Drop rows"],
                                            key=f"missing_strategy_{col}",
                                            index=["None", "Fill with mean", "Fill with median", "Drop rows"].index(st.session_state.missing_handling.get(col, "None")))
                else:  # Categorical columns
                    st.write(f"Missing data in this categorical column. Option: Fill with Mode.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                    strategy = st.selectbox(f"{col}:",
                                            ["None", "Fill with mode", "Drop rows"],
                                            key=f"missing_strategy_{col}",
                                            index=["None", "Fill with mode", "Drop rows"].index(st.session_state.missing_handling.get(col, "None")))

                # Save the selected strategy
                st.session_state.missing_handling[col] = strategy

                # Display mean/median/mode for numerical or categorical columns
                if st.session_state.data4[col].dtype in [np.float64, np.int64]:  # Numerical columns
                    st.write(f"**Mean**: {st.session_state.data4[col].mean():.2f}, **Median**: {st.session_state.data4[col].median():.2f}")
                else:  # Categorical columns
                    st.write(f"**Mode**: {st.session_state.data4[col].mode().iloc[0]}")

        # Handle missing data when the button is pressed
        if st.button("Handle Missing Data", key="handle_missing"):
            # Start a timer to calculate the time taken for handling missing data
            start_time = time.time()

            # Show a loading spinner while handling the missing data
            with st.spinner("Handling missing data, please wait..."):
                # Simulate a slight delay for the operation (replace this with the actual processing)
                time.sleep(1)  # Simulate processing delay

                # Show progress bar while handling missing data
                progress_bar = st.progress(0)
                for i, (col, strategy) in enumerate(st.session_state.missing_handling.items()):
                    if strategy == "Drop rows":
                        st.session_state.data4 = st.session_state.data4.dropna(subset=[col])
                    elif strategy == "Fill with mean":
                        st.session_state.data4[col] = st.session_state.data4[col].fillna(st.session_state.data4[col].mean())
                    elif strategy == "Fill with median":
                        st.session_state.data4[col] = st.session_state.data4[col].fillna(st.session_state.data4[col].median())
                    elif strategy == "Fill with mode":
                        st.session_state.data4[col] = st.session_state.data4[col].fillna(st.session_state.data4[col].mode().iloc[0])

                    # Simulate some progress during processing
                    time.sleep(0.1)  # Simulating progress for each column
                    progress_bar.progress(int((i + 1) / len(st.session_state.missing_handling) * 100))

                # Calculate the elapsed time and display the result
                elapsed_time = time.time() - start_time
                st.success(f"✅ Missing data handled in {elapsed_time:.2f} seconds! 🎉")
                

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")



# Step 5: Apply Balancing Techniques



if st.session_state.step == 5:
    st.title("⚖️ Step 5: Apply Balancing Techniques")
    
    # Short description of each balancing technique
    st.markdown("""
    In this step, we will apply balancing techniques to address class imbalance. 
    Here’s a brief guide on when to use each technique:

    - **Random Oversampling**: 
      - **Use Case**: When the minority class is underrepresented, random oversampling duplicates instances from the minority class to increase its representation. 
      - **Scenario**: If your target variable has a small percentage of positive cases (e.g., fraud detection), this technique can help the model learn better patterns for the minority class.
      - **Benefit**: Improves the model's ability to learn the minority class patterns, but can lead to overfitting if not handled carefully.

    - **Random Undersampling**: 
      - **Use Case**: When the majority class is overrepresented, random undersampling removes instances from the majority class to balance the dataset.
      - **Scenario**: If your dataset has many more negative cases than positive ones (e.g., in sentiment analysis where most reviews are positive), this technique can be helpful to prevent the model from being biased towards the majority class.
      - **Benefit**: Reduces model training time by reducing the number of majority class instances. However, important information may be lost by discarding instances.

    - **SMOTE (Synthetic Minority Over-sampling Technique)**: 
      - **Use Case**: When oversampling the minority class is needed without simply duplicating existing instances, SMOTE generates synthetic instances by interpolating between minority class instances.
      - **Scenario**: If you have a highly imbalanced dataset (e.g., rare disease diagnosis), SMOTE can help create synthetic data points that provide more information to the model without overfitting.
      - **Benefit**: Balances the dataset by creating new, unique instances rather than repeating existing ones. However, care must be taken to avoid introducing noise or outliers.

    Choose one of these techniques to proceed based on your dataset's imbalance and the specific problem you are working on.
    """)
    
    st.markdown(
    "<h2 style='color: blue;'>Select Balancing Method:</h2>", 
    unsafe_allow_html=True
        )


    # Ask user to select a balancing method
    balancing_method = st.selectbox("Balancing Methods:", 
                                    ["None", "Random Oversampling", "Random Undersampling", "SMOTE"])

    if balancing_method != "None":
        st.session_state.balancing_method = balancing_method

        if st.button("Apply Balancing Method"):
            # Apply the selected balancing technique
            with st.spinner(f"Applying {balancing_method}..."):
                # Create a copy of the dataset to avoid modifying the original data
                balanced_data = st.session_state.data4.copy()

                if balancing_method == "Random Oversampling":
                    # Apply Random Oversampling
                    majority_class = balanced_data[balanced_data[st.session_state.target] == balanced_data[st.session_state.target].mode()[0]]
                    minority_class = balanced_data[balanced_data[st.session_state.target] != balanced_data[st.session_state.target].mode()[0]]
                    minority_oversampled = resample(minority_class, 
                                                    replace=True, 
                                                    n_samples=majority_class.shape[0], 
                                                    random_state=123)
                    balanced_data = pd.concat([majority_class, minority_oversampled])

                elif balancing_method == "Random Undersampling":
                    # Apply Random Undersampling
                    majority_class = balanced_data[balanced_data[st.session_state.target] == balanced_data[st.session_state.target].mode()[0]]
                    minority_class = balanced_data[balanced_data[st.session_state.target] != balanced_data[st.session_state.target].mode()[0]]
                    majority_undersampled = resample(majority_class, 
                                                      replace=False, 
                                                      n_samples=minority_class.shape[0], 
                                                      random_state=123)
                    balanced_data = pd.concat([majority_undersampled, minority_class])

                elif balancing_method == "SMOTE":
                    # Apply SMOTE (Synthetic Minority Over-sampling Technique)
                    X = balanced_data.drop(columns=[st.session_state.target])
                    y = balanced_data[st.session_state.target]
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                    
                st.success(f"✅ {balancing_method} applied successfully!")

                # Display updated class distribution
                class_counts = balanced_data[st.session_state.target].value_counts()
                total_samples = len(balanced_data)
                class_percentages = (class_counts / total_samples) * 100
                st.write(f"### Updated Distribution of Target Variable: **{st.session_state.target}**")
                st.bar_chart(class_counts)

                # Show updated class distribution in a dataframe
                class_distribution_df = pd.DataFrame({
                    "Class": class_counts.index,
                    "Count": class_counts.values,
                    "Percentage": class_percentages.values
                })
                st.write(class_distribution_df)

                # Button to download the balanced dataset
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(balanced_data)
                st.download_button(
                    label="Download Balanced Dataset",
                    data=csv,
                    file_name="balanced_dataset.csv",
                    mime="text/csv"
                )

    else:
        st.warning("🚨 Please select a balancing method to apply.")
