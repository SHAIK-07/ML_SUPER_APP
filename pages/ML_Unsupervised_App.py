import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


# Set the page configuration
st.set_page_config(
    page_title="ML UnSupervised APP",  # Page title for browser tab
    page_icon="🚀",                      # Custom favicon
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
# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = 0  # Set the default step to 0 (Home page)
if "data5" not in st.session_state:
    st.session_state.data5 = None  # Store the uploaded dataset
if "X_train" not in st.session_state:
    st.session_state.X_train = None  # Features (numerical columns for clustering)
if "model" not in st.session_state:
    st.session_state.model = None  # Store the clustering model (e.g., KMeans, DBSCAN)
if "features" not in st.session_state:
    st.session_state.features = None  # Store the selected feature columns for clustering
if "missing_handling" not in st.session_state:
    st.session_state.missing_handling = {}  # Store the method used for handling missing data
if "outlier_handling" not in st.session_state:
    st.session_state.outlier_handling = {}  # Store method for handling outliers
if "scaling_methods" not in st.session_state:
    st.session_state.scaling_methods = {}  # Store the scaling method used (e.g., StandardScaler)
if "encoding_methods" not in st.session_state:
    st.session_state.encoding_methods = {}  # Store the encoding method for categorical data (if any)


# Function to navigate between steps
def navigate(step):
    st.session_state.step = step

# Steps
steps = [
    "🏠 Home",
    "📂 Upload and Preview Dataset",
    "🔍 Handling Duplicate Data",
    "🛠️ Handling Missing Data",
    "📊 Handling Outliers",
    "📏 Feature Scaling",
    "🔄 Encoding",
    "🤖 Train the Clustering Model",
    "📊 Visualization and Download"
]


for idx, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"nav_{idx}"):
        navigate(idx)

# Define the Home Page
def home_page():
    

    st.title("🚀 Welcome to ML UnSupervised APP!")
    st.markdown("""
    **ML UnSupervised APP** is designed to simplify the machine learning process for clustering tasks. 
    It automates key steps such as data preprocessing, model training, and evaluation, enabling users to 
    efficiently apply clustering algorithms without deep technical expertise.
    """)

    st.subheader("✨ Key Features:")
    st.markdown("""
    - 📂 **Upload Data**: Start by uploading your dataset in CSV or Excel format.
    - 🔍 **Data Cleaning**: Handle duplicates, missing values, and outliers effortlessly.
    - 📊 **Data Scaling & Encoding**: Automatically prepare data for clustering algorithms.
    - 🤖 **Model Training**: Train clustering models like KMeans and DBSCAN.
    - 📈 **Visualization**: Visualize clusters with interactive plots.
    - 📥 **Download Results**: Export cleaned data or clustering results.
    """)

    st.subheader("⚙️ How It Works:")
    st.markdown("""
    1. **Upload Your Dataset**: Begin by uploading a CSV or Excel file.
    2. **Preprocess the Data**: Clean and scale the data before training the model.
    3. **Train Clustering Model**: Choose from various clustering algorithms like KMeans, DBSCAN.
    4. **Visualize Clusters**: View the clustering results through interactive visualizations.
    5. **Download Results**: Export the final dataset or clustering output for further analysis.
    """)

    st.markdown("---")
    st.markdown("🚀 **Get started now by navigating to the first step from the sidebar!**")
    st.balloons()

# Step 0: Home
if st.session_state.step == 0:
    home_page()

# Initialize session state step if not already set
if 'step' not in st.session_state:
    st.session_state.step = 1


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
                    st.session_state.data5 = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    st.session_state.data5 = pd.read_excel(uploaded_file)
                else:
                    st.error("🚨 Unsupported file format. Please upload a CSV or Excel file.")
                    st.stop()

                # Show a preview of the first 5 rows of the dataset
                st.write("### Dataset Preview")
                st.dataframe(st.session_state.data5.head())

                # Button to trigger detailed EDA
                if st.button("Show Detailed EDA Report"):
                    # Generate profiling report for in-app display using YData Profiling
                    profile_in_app = ProfileReport(
                        st.session_state.data5,
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

                    # Create a download button for the profiling report
                   

                    # Display the elapsed time for generating the report
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Report generated in {elapsed_time:.2f} seconds! 🎉")
            except Exception as e:
                st.error(f"🚨 An error occurred while processing the file: {e}")


# Step 2: Handling Duplicate Data (Updated Step Number)
if st.session_state.step == 2:
    st.title("🔍 Step 2: Handling Duplicate Data")
    st.write("Duplicate data can lead to biased analysis, inaccurate models, and inefficient processing, so it's important to remove them.")

    if st.session_state.data5 is not None:
        # Show the number of duplicate rows in the dataset
        duplicate_count = st.session_state.data5.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")

        if duplicate_count > 0:
            # Option to remove duplicates
            if st.button("Remove Duplicates", key="remove_duplicates"):
                # Start a timer for estimated loading time
                start_time = time.time()

                # Show a loading spinner while removing duplicates
                with st.spinner("Removing duplicates, please wait..."):
                    # Remove duplicates
                    st.session_state.data5 = st.session_state.data5.drop_duplicates()

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Duplicates removed in {elapsed_time:.2f} seconds! 🎉")

                    
        else:
            st.write("✅ No duplicate rows found in the dataset.")

        

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


# Step 3: Handling Missing Data (Updated Step Number)
if st.session_state.step == 3:
    st.title("🛠️ Step 3: Handling Missing Data")
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
    if st.session_state.data5 is not None:
        # Calculate the number and percentage of missing values for each column
        missing_count = st.session_state.data5.isnull().sum()
        missing_percentage = (missing_count / len(st.session_state.data5)) * 100
        missing_info = pd.DataFrame({"Missing Count": missing_count, "Missing Percentage": missing_percentage})
        st.write("### Missing Data Summary")
        st.dataframe(missing_info)

        # Provide the option to handle missing data for each column
        for col in st.session_state.data5.columns:
            if missing_count[col] > 0:
                st.write(f"#### Column: {col}")

                # Show strategies based on column type
                if st.session_state.data5[col].dtype in [np.float64, np.int64]:  # Numerical columns
                    st.write(f"Missing data in this numerical column. Options: Fill with Mean, Median.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                    strategy = st.selectbox(f"{col}:", ["None", "Fill with mean", "Fill with median", "Drop rows"],
                                            key=f"missing_strategy_{col}",
                                            index=["None", "Fill with mean", "Fill with median", "Drop rows"].index(st.session_state.missing_handling.get(col, "None")))
                else:  # Categorical columns
                    st.write(f"Missing data in this categorical column. Option: Fill with Mode.")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                    strategy = st.selectbox(f"{col}:", ["None", "Fill with mode", "Drop rows"],
                                            key=f"missing_strategy_{col}",
                                            index=["None", "Fill with mode", "Drop rows"].index(st.session_state.missing_handling.get(col, "None")))

                st.session_state.missing_handling[col] = strategy

                # Display mean/median/mode for numerical or categorical columns
                if st.session_state.data5[col].dtype in [np.float64, np.int64]:  # Numerical columns
                    st.write(f"**Mean**: {st.session_state.data5[col].mean():.2f}, **Median**: {st.session_state.data5[col].median():.2f}")
                else:  # Categorical columns
                    st.write(f"**Mode**: {st.session_state.data5[col].mode().iloc[0]}")

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
                        st.session_state.data5 = st.session_state.data5.dropna(subset=[col])
                    elif strategy == "Fill with mean":
                        st.session_state.data5[col] = st.session_state.data5[col].fillna(st.session_state.data5[col].mean())
                    elif strategy == "Fill with median":
                        st.session_state.data5[col] = st.session_state.data5[col].fillna(st.session_state.data5[col].median())
                    elif strategy == "Fill with mode":
                        st.session_state.data5[col] = st.session_state.data5[col].fillna(st.session_state.data5[col].mode().iloc[0])

                    # Simulate some progress during processing
                    time.sleep(0.1)  # Simulating progress for each column
                    progress_bar.progress(int((i + 1) / len(st.session_state.missing_handling) * 100))

                # Calculate the elapsed time and display the result
                elapsed_time = time.time() - start_time
                st.success(f"✅ Missing data handled in {elapsed_time:.2f} seconds! 🎉")

                

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


# Step 5: Handling Outliers
if st.session_state.step == 4:
    st.title("📊 Step 4: Handling Outliers")
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

    if st.session_state.data5 is not None:
        # Ensure outlier_handling is initialized
        if "outlier_handling" not in st.session_state:
            st.session_state.outlier_handling = {}

        # Use selected feature columns instead of all numerical columns
        numerical_cols = [col for col in st.session_state.features if st.session_state.data5[col].dtype in ["float", "int"]]

        # Identify columns with outliers using the IQR method
        outlier_columns = []
        for col in numerical_cols:
            Q1 = st.session_state.data5[col].quantile(0.25)
            Q3 = st.session_state.data5[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Check if there are any outliers
            if any(st.session_state.data5[col] < lower_bound) or any(st.session_state.data5[col] > upper_bound):
                outlier_columns.append(col)

        if outlier_columns:
            st.write("Columns with outliers detected:")
            for col in outlier_columns:
                st.write(f"**{col}**")
                # Plotting the boxplot for columns with outliers
                fig, ax = plt.subplots(figsize=(10, 1))  # Create a figure and axis object
                sns.boxplot(x=st.session_state.data5[col], ax=ax, color='lightblue', flierprops=dict(marker='o', color='red', markersize=5))
                st.pyplot(fig)  # Pass the figure to st.pyplot()

                # Select method for handling outliers
                st.markdown(
                    f"<h3 style='color: blue;'>Select method to handle outliers in {col}:</h3>", 
                        unsafe_allow_html=True
                        )
                method = st.selectbox(f"{col}:",
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

                        Q1 = st.session_state.data5[col].quantile(0.25)
                        Q3 = st.session_state.data5[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        if method == "IQR":
                            st.session_state.filtered_data = st.session_state.filtered_data[(st.session_state.filtered_data[col] >= lower_bound) & (st.session_state.filtered_data[col] <= upper_bound)]
                        elif method == "Cap outliers":
                            st.session_state.filtered_data[col] = np.clip(st.session_state.filtered_data[col], lower_bound, upper_bound)
                        elif method == "Remove outliers":
                            st.session_state.filtered_data = st.session_state.filtered_data[(st.session_state.filtered_data[col] >= lower_bound) & (st.session_state.filtered_data[col] <= upper_bound)]

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
                        sns.boxplot(x=st.session_state.filtered_data[col], ax=ax, color='lightgreen', flierprops=dict(marker='o', color='red', markersize=5))
                        st.pyplot(fig)  # Pass the figure to st.pyplot()

                    # Show the updated data after handling outliers
                    st.write("### Updated Dataset (after handling outliers):")
                    st.dataframe(st.session_state.filtered_data)

        else:
            st.write("✅ No outliers detected in the dataset. 🎉")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")



if st.session_state.step == 5:
    st.title("📏 Step 5: Feature Scaling")
    st.write("Feature scaling ensures that all features contribute equally to the model by standardizing their ranges, improving model performance and convergence.")

    if st.session_state.filtered_data is not None:
        # Ensure scaling_methods is initialized
        if "scaling_methods" not in st.session_state:
            st.session_state.scaling_methods = {}

        # Separate numerical and categorical columns based on updated data types
        numerical_cols = []
        categorical_cols = []

        for col in st.session_state.features:
            if st.session_state.filtered_data[col].dtype in ["float64", "int64"]:
                # Check if the column has strictly increasing or sequential values
                if np.all(np.diff(st.session_state.filtered_data[col].dropna().sort_values()) == 1):
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
        st.write(st.session_state.filtered_data[numerical_cols].head())

        # Proceed with scaling if there are numerical columns
        if len(numerical_cols) > 0:
            # Allow the user to select a scaling method for each numerical column
            for col in numerical_cols:
                st.markdown(
                    f"<h3 style='color: blue;'>Select scaling method for {col}:</h3>", 
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
                            st.session_state.filtered_data[col] = scaler.fit_transform(st.session_state.filtered_data[[col]])
                        elif method == "Min-Max Scaling":
                            scaler = MinMaxScaler()
                            st.session_state.filtered_data[col] = scaler.fit_transform(st.session_state.filtered_data[[col]])
                        elif method == "Robust Scaling":
                            scaler = RobustScaler()
                            st.session_state.filtered_data[col] = scaler.fit_transform(st.session_state.filtered_data[[col]])
                        elif method == "MaxAbs Scaling":
                            scaler = MaxAbsScaler()
                            st.session_state.filtered_data[col] = scaler.fit_transform(st.session_state.filtered_data[[col]])

                        # Update progress bar
                        time.sleep(0.5)  # Simulate processing time for each column
                        progress_bar.progress(int((idx + 1) / total_columns * 100))

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Feature scaling applied in {elapsed_time:.2f} seconds! 🎉")

                    # Show preview of filtered data after scaling
                    st.write("### Preview of Data After Scaling:")
                    st.write(st.session_state.filtered_data[numerical_cols].head())

        else:
            st.warning("✅ No numerical columns found in the dataset. You can skip this step.")
            st.write("Since there are no numerical columns, feature scaling is not necessary. You can proceed to the next step.")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


if st.session_state.step == 6:
    st.title("🔄 Step 6: Encoding")
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


    if st.session_state.filtered_data is not None:
        # Ensure encoding_methods is initialized
        if "encoding_methods" not in st.session_state:
            st.session_state.encoding_methods = {}

        # Use only selected categorical feature columns
        existing_columns = [col for col in st.session_state.features if col in st.session_state.filtered_data.columns]
        categorical_cols = [col for col in existing_columns if st.session_state.filtered_data[col].dtype in ["object", "category"]]

        if st.session_state.target in categorical_cols:
            # Automatically apply Label Encoding if the target is categorical
            le = LabelEncoder()
            st.session_state.filtered_data[st.session_state.target] = le.fit_transform(st.session_state.filtered_data[st.session_state.target])
            categorical_cols = [col for col in categorical_cols if col != st.session_state.target]  # Exclude target from encoding

            st.warning(f"🚨 The target variable '{st.session_state.target}' has been label-encoded.")

        if len(categorical_cols) == 0:
            st.write("There are no categorical columns to encode. You can skip this step.")
        else:
            # Show preview of filtered data before encoding
            st.write("### Preview of Data Before Encoding:")
            st.write(st.session_state.filtered_data[categorical_cols].head())

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
                    unique_categories = st.session_state.filtered_data[col].dropna().unique()
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
                            st.success(f"✅ Ordinal encoding applied to {col}. The encoded values are: {category_mapping}")
                        else:
                            st.warning("🚨 The number of categories in the order must match the number of unique categories in the column.")

            # Button to update the encoding methods
            if st.button("Update Encoding Methods", key="update_encoding"):
                st.success("✅ Encoding methods updated! Click 'Apply Encoding' to finalize.")

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
                            st.session_state.filtered_data[col] = le.fit_transform(st.session_state.filtered_data[col])
                        elif method == "One-Hot Encoding":
                            st.session_state.filtered_data = pd.get_dummies(st.session_state.filtered_data, columns=[col], drop_first=True)
                        elif method == "Target Encoding":
                            # Target encoding (mean of target per category)
                            target_mean = st.session_state.filtered_data.groupby(col)[st.session_state.target].mean()
                            st.session_state.filtered_data[col] = st.session_state.filtered_data[col].map(target_mean)
                        elif method == "Binary Encoding":
                            # Binary encoding (for categorical variables with high cardinality)
                            binary_encoded = st.session_state.filtered_data[col].apply(lambda x: format(int(x), 'b'))
                            st.session_state.filtered_data[col] = binary_encoded
                        elif isinstance(method, dict):  # For Ordinal Encoding
                            # Apply ordinal encoding using the saved category mapping
                            st.session_state.filtered_data[col] = st.session_state.filtered_data[col].map(method)

                        # Update progress bar
                        time.sleep(0.5)  # Simulate processing time for each column
                        progress_bar.progress(int((idx + 1) / total_columns * 100))

                    # Calculate the elapsed time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Encoding applied in {elapsed_time:.2f} seconds! 🎉")

                    # Show preview of filtered data after encoding
                    st.write("### Preview of Data After Encoding:")
                    st.write(st.session_state.filtered_data.head())

                    # Update the features list to reflect the new columns after encoding
                    st.session_state.features = [col for col in st.session_state.filtered_data.columns if col != st.session_state.target]

    else:
        st.warning("🚨 Please upload a dataset in Step 1.")

# Step 7: Clustering and Feature Extraction (PCA)
if st.session_state.step == 7:
    st.title("🤖 Step 7: Clustering and Feature Extraction (PCA)")
    st.write("""
        In this step, we perform unsupervised machine learning tasks:
        
        - **Clustering**: This technique groups similar data points together without labeled outcomes. It helps in identifying patterns and structures within the data.

            **Use Cases for Clustering Algorithms**:
            - **KMeans**:  
                - Use when you want to segment your data into distinct, non-overlapping groups.  
                - Commonly used in customer segmentation, market basket analysis, and image compression.
                
            - **DBSCAN**:  
                - Use when your data contains noise or outliers, and you want to detect clusters of arbitrary shape.  
                - Ideal for anomaly detection, geospatial data clustering, and image segmentation.
                
            - **Agglomerative Clustering**:  
                - Use when hierarchical relationships between clusters are important and you need a bottom-up approach.  
                - Suitable for document clustering, gene expression data analysis, and hierarchical taxonomies.

        - **Feature Extraction (PCA)**:  
            - Principal Component Analysis (PCA) reduces the number of features in the dataset by transforming them into a smaller set of orthogonal features, or principal components, which explain the most variance in the data. 
            - **Use Case for PCA**:  
                - Use when you have a high-dimensional dataset and want to reduce its complexity while preserving important information.  
                - Commonly used for noise reduction, data visualization, and improving the performance of algorithms like linear regression, SVM, and neural networks.

    """)


    if st.session_state.data5 is not None:
        # Scale the data before clustering and PCA
        numerical_cols = st.session_state.data5.select_dtypes(include=["float", "int"]).columns
        if len(numerical_cols) > 0:
            # Standard Scaling for clustering and PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(st.session_state.data5[numerical_cols])

            # User selection for Clustering or Feature Extraction
            task_type = st.radio("Select Task Type:", ["Clustering", "Feature Extraction (PCA)"], key="task_type")

            if task_type == "Clustering":
                # Clustering options
                st.markdown(
                "<h2 style='color: blue;'> Select Clustering Algorithm:</h2>", 
                unsafe_allow_html=True
                    )
                clustering_method = st.selectbox("Clustering Algorithms:", ["None","KMeans", "DBSCAN", "Agglomerative Clustering"], key="clustering_method")

                if clustering_method == "KMeans":
                    num_clusters = st.slider("Select number of clusters for KMeans:", 2, 10, 3, key="num_clusters")
                    max_iter = st.slider("Select maximum number of iterations for KMeans:", 100, 1000, 300, key="max_iter")
                    st.write("Parameters for KMeans:")
                    st.write("n_clusters: Number of clusters")
                    st.write("max_iter: Maximum number of iterations for the algorithm")

                elif clustering_method == "DBSCAN":
                    eps = st.slider("Select eps for DBSCAN:", 0.1, 5.0, 0.5, key="eps")
                    min_samples = st.slider("Select min_samples for DBSCAN:", 1, 10, 5, key="min_samples")
                    st.write("Parameters for DBSCAN:")
                    st.write("eps: Maximum distance between two samples")
                    st.write("min_samples: Minimum number of samples in a neighborhood")

                elif clustering_method == "Agglomerative Clustering":
                    num_clusters = st.slider("Select number of clusters for Agglomerative Clustering:", 2, 10, 3, key="agg_num_clusters")
                    st.markdown(
                    f"<h3 style='color: blue;'>Select affinity for Agglomerative Clustering::</h3>", 
                        unsafe_allow_html=True
                        )
                    affinity = st.selectbox("affinities for Agglomerative Clustering:", ["euclidean", "l1", "l2", "manhattan", "cosine"], key="affinity")
                    st.write("Parameters for Agglomerative Clustering:")
                    st.write("n_clusters: Number of clusters")
                    st.write("affinity: Metric used for the linkage")

                # Train the model button
                if st.button("Train Clustering Model", key="train_clustering"):
                    # Train the selected clustering model
                    if clustering_method == "KMeans":
                        kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter)
                        clusters = kmeans.fit_predict(scaled_data)
                        st.session_state.data5['Cluster'] = clusters  # Add the cluster labels to the dataset
                        st.success(f"✅ Clustering performed using KMeans with {num_clusters} clusters and max_iter={max_iter}.")

                    elif clustering_method == "DBSCAN":
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters = dbscan.fit_predict(scaled_data)
                        st.session_state.data5['Cluster'] = clusters  # Add the cluster labels to the dataset
                        st.success("✅ Clustering performed using DBSCAN.")

                    elif clustering_method == "Agglomerative Clustering":
                        agglomerative = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity)
                        clusters = agglomerative.fit_predict(scaled_data)
                        st.session_state.data5['Cluster'] = clusters  # Add the cluster labels to the dataset
                        st.success(f"✅ Clustering performed using Agglomerative Clustering with {num_clusters} clusters and affinity={affinity}.")

            elif task_type == "Feature Extraction (PCA)":
                # Feature Extraction using PCA
                st.markdown(
                    f"<h3 style='color: blue;'>Select strategy for Feature Extra:</h3>", 
                        unsafe_allow_html=True
                        )
                pca_option = st.selectbox("Currently we have only PCA", ["None", "PCA"], key="pca_option")
                if pca_option == "PCA":
                    st.write("Feature Extraction using PCA (Principal Component Analysis):")
                    st.write(f"Number of Columns in the dataset: {scaled_data.shape[1]}")
                    dimensions = st.slider("Select number of dimensions (components) for PCA:", 1, scaled_data.shape[1], 2, key="pca_dimensions")
                    
                    # Train the PCA model
                    if st.button("Train PCA Model", key="train_pca"):
                        pca = PCA(n_components=dimensions)
                        pca_result = pca.fit_transform(scaled_data)

                        # Show explained variance ratio
                        st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
                        st.write(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

                        # Add PCA components to the dataset
                        for i in range(dimensions):
                            st.session_state.data5[f'PCA{i+1}'] = pca_result[:, i]
                        st.success(f"✅ Feature Extraction using PCA with {dimensions} dimensions has been applied successfully.")
                else:
                    st.write("⚠️ You selected None. No PCA applied.")

            
        else:
            st.warning("🚨 No numerical columns found to perform clustering or PCA.")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")



import plotly.express as px
import pandas as pd

# Step 8: Visualization and Dataset Download

if st.session_state.step == 8:
    st.title("📊 Step 8: Visualization and Download")
    st.write("""
        In this step, we will visualize the results of the clustering and feature extraction (PCA).
        You can also download the updated dataset with clustering labels and PCA components.
    """)

    if st.session_state.data5 is not None:
        # Show a preview of the updated dataset
        st.write("Preview of the updated dataset:")
        st.dataframe(st.session_state.data5.head())  # Display the first few rows of the dataset

        # Check if clustering has been performed
        if 'Cluster' in st.session_state.data5.columns:
            st.write("Clustering Results Visualization:")

            # 2D Visualization of clusters based on the first two columns of the data
            fig = px.scatter(st.session_state.data5, x=st.session_state.data5.columns[0], y=st.session_state.data5.columns[1], 
                             color='Cluster', title="Clustering Visualization",
                             labels={st.session_state.data5.columns[0]: st.session_state.data5.columns[0], 
                                     st.session_state.data5.columns[1]: st.session_state.data5.columns[1]})
            st.plotly_chart(fig)

            # Option to download the updated dataset with cluster labels
            st.download_button(
                label="Download Dataset with Clusters",
                data=st.session_state.data5.to_csv(index=False).encode('utf-8'),
                file_name="dataset_with_clusters.csv",
                mime="text/csv"
            )

        # Check if PCA was applied
        elif 'PCA1' in st.session_state.data5.columns and 'PCA2' in st.session_state.data5.columns:
            st.write("PCA Results Visualization:")

            # Dropdown to select the type of PCA visualization
            pca_options = ['2D Visualization (PCA1 vs PCA2)', '3D Visualization (PCA1, PCA2, PCA3)', 'Pairwise Scatter Plots']
            selected_option = st.selectbox("Select the PCA visualization type:", pca_options)

            # Button to trigger PCA visualization
            if st.button("Visualize PCA"):
                if selected_option == '2D Visualization (PCA1 vs PCA2)':
                    # Show 2D visualization using PCA1 and PCA2
                    fig = px.scatter(st.session_state.data5, x='PCA1', y='PCA2', 
                                     title="PCA 2D Components Visualization",
                                     labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'})
                    st.plotly_chart(fig)

                elif selected_option == '3D Visualization (PCA1, PCA2, PCA3)':
                    # Show 3D visualization using PCA1, PCA2, and PCA3
                    if 'PCA3' in st.session_state.data5.columns:
                        fig = px.scatter_3d(st.session_state.data5, x='PCA1', y='PCA2', z='PCA3', 
                                            title="PCA 3D Components Visualization",
                                            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2', 'PCA3': 'Principal Component 3'})
                        st.plotly_chart(fig)
                    else:
                        st.warning("🚨 PCA3 is not available for 3D visualization. Please apply PCA with 3 components.")

                elif selected_option == 'Pairwise Scatter Plots':
                    # Generate pairwise scatter plots for all combinations of PCA components
                    pca_columns = [f'PCA{i+1}' for i in range(st.session_state.pca_components)]  # Dynamic PCA columns
                    from itertools import combinations
                    pairs = list(combinations(pca_columns, 2))

                    for pair in pairs:
                        fig = px.scatter(st.session_state.data5, x=pair[0], y=pair[1], 
                                         title=f"Scatter Plot of {pair[0]} vs {pair[1]}",
                                         labels={pair[0]: pair[0], pair[1]: pair[1]})
                        st.plotly_chart(fig)

            # Option to download the dataset with PCA components
            st.download_button(
                label="Download Dataset with PCA Components",
                data=st.session_state.data5.to_csv(index=False).encode('utf-8'),
                file_name="dataset_with_pca.csv",
                mime="text/csv"
            )

        else:
            st.warning("🚨 Clustering or PCA has not been applied yet. Please complete Step 7 first.")
    else:
        st.warning("🚨 Please upload a dataset in Step 1.")


# Finish button for model comparison
    if st.button("Finish", key="compare_finish"):
        st.success("✅ Model comparison completed!")
        st.balloons()
