import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Flatten, MaxPooling1D
from PIL import Image
import io

# ------------------- Streamlit App -------------------
# File upload widget moved to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# When no file is uploaded
if uploaded_file is None:
    st.title("Welcome to Water Quality Analysis")
    image = Image.open("water quality analysis.png")  # Ensure this image exists in the correct directory
    st.image(image, use_container_width=True)

# Sidebar for navigation
section = st.sidebar.radio("Select Section", ["Overview", "EDA", "Predictive Analysis", "Water Quality Index"])

# When file is uploaded
if uploaded_file is not None:
    try:
        # Convert the uploaded binary file to a string buffer
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        st.title("Water Quality Analysis")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

    # ------------------- DATA PREPROCESSING -------------------
    numeric_cols_to_convert = [
        'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
        'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
        'Sulfide', 'Carbon Dioxide', 'Air Temperature (0C)'
    ]

    # Convert columns to numeric and handle errors
    for col in numeric_cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str), format='%B %Y')
    df.sort_values('Date', inplace=True)

    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ------------------- SECTION 1: OVERVIEW -------------------
    if section == "Overview":
        st.header("Data Overview 💧")
        st.subheader("Dataset Summary")
        st.write(f"Total rows: {df.shape[0]}")
        st.write(f"Total columns: {df.shape[1]}")
        st.write("""
        The dataset consists of water quality data collected from multiple monitoring sites. 
        It includes a variety of environmental factors such as water temperature, pH levels, ammonia, nitrate, 
        phosphate concentrations, dissolved oxygen, and volcanic activity (sulfide and carbon dioxide). 
        Additionally, weather data such as wind direction and air temperature, along with the site's specific 
        information, are included. The dataset spans several months and years, allowing for a time-series analysis of water quality.
        """)
        st.subheader("Column Descriptions")
        st.write("""
        - **Date**: Date of data collection, which is a combination of the 'Year' and 'Month' columns.
        - **Water Quality Parameters**: This includes temperature (surface, middle, and bottom layers), pH, ammonia, nitrate, 
          phosphate, dissolved oxygen, sulfide, and carbon dioxide levels.
        - **Weather and Site Information**: Weather conditions (categorical) and wind direction (categorical), as well as site identifiers.
        - **Normalized Values**: All numerical columns have been scaled between 0 and 1 using MinMaxScaler to facilitate machine learning model training.
        """)
        st.subheader("Data Preprocessing 🛠️")
        st.write("""
        - **Missing Values**: Missing numeric values have been filled using the median value of the respective column to ensure continuity of the data.
        - **Duplicate Removal**: Duplicate records have been removed to ensure the dataset's quality and avoid redundancy.
        - **Normalization**: All numeric features have been normalized to a range of 0 to 1 to help improve model training performance.
                   """)

        with st.expander("Summary Statistics"):
            st.subheader("Summary Statistics 📊")
            st.write(df.select_dtypes(include=[float, int]).describe())

    # ------------------- SECTION 2: EDA -------------------
    elif section == "EDA":
        with st.expander("Temporal Analysis: Temperature Trends Over Time"):
            st.subheader("Temperature Trends Over Time 📅")
            plt.figure(figsize=(14, 6))
            for temp_col in ['Surface temp', 'Middle temp', 'Bottom temp']:
                if temp_col in df.columns:
                    plt.plot(df['Date'], df[temp_col], label=temp_col)
            plt.title('Temperature Trends Over Time 🌡️')
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

        with st.expander("Correlation Heatmap 🔥"):
            st.subheader("Feature Correlation Heatmap 🔍")
            corr_matrix = df[numeric_cols].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
            plt.title('Feature Correlation Heatmap 🧠')
            plt.tight_layout()
            st.pyplot(plt)

        with st.expander("Parameter Relationships 🤝"):
            st.subheader("Air vs Surface Temperature and Surface Temp vs Dissolved Oxygen 🌬️")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.scatterplot(x='Air Temperature (0C)', y='Surface temp', data=df, ax=axes[0])
            axes[0].set_title('Air vs Surface Temperature')
            sns.scatterplot(x='Surface temp', y='Dissolved Oxygen', data=df, ax=axes[1])
            axes[1].set_title('Surface Temp vs Dissolved Oxygen')
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Time-Series of Key Water Quality Parameters 📈"):
            st.subheader("Time-Series of Key Water Quality Parameters")
            fig, axes = plt.subplots(3, 2, figsize=(15, 10))
            params = ['Dissolved Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Carbon Dioxide']
            for i, param in enumerate(params):
                ax = axes[i//2, i%2]
                ax.plot(df['Date'], df[param])
                ax.set_title(param)
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Dissolved Oxygen Distribution by Site 🏞️"):
            st.subheader("Dissolved Oxygen Levels by Site 🌿")
            if 'Site' in df.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='Site', y='Dissolved Oxygen', data=df)
                plt.title('Dissolved Oxygen Levels by Site')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

        st.write("\nEDA Complete! ✅")

    # ------------------- SECTION 3: PREDICTIVE ANALYSIS -------------------
    elif section == "Predictive Analysis":
        st.header("Water Quality Prediction Using Deep Learning Models 🔮")

        # Sidebar controls
        epochs = st.sidebar.slider("Epochs", 1, 100, 10)
        batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64])

        # Preprocess categorical variables
        df['Weather Condition'] = df['Weather Condition'].astype('category').cat.codes
        df['Wind Direction'] = df['Wind Direction'].astype('category').cat.codes
        df['Site'] = df['Site'].astype('category').cat.codes

        # Identify columns available in the uploaded dataset
        all_possible_targets = ['pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide']
        available_targets = [col for col in all_possible_targets if col in df.columns]

        # Ask user to select targets from available ones (no default selected)
        target_cols = st.multiselect(
            "Select water quality parameters to predict:",
            options=available_targets,
            default=[]
        )

        if not target_cols:
            st.warning("⚠️ Please select at least one parameter from the list to proceed with prediction.")
            st.stop()

        # Set features only if all are present in the dataset
        required_features = ['Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature', 'pH',
                             'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide']
        water_cols = [col for col in required_features if col in df.columns]

        if len(water_cols) < len(required_features):
            missing = set(required_features) - set(water_cols)
            st.error(f"The following required input features are missing in your dataset: {', '.join(missing)}")
            st.stop()

        # Prepare input-output with prediction gap
        prediction_gap_weeks = 1
        X = df[water_cols].values[:-prediction_gap_weeks]
        Y = df[target_cols].values[prediction_gap_weeks:]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Build models
        def create_model(type, input_shape, output_units):
            model = Sequential()
            if type == 'cnn':
                model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
            elif type == 'lstm':
                model.add(LSTM(64, input_shape=input_shape, activation='relu'))
            elif type == 'cnn_lstm':
                model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(64, activation='relu'))

            model.add(Dense(64, activation='relu'))
            model.add(Dense(output_units))
            model.compile(optimizer='adam', loss='mse')
            return model

        models = {}
        predictions = {}
        metrics = {'Model': [], 'Target': [], 'MAE': [], 'RMSE': []}

        for mtype in ['cnn', 'lstm', 'cnn_lstm']:
            if mtype == 'lstm':
                X_train_m = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_test_m = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            else:
                X_train_m = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test_m = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = create_model(mtype, X_train_m.shape[1:], len(target_cols))
            model.fit(X_train_m, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            pred = model.predict(X_test_m)

            models[mtype] = model
            predictions[mtype] = pred

            # Plot predictions
            st.subheader(f"Model: {mtype.upper()} 🔧")
            for i, col in enumerate(target_cols):
                mae = mean_absolute_error(Y_test[:, i], pred[:, i])
                rmse = np.sqrt(mean_squared_error(Y_test[:, i], pred[:, i]))

                metrics['Model'].append(mtype.upper())
                metrics['Target'].append(col)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(Y_test[:100, i], label='Actual', linestyle='--')
                ax.plot(pred[:100, i], label='Predicted')
                ax.set_title(f'{mtype.upper()} Model - Actual vs Predicted: {col}')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel(col)
                ax.legend()
                st.pyplot(fig)

        # Display Metrics Comparison
        st.subheader("Model Performance Comparison 📊")
        df_metrics = pd.DataFrame(metrics)

        for metric_type in ['MAE', 'RMSE']:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_pivot = df_metrics.pivot(index='Target', columns='Model', values=metric_type)
            df_pivot.plot(kind='bar', ax=ax, colormap='tab10')
            ax.set_title(f'Model Performance Comparison ({metric_type})', fontsize=14)
            ax.set_ylabel(metric_type)
            ax.set_xlabel('Water Quality Parameter')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

        st.write("""
        The **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** are used to evaluate prediction accuracy
        for each water quality parameter. Lower values indicate better model performance.
        """)

    # ------------------- SECTION 4: WATER QUALITY INDEX -------------------
    elif section == "Water Quality Index":
        st.header("Water Quality Index (WQI) 🌊")
        df['WQI'] = 0.1 * df['Surface temp'] + 0.2 * df['pH'] + 0.15 * df['Ammonia'] + 0.2 * df['Dissolved Oxygen']

        def classify_wqi(wqi):
            if wqi > 0.8: return 'Excellent'
            elif wqi > 0.6: return 'Good'
            elif wqi > 0.4: return 'Fair'
            else: return 'Poor'

        df['WQI_Category'] = df['WQI'].apply(classify_wqi)

        st.subheader("WQI Trend Over Time 📉")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=df['Date'], y=df['WQI'], label='WQI', color='purple')
        plt.title('Water Quality Index Over Time')
        plt.xlabel('Date')
        plt.ylabel('WQI')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)

        st.subheader("WQI Statistics 📑")
        st.write(df['WQI'].describe())

        st.subheader("WQI Classification Distribution 📊")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='WQI_Category', data=df, ax=ax)
        ax.set_title('WQI Classification Distribution')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Water Quality Index (WQI) is a comprehensive measure of water quality that incorporates several key water quality parameters. The categories range from **Excellent** to **Poor** based on the values calculated.")

