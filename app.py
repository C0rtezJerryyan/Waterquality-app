import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Flatten, MaxPooling1D
from PIL import Image
import io

# Load data function to read csv or excel files (added because you used it but never defined)
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ------------------- Streamlit App -------------------
# File upload widget moved to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

# When no file is uploaded
if uploaded_file is None:
    st.title("Welcome to Water Quality Analysis")
    try:
        image = Image.open("water quality analysis.png")  # Ensure this image exists
        st.image(image, use_container_width=True)
    except Exception:
        st.write("Welcome image not found.")

# Sidebar for navigation
section = st.sidebar.radio("Select Section", ["Overview", "EDA", "Predictive Analysis", "Water Quality Index"])

# When file is uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        # ------------------- DATA PREPROCESSING -------------------
        numeric_cols_to_convert = [
            'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
            'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
            'Sulfide', 'Carbon Dioxide', 'Air Temperature (0C)'
        ]

        # Convert columns to numeric and handle errors
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.drop_duplicates(inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        if 'Month' in df.columns and 'Year' in df.columns:
            df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str), format='%B %Y', errors='coerce')
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
            phosphate concentrations, dissolved oxygen, sulfide and carbon dioxide. Weather data such as wind direction 
            and air temperature, along with site info, are included. The dataset spans months and years for time-series analysis.
            """)
            st.subheader("Column Descriptions")
            st.write("""
            - *Date*: Date of data collection, from 'Year' and 'Month'.
            - *Water Quality Parameters*: temperature layers, pH, ammonia, nitrate, phosphate, dissolved oxygen, sulfide, carbon dioxide.
            - *Weather and Site Info*: Weather condition (categorical), wind direction, site IDs.
            - *Normalized Values*: Numeric columns scaled 0-1 via MinMaxScaler.
            """)
            st.subheader("Data Preprocessing 🛠️")
            st.write("""
            - Missing numeric values filled with median.
            - Duplicate records removed.
            - Numeric features normalized (0 to 1).
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
                if 'Air Temperature (0C)' in df.columns and 'Surface temp' in df.columns:
                    sns.scatterplot(x='Air Temperature (0C)', y='Surface temp', data=df, ax=axes[0])
                axes[0].set_title('Air vs Surface Temperature')
                if 'Surface temp' in df.columns and 'Dissolved Oxygen' in df.columns:
                    sns.scatterplot(x='Surface temp', y='Dissolved Oxygen', data=df, ax=axes[1])
                axes[1].set_title('Surface Temp vs Dissolved Oxygen')
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("Time-Series of Key Water Quality Parameters 📈"):
                st.subheader("Time-Series of Key Water Quality Parameters")
                fig, axes = plt.subplots(3, 2, figsize=(15, 10))
                params = ['Dissolved Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Carbon Dioxide']
                for i, param in enumerate(params):
                    if param in df.columns:
                        ax = axes[i // 2, i % 2]
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

            # Use df already loaded from uploaded_file
            st.success("Dataset loaded successfully!")
            st.dataframe(df.head())

            # Sidebar
            epochs = st.sidebar.slider("Epochs", 1, 100, 10)
            batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64])

            # SITE selection
            if 'Site' in df.columns:
                sites = df['Site'].dropna().unique()
                selected_site = st.sidebar.selectbox("Select Site", options=sorted(sites))
                df_filtered = df[df['Site'] == selected_site].copy()
            else:
                st.sidebar.info("No 'Site' column found in dataset.")
                df_filtered = df.copy()

            # Frequency selector
            frequency = st.sidebar.selectbox("Select Frequency", ["Weekly", "Monthly", "Yearly"])

            # Convert categorical columns to numeric codes safely
            for cat_col in ['Weather Condition', 'Wind Direction', 'Site']:
                if cat_col in df_filtered.columns:
                    df_filtered[cat_col] = df_filtered[cat_col].astype('category').cat.codes

            # Target selection
            all_possible_targets = ['pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide']
            available_targets = [col for col in all_possible_targets if col in df_filtered.columns]

            target_cols = st.multiselect(
                "Select water quality parameters to predict:",
                options=available_targets,
                default=[]
            )

            if not target_cols:
                st.warning("⚠️ Please select at least one parameter to proceed.")
                st.stop()

            # Required input features present in dataset
            required_features = ['Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature', 'pH',
                                 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide']
            water_cols = [col for col in required_features if col in df_filtered.columns]

            if len(water_cols) < len(required_features):
                missing = set(required_features) - set(water_cols)
                st.error(f"Missing input features: {', '.join(missing)}")
                st.stop()

            # Ensure all input features and target columns are numeric and drop missing rows
            df_clean = df_filtered[water_cols + target_cols].copy()
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce')  # Convert all to numeric
            df_clean.dropna(inplace=True)  # Drop rows with NaNs

            if df_clean.shape[0] < 10:
                st.error("Not enough clean data rows after removing NaNs.")
                st.stop()

            # Prepare data with prediction gap (assuming weekly = 1)
            prediction_gap_weeks = 1

            X = df_clean[water_cols].values[:-prediction_gap_weeks].astype(np.float32)
            Y = df_clean[target_cols].values[prediction_gap_weeks:].astype(np.float32)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            def create_model(model_type, input_shape, output_units):
                model = Sequential()
                if model_type == 'cnn':
                    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Flatten())
                elif model_type == 'lstm':
                    model.add(LSTM(64, input_shape=input_shape, activation='relu'))
                elif model_type == 'cnn_lstm':
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

                st.write(f"Training {mtype.upper()} model...")
                model = create_model(mtype, X_train_m.shape[1:], len(target_cols))
                model.fit(X_train_m, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                pred = model.predict(X_test_m)

                models[mtype] = model
                predictions[mtype] = pred

                st.subheader(f"{mtype.upper()} Model Results")

                for i, col in enumerate(target_cols):
                    mae = mean_absolute_error(Y_test[:, i], pred[:, i])
                    rmse = np.sqrt(mean_squared_error(Y_test[:, i], pred[:, i]))

                    metrics['Model'].append(mtype.upper())
                    metrics['Target'].append(col)
                    metrics['MAE'].append(mae)
                    metrics['RMSE'].append(rmse)

                    st.write(f"Parameter: {col}")
                    st.write(f" - MAE: {mae:.4f}")
                    st.write(f" - RMSE: {rmse:.4f}")

                    plt.figure(figsize=(10, 4))
                    plt.plot(Y_test[:, i], label='Actual')
                    plt.plot(pred[:, i], label='Predicted')
                    plt.title(f"{mtype.upper()} Prediction vs Actual for {col}")
                    plt.legend()
                    st.pyplot(plt)
                    plt.close()

            st.subheader("Summary Metrics Table")
            df_metrics = pd.DataFrame(metrics)
            st.dataframe(df_metrics)
            # After displaying the summary metrics table
            st.subheader("Performance Comparison (MAE & RMSE)")

            import matplotlib.ticker as ticker

            # Prepare data for bar chart
            # Group by Model and Target, then get mean MAE and RMSE (usually only one value per combo)
            pivot_mae = df_metrics.pivot(index='Target', columns='Model', values='MAE')
            pivot_rmse = df_metrics.pivot(index='Target', columns='Model', values='RMSE')

            # Plot MAE comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_mae.plot(kind='bar', ax=ax)
            ax.set_title('Mean Absolute Error (MAE) Comparison')
            ax.set_ylabel('MAE')
            ax.set_xlabel('Water Quality Parameter')
            ax.legend(title='Model')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Plot RMSE comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_rmse.plot(kind='bar', ax=ax)
            ax.set_title('Root Mean Squared Error (RMSE) Comparison')
            ax.set_ylabel('RMSE')
            ax.set_xlabel('Water Quality Parameter')
            ax.legend(title='Model')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ------------------- SECTION 4: WATER QUALITY INDEX -------------------
        elif section == "Water Quality Index":
            st.header("Water Quality Index (WQI) by Site 🌊")

            # Ensure Site column exists
            if 'Site' not in df.columns:
                st.error("No 'Site' column found in dataset for WQI calculation.")
                st.stop()

            sites = df['Site'].dropna().unique()
            selected_site = st.selectbox("Select Site for WQI calculation:", options=sorted(sites))

            # Filter df by selected site
            df_site = df[df['Site'] == selected_site].copy()

            # Parameters used for WQI calculation
            wqi_params = ['pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']

            if not all(param in df_site.columns for param in wqi_params):
                st.error("Required parameters for WQI calculation not found in dataset.")
                st.stop()

            # Calculate WQI using your formula
            df_site['WQI'] = (
                df_site['pH'] +
                df_site['Ammonia'] +
                df_site['Nitrate'] +
                df_site['Phosphate'] +
                df_site['Dissolved Oxygen']
            ) / 5

            st.write(f"Summary statistics of WQI for Site: **{selected_site}**")
            st.write(df_site[['Date', 'WQI']].describe())

            st.subheader(f"WQI Trend over Time for Site: {selected_site}")
            plt.figure(figsize=(14, 6))
            plt.plot(df_site['Date'], df_site['WQI'], label='WQI', color='b')
            plt.xlabel('Date')
            plt.ylabel('Water Quality Index')
            plt.title(f'Water Quality Index Over Time at Site: {selected_site}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

st.sidebar.markdown("---")
st.sidebar.markdown("© Water Quality App 2025")
