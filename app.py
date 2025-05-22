from PIL import Image
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
import io

# Load data function to read csv or excel files
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

import streamlit as st
import pandas as pd
from PIL import Image
import base64
import io

# ----------------- Custom Background Logic -----------------
def get_base64_of_bin_file(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_fullscreen_background(image_path):
    base64_str = get_base64_of_bin_file(image_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        </style>
        """, unsafe_allow_html=True)

# ----------------- Sidebar -----------------
st.sidebar.title("Water Quality Analysis App")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

section = st.sidebar.selectbox("Select Section", [
    "Overview",
    "Exploratory Data Analysis",
    "Predictive Analysis",
    "Water Quality Index"
])

# ----------------- If No File: Show Welcome Screen -----------------
if uploaded_file is None:
    try:
        set_fullscreen_background("water quality analysis.jpg")  # Use your image file here
    except:
        st.title("Welcome to Water Quality Analysis")
        st.info("Please upload a dataset file from the sidebar to begin.")
    st.stop()

# ----------------- Reset to White Background After Upload -----------------
st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
        background-image: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- Read Uploaded File -----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# ----------------- Sections Placeholder -----------------
if section == "Overview":
    st.header("📊 Overview")
    st.write("Overview content goes here.")

elif section == "Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")
    st.write("EDA content goes here.")

elif section == "Predictive Analysis":
    st.header("🤖 Predictive Analysis")
    st.write("Model prediction tools go here.")

elif section == "Water Quality Index":
    st.header("🌊 Water Quality Index")
    st.write("WQI calculations go here.")

# Main app content after upload
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        # Preview data for debugging and confirmation (optional)
        # st.write(df.head())

        # Convert Month to string if present (to avoid errors)
        if 'Month' in df.columns:
            df['Month'] = df['Month'].astype(str).str.strip()

        # Create 'Date' from 'Month' and 'Year' if both are present
        if ('Month' in df.columns) and ('Year' in df.columns):
            df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str), format='%B %Y', errors='coerce')
            if df['Date'].isnull().any():
                st.warning("Some rows have invalid or missing 'Date' after combining Month and Year.")
            # Sort by Date ascending
            df.sort_values('Date', inplace=True)

            # Drop the 'Year' column as it's no longer needed
            df.drop(columns=['Year'], inplace=True)

        # Convert numeric columns properly
        numeric_cols_to_convert = [
            'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
            'pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen',
            'Sulfide', 'Carbon Dioxide', 'Air Temperature (0C)'
        ]
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)
        # Fill missing numeric values with median of column
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Normalize numeric columns
        scaler = MinMaxScaler()
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Sections logic
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
            - *Date*: Date of data collection, calculated from 'Month'.
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
                st.write(df.select_dtypes(include=['float', 'int']).describe())

        elif section == "Exploratory Data Analysis":
            st.title("Exploratory Data Analysis")
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

            st.write("EDA Complete! ✅")

        elif section == "Predictive Analysis":
            st.header("Water Quality Prediction Using Deep Learning Models 🔮")
            st.success("Dataset loaded successfully!")
            st.dataframe(df.head())

            # Sidebar for model parameters
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

            # Frequency selector (not used currently, future enhancement)
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

            required_features = ['Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature', 'pH',
                                 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide']
            water_cols = [col for col in required_features if col in df_filtered.columns]

            if len(water_cols) < len(required_features):
                missing = set(required_features) - set(water_cols)
                st.error(f"Missing input features: {', '.join(missing)}")
                st.stop()

            # Prepare clean data
            df_clean = df_filtered[water_cols + target_cols].copy()
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce')  # Ensure numeric
            df_clean.dropna(inplace=True)

            if df_clean.shape[0] < 10:
                st.error("Not enough clean data rows after removing NaNs.")
                st.stop()

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

            st.subheader("Performance Comparison (MAE & RMSE)")
            import matplotlib.ticker as ticker

            pivot_mae = df_metrics.pivot(index='Target', columns='Model', values='MAE')
            pivot_rmse = df_metrics.pivot(index='Target', columns='Model', values='RMSE')

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

        elif section == "Water Quality Index":
            st.header("Water Quality Index (WQI) by Site 🌊")

            if 'Site' not in df.columns:
                st.error("No 'Site' column found in dataset for WQI calculation.")
                st.stop()

            sites = sorted(df['Site'].dropna().unique().tolist())
            selected_site = st.selectbox("Select Site for WQI calculation:", options=["All Sites"] + sites)

            wqi_params = ['pH', 'Ammonia', 'Nitrate', 'Phosphate', 'Dissolved Oxygen']

            if not all(param in df.columns for param in wqi_params):
                st.error("Some required parameters for WQI calculation are missing.")
                st.stop()

            def compute_wqi(row):
                try:
                    ideal = {
                        'pH': 7,
                        'Ammonia': 0.5,
                        'Nitrate': 10,
                        'Phosphate': 0.1,
                        'Dissolved Oxygen': 6
                    }
                    weights = {
                        'pH': 0.2,
                        'Ammonia': 0.2,
                        'Nitrate': 0.2,
                        'Phosphate': 0.2,
                        'Dissolved Oxygen': 0.2
                    }
                    q_scores = []
                    for param in wqi_params:
                        val = row[param]
                        ideal_val = ideal[param]
                        if param == 'Dissolved Oxygen':
                            qi = (val / ideal_val) * 100
                        else:
                            qi = (ideal_val / (val + 1e-6)) * 100
                        qi = min(qi, 100)
                        q_scores.append(qi * weights[param])
                    return sum(q_scores)
                except:
                    return np.nan

            if selected_site == "All Sites":
                wqi_by_site = df.groupby('Site')[wqi_params].mean().dropna()
                wqi_by_site['WQI'] = wqi_by_site.apply(compute_wqi, axis=1)

                st.subheader("Average WQI by Site")
                st.dataframe(wqi_by_site[['WQI']])

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x=wqi_by_site.index, y=wqi_by_site['WQI'], palette='Blues_d')
                plt.axhline(80, color='green', linestyle='--', label='Good Quality Threshold')
                plt.axhline(50, color='orange', linestyle='--', label='Marginal Quality')
                plt.axhline(30, color='red', linestyle='--', label='Poor Quality')
                plt.title('Water Quality Index by Site')
                plt.ylabel('WQI')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(fig)

                st.subheader("Recommendations Based on WQI")
                for site, row in wqi_by_site.iterrows():
                    score = row['WQI']
                    if score >= 80:
                        st.success(f"✅ *{site}*: Excellent water quality. Suitable for most uses.")
                    elif score >= 50:
                        st.warning(f"⚠️ *{site}*: Moderate water quality. May require treatment.")
                    else:
                        st.error(f"🚫 *{site}*: Poor water quality. Likely unsuitable without remediation.")
            else:
                df_site = df[df['Site'] == selected_site].copy()
                df_site.dropna(subset=wqi_params, inplace=True)
                df_site['WQI'] = df_site.apply(compute_wqi, axis=1)

                st.subheader(f"Water Quality Index Over Time – {selected_site}")
                plt.figure(figsize=(12, 5))
                plt.plot(df_site['Date'], df_site['WQI'], marker='o')
                plt.axhline(80, color='green', linestyle='--', label='Good Quality Threshold')
                plt.axhline(50, color='orange', linestyle='--', label='Marginal Quality')
                plt.axhline(30, color='red', linestyle='--', label='Poor Quality')
                plt.title(f'WQI Over Time - {selected_site}')
                plt.xlabel('Date')
                plt.ylabel('WQI')
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)

                avg_wqi = df_site['WQI'].mean()
                st.metric("Average WQI", f"{avg_wqi:.2f}")

                st.subheader("Recommendation")
                if avg_wqi >= 80:
                    st.success("✅ Excellent water quality. Continue monitoring and conservation.")
                elif avg_wqi >= 50:
                    st.warning("⚠️ Moderate quality. Investigate potential sources of pollutants.")
                else:
                    st.error("🚫 Poor water quality. Immediate action and remediation recommended.")
