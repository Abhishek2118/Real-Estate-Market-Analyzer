import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset (You can replace it with real estate data)
def load_data():
    data = {
        'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Area_sqft': [1200, 1500, 1100, 1800, 1300],
        'Bedrooms': [2, 3, 2, 4, 3],
        'Bathrooms': [1, 2, 1, 3, 2],
        'Price': [500000, 650000, 400000, 700000, 480000]
    }
    return pd.DataFrame(data)

# Train ML Model for Price Prediction
def train_model(df):
    X = df[['Area_sqft', 'Bedrooms', 'Bathrooms']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Streamlit UI
def main():
    st.set_page_config(page_title="Real Estate Analyzer", layout="wide", page_icon="🏡")
    st.markdown("""
        <style>
            .main-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #2E86C1;
            }
            .sub-title {
                text-align: center;
                font-size: 20px;
                color: #117A65;
            }
            .stButton>button {
                background-color: #2E86C1;
                color: white;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-title'>🏡 AI-Based Real Estate Market Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Analyze, Predict & Visualize Real Estate Market Trends</div>", unsafe_allow_html=True)
    
    df = load_data()
    st.sidebar.header("🔍 Data Overview")
    st.sidebar.write("### Sample Real Estate Data")
    st.sidebar.dataframe(df)
    
    # Train Model
    model, X_test, y_test, y_pred = train_model(df)
    
    # Model Evaluation
    st.sidebar.write("### Model Performance Metrics")
    st.sidebar.write(f"✔️ Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.sidebar.write(f"✔️ Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.sidebar.write(f"✔️ R² Score: {r2_score(y_test, y_pred):.2f}")
    
    # Visualization
    st.write("## 📊 Price Distribution")

    # Creating the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size

    # Scatter plot with improved aesthetics
    sc = ax.scatter(
        y_test, y_pred, 
        c=np.abs(y_test - y_pred),  # Color by error magnitude
        cmap='coolwarm',  # Use a color gradient for better visualization
        edgecolors='black',  # Add edge color for better distinction
        alpha=0.7,  # Transparency for clarity
        s=80  # Increase marker size
    )

    # Adding a color bar to show the error intensity
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Prediction Error", fontsize=12)

    # Reference Line for Ideal Predictions
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="black", linestyle="dashed", linewidth=2)

    # Labels and Title
    ax.set_xlabel("Actual Prices", fontsize=12, fontweight='bold', color='darkblue')
    ax.set_ylabel("Predicted Prices", fontsize=12, fontweight='bold', color='darkblue')
    ax.set_title("Actual vs Predicted Prices", fontsize=14, fontweight='bold', color='darkred')

    # Adding Grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Displaying the plot in Streamlit
    st.pyplot(fig)

    # User Input for Prediction
    st.write("## 🔮 Predict Property Price")
    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Enter Area (sqft):", min_value=500, max_value=5000, value=1200, step=100)
    with col2:
        bedrooms = st.number_input("Enter Number of Bedrooms:", min_value=1, max_value=10, value=2)
    with col3:
        bathrooms = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=5, value=1)
    
    if st.button("🔍 Predict Price"):
        input_data = np.array([[area, bedrooms, bathrooms]])  # Ensure 2D array
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Property Price: ${predicted_price:,.2f}")
    
    # Additional Market Insights
    st.write("## 📈 Market Insights")
    avg_price = df['Price'].mean()
    st.info(f"🏠 Average Property Price in Dataset: ${avg_price:,.2f}")
    
    # Price Trend Visualization
    st.write("## 📊 Price Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📉 Price Trend by Area Size")
        fig, ax = plt.subplots()
        ax.plot(df['Area_sqft'], df['Price'], marker='o', linestyle='-', color='b')
        ax.set_xlabel("Area (sqft)")
        ax.set_ylabel("Price")
        ax.set_title("Property Price Trend by Area")
        st.pyplot(fig)
    
    with col2:
        st.write("### 📊 Price Trend by Bedrooms")
        fig, ax = plt.subplots()
        ax.bar(df['Bedrooms'], df['Price'], color='g')
        ax.set_xlabel("Number of Bedrooms")
        ax.set_ylabel("Price")
        ax.set_title("Property Price Trend by Bedrooms")
        st.pyplot(fig)

# Run the main function
if __name__ == "__main__":
    main()
