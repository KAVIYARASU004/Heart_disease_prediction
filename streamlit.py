import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data():
    return pd.read_csv("heart.csv")

# Apply custom CSS
def apply_custom_styles():
    st.markdown(
        """
        <style>
        h1 {
            font-size: 2.5rem !important;
        }
        h2 {
            font-size: 2rem !important;
        }
        h3 {
            font-size: 1.75rem !important;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            font-size: 1.5rem !important;
        }
        .stMarkdown p {
            font-size: 1.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main Streamlit app
def main():
    apply_custom_styles()

    st.title("Heart Disease Dataset Analysis")
    st.sidebar.title("Navigation")

    # Load dataset
    data = load_data()

    # Sidebar options
    options = ["Dataset Overview", "Data Visualization", "Prediction Model"]
    choice = st.sidebar.radio("Select an option:", options)

    if choice == "Dataset Overview":
        st.header("Dataset Overview")
        st.write(data.head())

        # Display dataset information
        st.subheader("Dataset Summary")
        st.write(data.describe())

        st.subheader("Data Types")
        st.write(data.dtypes)

    elif choice == "Data Visualization":
        st.header("Data Visualization")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

        # Pairplot for selected features
        st.subheader("Pairplot of Features")
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        selected_features = st.multiselect(
            "Select numeric features to visualize:", numeric_cols, default=["age", "thalach", "chol"]
        )
        if selected_features:
            try:
                pairplot = sns.pairplot(data[selected_features + ["target"]], hue="target", diag_kind="kde")
                st.pyplot(pairplot.fig)
            except ValueError as e:
                st.error(f"Error creating pairplot: {e}")

    elif choice == "Prediction Model":
        st.header("Prediction Model")

        # Feature selection
        X = data.drop("target", axis=1)
        y = data["target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model performance
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Prediction
        st.subheader("Make a Prediction")
        input_data = {}
        for column in X.columns:
            input_data[column] = st.number_input(f"{column}", value=float(data[column].mean()))
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write("Prediction: Heart Disease" if prediction == 1 else "Prediction: No Heart Disease")

if __name__ == "__main__":
    main()
