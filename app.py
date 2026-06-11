import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Support Ticket Classifier",
    layout="wide"
)

st.title(
    "🎫 Support Ticket Classification System"
)

st.write(
    "Predict the category of a support ticket using Machine Learning."
)
from joblib import load

model = load(
    "model/ticket_classifier.pkl"
)

vectorizer = load(
    "model/vectorizer.pkl"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    "85.6%"
)

col2.metric(
    "Categories",
    "8"
)

col3.metric(
    "Tickets",
    "47,837"
)

ticket_text = st.text_area(
    "Enter Ticket Description"
)
st.info("""
Example Tickets:

• I forgot my password and cannot access my account

• My laptop battery drains very quickly

• Please provide access to the shared drive

• Need approval to purchase a new monitor

• Unable to connect to the VPN
""")

if st.button(
    "Predict Category"
):

    ticket_vector = vectorizer.transform(
        [ticket_text]
    )

    prediction = model.predict(
        ticket_vector
    )

    st.markdown("---")

    st.subheader("🎯 Prediction Result")

    st.metric(
        "Predicted Category",
        prediction[0]
    )
st.subheader("ℹ️ Project Information")

st.write("""
This machine learning model classifies IT support tickets into categories such as:

- Access
- Hardware
- HR Support
- Storage
- Purchase
- Administrative Rights
- Internal Project
- Miscellaneous

Model: LinearSVC
Accuracy: 85.6%
Dataset Size: 47,837 tickets
""")    

category_df = pd.read_csv("data/category_counts.csv")
category_df.columns = ["Category", "Count"]

st.subheader("📊 Ticket Category Distribution")

fig = px.bar(
    category_df,
    x="Category",
    y="Count",
    title="Distribution of Ticket Categories"
)

st.plotly_chart(
    fig,
    use_container_width=True
)    

st.subheader("📈 Model Evaluation")

st.image(
    "images/confusion_matrix.png",
    caption="Confusion Matrix",
    width=600
)

st.markdown("---")

st.success(
    "Model loaded successfully."
)

st.markdown(
    "Developed by Pranathi | Future Interns ML Task 2"
)

