import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("car_price_model.pkl")
brand_encoder = joblib.load("brand_encoder.pkl")
model_encoder = joblib.load("model_encoder.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cars.csv")

brand_model_dict = (
    df.groupby("Brand")["Model"]
      .unique()
      .apply(list)
      .to_dict()
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main{
    background:#F4F7FB;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#0F4C81;
}

.subtitle{
    text-align:center;
    color:#6c757d;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#0F4C81;
    color:white;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#0C3A63;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 3px 10px rgba(0,0,0,0.12);
}

.result-card{
    background:#ffffff;
    border-left:8px solid #16A34A;
    padding:25px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
    text-align:center;
    margin-top:20px;
}

.sidebar .sidebar-content{
    background:#0F4C81;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🚗 Car Price AI")

    st.write("### Predict Using")

    st.success("✔ Brand")
    st.success("✔ Model")
    st.success("✔ Manufacturing Year")
    st.success("✔ Kilometres Driven")

    st.divider()

    st.info("""
Estimate the resale value of
second-hand cars instantly using
Machine Learning.
""")

# ---------------- HEADER ----------------
st.markdown(
    "<div class='title'>🚗 Second Hand Car Price Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict the resale value of used cars in India using AI</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT SECTION ----------------
left, right = st.columns([2,1])

with left:

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "🏷 Select Brand",
            sorted(brand_model_dict.keys())
        )

        year = st.number_input(
            "📅 Manufacturing Year",
            min_value=1995,
            max_value=2025,
            value=2018
        )

    with col2:

        car_model = st.selectbox(
            "🚘 Select Model",
            sorted(brand_model_dict[brand])
        )

        kms = st.number_input(
            "🛣 Kilometres Driven",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000
        )

    st.write("")

    if st.button("🚗 Predict Resale Price"):

        brand_encoded = brand_encoder.transform([brand])[0]
        model_encoded = model_encoder.transform([car_model])[0]

        input_data = [[
            brand_encoded,
            model_encoded,
            year,
            kms
        ]]

        predicted_price = model.predict(input_data)[0]

        st.balloons()

        st.markdown(f"""
<div class="result-card">

<h2>💰 Estimated Resale Value</h2>

<h1 style="color:#16A34A;">
₹ {predicted_price:,.0f}
</h1>

<p>
This is the estimated market value of the selected
second-hand vehicle based on the trained Machine Learning model.
</p>

</div>
""", unsafe_allow_html=True)

with right:

    st.subheader("📊 Vehicle Summary")

    st.metric("Brand", brand)

    st.metric("Model", car_model)

    st.metric("Year", year)

    st.metric("Kilometres", f"{kms:,}")

    st.divider()

    st.subheader("💡 Tips")

    st.info("✔ Lower kilometres usually increase resale value.")

    st.info("✔ Newer manufacturing years generally fetch higher prices.")

    st.info("✔ Popular brands often retain better resale value.")

st.divider()

st.caption("🚀 Built with Streamlit • Scikit-Learn • Machine Learning")
