import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------- Login System ----------
users = {"admin": "admin123", "user": "user123"}
login_status = False

# Inject custom CSS for pink background during login and file upload
st.markdown("""
    <style>
    .main-login, .main-upload {
        background-color: #ffe6f0 !important;
        padding: 2rem;
        border-radius: 15px;
    }
    body {
        background-color: #ffc0cb;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
    }
    h1, h2, h3 {
        color: #b30059;
        font-weight: 700;
        text-align: center;
    }
    .stButton > button {
        background-color: #ff66b2;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #cc0066;
    }
    </style>
""", unsafe_allow_html=True)

# Login form
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.container():
        st.markdown("<div class='main-login'>", unsafe_allow_html=True)
        st.title("🔐 Energy Dashboard Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------- Main Dashboard ----------
st.set_page_config(page_title="Energy Dashboard", layout="wide")

# Logo and Title
st.image("logo.png", width=120)
st.title("⚡ Indian Energy Consumer Dashboard")
st.markdown("Visualize and explore energy usage patterns across India.")

# Dataset Upload
with st.container():
    st.markdown("<div class='main-upload'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Energy Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

st.title("🏠 Energy Dashboard for Housing Complex")

# Sidebar Filters
region = st.sidebar.selectbox("🌍 Select Region", ["All"] + sorted(df["Region"].unique().tolist()))
if region != "All":
    df = df[df["Region"] == region]

# Display Data
st.subheader("📊 Household Energy Consumption Overview")
st.dataframe(df.head())

# Metrics
avg_energy = df["Monthly_Energy_Consumption_kWh"].mean()
total_energy = df["Monthly_Energy_Consumption_kWh"].sum()

col1, col2 = st.columns(2)
with col1:
    st.metric("🔋 Average Monthly Consumption (kWh)", f"{avg_energy:.2f}")
with col2:
    st.metric("⚡ Total Energy Consumption (kWh)", f"{total_energy:.0f}")

# Scatter Plot: Income vs Energy
st.subheader("💰 Income vs Energy Consumption")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="Monthly_Income_INR", y="Monthly_Energy_Consumption_kWh", hue="Region", ax=ax1)
st.pyplot(fig1)

# Bar Plot: Appliance-wise Energy
st.subheader("🔧 Appliance-wise Count vs Energy Consumption")
appliances = ["Appliance_AC", "Appliance_Fan", "Appliance_Light", "Fridge", "Washing_Machine", "EV_Charging"]
selected_appliance = st.selectbox("Select Appliance", appliances)

grouped = df.groupby(selected_appliance)["Monthly_Energy_Consumption_kWh"].mean().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=grouped, x=selected_appliance, y="Monthly_Energy_Consumption_kWh", ax=ax2)
ax2.set_xlabel(f"No. of {selected_appliance.replace('_', ' ')}")
ax2.set_ylabel("Avg Energy Consumption (kWh)")
st.pyplot(fig2)

# Smart Recommendations
st.subheader("🤖 Smart Recommendations")
recommendations = []

for _, row in df.iterrows():
    if row["Monthly_Energy_Consumption_kWh"] > 250:
        msg = f"Household ID {row['Household_ID']} - High usage! Switch to solar and LED bulbs."
        st.warning(msg)
        recommendations.append(msg)
    elif row["EV_Charging"] == 1:
        msg = f"Household ID {row['Household_ID']} - Install a separate EV meter for optimal billing."
        st.info(msg)
        recommendations.append(msg)

# Download Recommendations
if recommendations:
    st.download_button("📥 Download Recommendations", "\n".join(recommendations), "recommendations.txt")

# Optional: Download Filtered Data
st.download_button("📁 Download Filtered Data", df.to_csv(index=False), file_name="filtered_energy_data.csv")

# ML Prediction Section
st.markdown("## 🔮 Predict Energy Consumption")

@st.cache_data
def train_model(data):
    features = ["Monthly_Income_INR", "Appliance_AC", "Appliance_Fan", "Appliance_Light", "Fridge", "Washing_Machine", "EV_Charging"]
    X = data[features]
    y = data["Monthly_Energy_Consumption_kWh"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model(df)

with st.form("prediction_form"):
    income = st.number_input("Monthly Income (INR)", min_value=0, value=30000)
    ac = st.slider("No. of ACs", 0, 5, 1)
    fan = st.slider("No. of Fans", 0, 10, 3)
    light = st.slider("No. of Lights", 0, 15, 5)
    fridge = st.selectbox("Has Fridge?", [0, 1])
    washing_machine = st.selectbox("Has Washing Machine?", [0, 1])
    ev = st.selectbox("Has EV Charging?", [0, 1])
    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([[income, ac, fan, light, fridge, washing_machine, ev]],
                                columns=["Monthly_Income_INR", "Appliance_AC", "Appliance_Fan", "Appliance_Light",
                                         "Fridge", "Washing_Machine", "EV_Charging"])
        prediction = model.predict(input_df)[0]
        st.success(f"🔋 Predicted Monthly Energy Consumption: {prediction:.2f} kWh")

# Conclusion Section
st.markdown("### 📌 Conclusion")
st.markdown("""
This dashboard helps analyze household energy usage and gives smart, region-specific recommendations for energy efficiency.

**Next Steps for Enhancement:**
- Add machine learning to predict energy consumption. ✅
- Visualize energy trends over time.
- Deploy using Streamlit Cloud or Docker.
""")
