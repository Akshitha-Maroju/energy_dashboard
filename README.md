# ⚡ Energy Dashboard

A Streamlit-based web app to analyze and predict household energy usage across Indian regions.

## 🚀 Features

- 🔐 User Login System
- 📂 Upload CSV energy data
- 📊 Visualizations (Income vs Usage, Appliance analysis)
- 🌍 Region-based filtering
- 💡 Smart energy-saving recommendations
- 🤖 ML-based consumption prediction (Random Forest)
- 📥 Download filtered results

## 📁 Sample Data Columns

- Household_ID, Region, Monthly_Income_INR
- Monthly_Energy_Consumption_kWh
- Appliance_AC, Appliance_Fan, Appliance_Light
- Fridge, Washing_Machine, EV_Charging

## ▶️ Run the App

```bash
git clone https://github.com/yourusername/energy-dashboard.git
cd energy-dashboard
pip install -r requirements.txt
streamlit run app.py
