
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(layout='wide', page_title='Diabetes Diagnosis Classifier', page_icon='🏥')

# ---------- Palette (exact colors provided) ----------
BG = '#FFF6FB'      # page background - very light pink
PINK = '#FFD6E8'
SKY = '#7FD6FF'
MINT = '#6FE3B4'
NAVY = '#2B4A66'    # sidebar / headers / dark accent

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: {BG} !important; }}
h1, h2, h3 {{ color: {NAVY}; }}
.stButton>button {{
    background-color: {NAVY}; color: white; border-radius: 8px; border: none;
}}
div[data-baseweb="select"] > div {{ border-color: {NAVY} !important; background-color: white; }}
.block-container {{ padding-top: 2rem; }}
[data-testid="stMetric"] {{
    background-color: white; border: 1px solid {PINK}; border-left: 6px solid {NAVY};
    border-radius: 10px; padding: 10px 14px;
}}
</style>
""", unsafe_allow_html=True)

# CLASS colors mapped from the same palette
CLASS_COLORS = {'N': MINT, 'P': SKY, 'Y': NAVY}
SEQ_SCALE = [[0, BG], [0.5, SKY], [1, NAVY]]

st.sidebar.title("⚕️ Navigation")
page = st.sidebar.radio("", ["Dataset information", "Data analysis", "Prediction"])

@st.cache_data
def load_data():
    return pd.read_csv('diabetes_data_cleaned.csv')

@st.cache_resource
def load_model():
    model = joblib.load('final_diabetes_classification_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

df = load_data()
model, le = load_model()

PLOT_LAYOUT = dict(plot_bgcolor='white', paper_bgcolor='white', font_color=NAVY)

if page == "Dataset information":
    st.title("💉 Diabetes Diagnosis Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي المرضى (Total)", len(df))
    c2.metric("غير مصاب (N)", int((df['CLASS'] == 'N').sum()))
    c3.metric("قبل السكري (P)", int((df['CLASS'] == 'P').sum()))
    c4.metric("مصاب (Y)", int((df['CLASS'] == 'Y').sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

elif page == "Data analysis":
    st.title("📊 Data Analysis")

    chart_choice = st.selectbox(
        'اختاري الرسم البياني (Chart)',
        [
            'عدد المرضى في كل تشخيص (Diagnosis Counts)',
            'متوسط مؤشر كتلة الجسم لكل تشخيص (Mean BMI per Class)',
            'متوسط السكر التراكمي لكل تشخيص (Mean HbA1c per Class)',
            'توزيع التشخيص حسب الجنس (Diagnosis by Gender)',
            'توزيع الأعمار حسب التشخيص (Age Distribution by Class)',
            'مقارنة متوسط قيم الدهون لكل تشخيص (Lipid Comparison)'
        ]
    )

    if chart_choice.startswith('عدد المرضى'):
        counts = df['CLASS'].value_counts().reset_index()
        counts.columns = ['CLASS', 'count']
        fig = px.pie(
            counts, names='CLASS', values='count', color='CLASS',
            color_discrete_map=CLASS_COLORS,
            title='عدد المرضى في كل تشخيص', hole=0.45
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice.startswith('متوسط مؤشر'):
        df_grouped = df.groupby('CLASS')['BMI'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            df_grouped, x='CLASS', y='BMI', color='CLASS',
            color_discrete_map=CLASS_COLORS,
            title='متوسط مؤشر كتلة الجسم لكل تشخيص', text_auto='.2s'
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice.startswith('متوسط السكر'):
        df_grouped = df.groupby('CLASS')['HbA1c'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            df_grouped, x='CLASS', y='HbA1c', color='CLASS',
            color_discrete_map=CLASS_COLORS,
            title='متوسط السكر التراكمي لكل تشخيص', text_auto='.2s'
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice.startswith('توزيع التشخيص'):
        gender_class = df.groupby(['Gender', 'CLASS']).size().reset_index(name='count')
        fig = px.bar(
            gender_class, x='Gender', y='count', color='CLASS', barmode='group',
            color_discrete_map=CLASS_COLORS,
            title='توزيع التشخيص حسب الجنس'
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice.startswith('توزيع الأعمار'):
        fig = px.histogram(
            df, x='AGE', color='CLASS', color_discrete_map=CLASS_COLORS,
            title='توزيع الأعمار حسب التشخيص', barmode='overlay', opacity=0.75
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    else:
        lab_cols = ['Chol', 'TG', 'HDL', 'LDL', 'VLDL']
        melted = df.melt(id_vars='CLASS', value_vars=lab_cols, var_name='Lab Test', value_name='Value')
        grouped = melted.groupby(['CLASS', 'Lab Test'])['Value'].mean().reset_index()
        fig = px.bar(
            grouped, x='Lab Test', y='Value', color='CLASS', barmode='group',
            color_discrete_map=CLASS_COLORS,
            title='مقارنة متوسط قيم الدهون لكل تشخيص'
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)



elif page == "Prediction":
    st.title("🔮 Predict Diabetes Diagnosis")

    gender = st.selectbox('الجنس (Gender)', ['M', 'F'])
    age = st.number_input('العمر  (Age)', 1, 100, 40)
    urea = st.number_input('نسبة اليوريا في الدم (Urea)', 0.0, 50.0, 5.0)
    cr = st.number_input('الكرياتينين (Cr)', 0, 500, 50)
    hba1c = st.number_input('السكر التراكمي (HbA1c %)', 0.0, 20.0, 5.5)
    chol = st.number_input('الكوليسترول الكلي (Chol)', 0.0, 15.0, 4.5)
    tg = st.number_input('الدهون الثلاثية (TG)', 0.0, 15.0, 1.5)
    hdl = st.number_input('الكوليسترول الجيد (HDL)', 0.0, 10.0, 1.2)
    ldl = st.number_input('الكوليسترول الضار (LDL)', 0.0, 15.0, 2.5)
    vldl = st.number_input('كوليسترول منخفض الكثافة جدًا (VLDL)', 0.0, 40.0, 1.0)
    bmi = st.number_input('مؤشر كتلة الجسم (BMI)', 10.0, 50.0, 25.0)

    def bmi_category(bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        else: return 'Obese'

    if st.button('Predict'):
        input_df = pd.DataFrame([{
            'Gender': gender, 'AGE': age, 'Urea': urea, 'Cr': cr, 'HbA1c': hba1c,
            'Chol': chol, 'TG': tg, 'HDL': hdl, 'LDL': ldl, 'VLDL': vldl, 'BMI': bmi,
            'BMI_Category': bmi_category(bmi)
        }])
        pred = model.predict(input_df)
        pred_label = le.inverse_transform(pred)[0]
        st.success(f"Predicted Class: {pred_label}")
