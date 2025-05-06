

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 读取训练集数据（注意指定编码格式）
train_data = pd.read_csv('train_data.csv', encoding='utf-8')  # 或 encoding='gbk'

# 清洗列名：去除首尾空格、特殊符号
train_data.columns = train_data.columns.str.strip().str.replace(' ', '_')

# 检查列名
print("修正后列名:", train_data.columns.tolist())

# 分离输入特征和目标变量（根据实际列名调整）
X = train_data[['Age', 'Sex', 'Race', 'Histological_Type',
                'T', 'Liver_metastasis',
                'Chemotherapy', 'Marital_status']]
y = train_data['Vital_status']

# 创建并训练Random Forest模型
rf_params = {
    'n_estimators': 200,
    'min_samples_split': 10,
    'max_depth': 20
}

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X, y)

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
Age_mapper = {'＜60': 1, '60-73': 2, '＞73': 3}
Sex_mapper = {'male': 1, 'female': 2}
Race_mapper = {"White": 1, "Black": 2, "Other": 3}
Histological_Type_mapper = {"Adenocarcinoma": 1, "Squamous-cell carcinoma": 2} 
T_mapper = {"T4": 4, "T1": 1, "T2": 2, "T3": 3}
Liver_metastasis_mapper = {"No": 1, "Yes": 2}  
Radiation_mapper = {"No": 1, "Yes": 2}      
Chemotherapy_mapper = {"No": 1, "Yes": 2}    
Marital_status_mapper = {"Married/Partnered": 1, "Unmarried/Unstable Relationship": 2}

def predict_Vital_status(Age, Sex, Race, Histological_Type,
                         T, Liver_metastasis, Radiation,
                         Chemotherapy, Marital_status):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[Age]],
        'Sex': [Sex_mapper[Sex]],
        'Race': [Race_mapper[Race]],
        'Histological_Type': [Histological_Type_mapper[Histological_Type]],
        'T': [T_mapper[T]],
        'Liver_metastasis': [Liver_metastasis_mapper[Liver_metastasis]],
        'Radiation': [Radiation_mapper[Radiation]],
        'Chemotherapy': [Chemotherapy_mapper[Chemotherapy]],
        'Marital_status': [Marital_status_mapper[Marital_status]]
    })
    prediction = rf_model.predict(input_data)[0]
    survival_probability = rf_model.predict_proba(input_data)[0][0]  # Alive的概率
    class_label = class_mapping[prediction]
    return class_label, survival_probability

# 创建Web应用程序
st.title("6-month survival of NSCLC-BM patients based on Random Forest")
st.sidebar.write("Variables")

Age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
Sex = st.sidebar.selectbox("Sex", options=list(Sex_mapper.keys()))
Race = st.sidebar.selectbox("Race", options=list(Race_mapper.keys()))
Histological_Type = st.sidebar.selectbox("Histological_Type", options=list(Histological_Type_mapper.keys()))
T = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
Liver_metastasis = st.sidebar.selectbox("Liver metastasis", options=list(Liver_metastasis_mapper.keys()))
Radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))
Marital_status = st.sidebar.selectbox("Marital_status", options=list(Marital_status_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        Age=Age,
        Sex=Sex,
        Race=Race,
        Histological_Type=Histological_Type,
        T=T,
        Liver_metastasis=Liver_metastasis,
        Radiation=Radiation,
        Chemotherapy=Chemotherapy,
        Marital_status=Marital_status
    )
    st.write("Predicted Vital Status:", prediction)
    st.write("Probability of 6-month survival is:", f"{probability:.2%}")  # 格式化百分比显示