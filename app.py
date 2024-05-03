import streamlit as st
import pandas as pd
import numpy as np
import pickle

DATASET_PATH = r"heart_2020_cleaned.csv"
LOG_MODEL_PATH = r"logistic_regression.pkl"


def load_dataset() -> pd.DataFrame:
    heart_df = pd.read_csv(DATASET_PATH)
    return heart_df


def user_input_features(heart: pd.DataFrame) -> pd.DataFrame:
    race = st.sidebar.selectbox("Race", options=heart["Race"].unique())
    sex = st.sidebar.selectbox("Sex", options=heart["Sex"].unique())
    age_cat = st.sidebar.selectbox("Age category", options=heart["AgeCategory"].unique())
    if "BMICategory" in heart.columns:
        bmi_cat_options = heart["BMICategory"].unique()
    else:
        bmi_cat_options = ["Underweight", "Normal weight", "Overweight", "Obese"]
    bmi_cat = st.sidebar.selectbox("BMI category", options=bmi_cat_options)
    sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
    gen_health = st.sidebar.selectbox("How can you define your general health?", options=heart["GenHealth"].unique())
    phys_health = st.sidebar.number_input("For how many days during the past 30 days was "
                                          "your physical health not good?", 0, 30, 0)
    ment_health = st.sidebar.number_input("For how many days during the past 30 days was "
                                          "your mental health not good?", 0, 30, 0)
    phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.) "
                                     "in the past month?", options=["No", "Yes"])
    smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in "
                                   "your entire life (approx. 5 packs)?", options=["No", "Yes"])
    alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men) "
                                          "or more than 7 (women) in a week?", options=["No", "Yes"])
    stroke = st.sidebar.selectbox("Did you have a stroke?", options=["No", "Yes"])
    diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking "
                                      "or climbing stairs?", options=["No", "Yes"])
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?", options=heart["Diabetic"].unique())
    asthma = st.sidebar.selectbox("Do you have asthma?", options=["No", "Yes"])
    kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=["No", "Yes"])
    skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=["No", "Yes"])

    features = pd.DataFrame({
        "PhysicalHealth": [phys_health],
        "MentalHealth": [ment_health],
        "SleepTime": [sleep_time],
        "BMICategory": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "Stroke": [stroke],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Race": [race],
        "Diabetic": [diabetic],
        "PhysicalActivity": [phys_act],
        "GenHealth": [gen_health],
        "Asthma": [asthma],
        "KidneyDisease": [kid_dis],
        "SkinCancer": [skin_canc]
    })

    return features


def main():
    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon=":heart:"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    heart = load_dataset()

    col1, col2 = st.columns([1, 3])

    with col1:
        submit = st.button("Predict")

    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
       
        """)

    input_df = user_input_features(heart)

    if submit:
        log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))
        df = pd.concat([input_df, heart], axis=0)
        df = df.drop(columns=["HeartDisease"])
        df.fillna(0, inplace=True)
        df = df[:1]
        
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        
        if prediction == 0:
            st.markdown(f"**The probability that you'll have heart disease is "
                        f"{round(prediction_prob[0][1] * 100, 2)}%. You are healthy!**")
        else:
            st.markdown(f"**The probability that you will have heart disease is "
                        f"{round(prediction_prob[0][1] * 100, 2)}%. It sounds like you are not healthy.**")


if __name__ == "__main__":
    main()
