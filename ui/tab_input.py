import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as components
from modules.texts import TEXT

def render_input_tab(df_original, df_data, scaler, num_cols, log_model, knn_model):
    t = TEXT

    st.header("📝 Personal Information")

    col1, col2 = st.columns(2)

    # LEFT
    with col1:
        st.markdown("### Categorical Data")

        country = st.selectbox(
            t["country"],
            ["United-States", "Mexico", "Philippines", "Germany",
             "Canada", "India", "England", "Vietnam", "Other"],
            key="country"
        )

        marital_options = [
            "Never-married", "Married-civ-spouse", "Divorced",
            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ]
        marital_status = st.selectbox(t["Marital_Status"], marital_options, key="marital_status")

        occupation_options = [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
        ]
        occupation = st.selectbox(t["Occupation"], occupation_options, key="occupation")

        workclass_options = [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ]
        workclass = st.selectbox(t["Workclass"], workclass_options, key="workclass")

        sex = st.selectbox(t["Sex"], ["Male", "Female"], key="sex")

        race_options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
        race = st.selectbox(t["Race"], race_options, key="race")

    # RIGHT
    with col2:
        st.markdown("### Numeric Data")

        gain = st.number_input(t["gain"], 0, 99999, 0, key="gain")
        loss = st.number_input(t["loss"], 0, 4356, 0, key="loss")
        age = st.slider(t["age"], 17, 90, 30, key="age")
        hours = st.slider(t["hours"], 1, 80, 40, key="hours")
        edu = st.slider(t["edu"], 1, 16, 10, key="edu")

        education_dict = {
            1: "Preschool", 2: "1st–4th", 3: "5th–6th", 4: "7th–8th",
            5: "9th", 6: "10th", 7: "11th", 8: "12th",
            9: "High School", 10: "Some college", 11: "Associate-voc",
            12: "Associate-acdm", 13: "Bachelors", 14: "Masters",
            15: "Professional degree", 16: "Doctorate"
        }

        st.markdown(
            f"<div class='edu-label'>🎓 {edu} — {education_dict[edu]}</div>",
            unsafe_allow_html=True
        )

    # PROCESS BUTTON
    if st.button(t["predict"], type="primary", use_container_width=True):
        with st.spinner("Processing..."):

            input_dict = {
                "age": age,
                "education.num": edu,
                "capital.gain": gain,
                "capital.loss": loss,
                "hours.per.week": hours,
                "native.country_US": 1 if country == "United-States" else 0
            }

            # Dummies
            for m in marital_options[1:]:
                input_dict[f"marital.status_{m}"] = 1 if marital_status == m else 0

            for o in occupation_options[1:]:
                input_dict[f"occupation_{o}"] = 1 if occupation == o else 0

            for w in workclass_options[1:]:
                input_dict[f"workclass_{w}"] = 1 if workclass == w else 0

            for r in race_options[1:]:
                input_dict[f"race_{r}"] = 1 if race == r else 0

            input_dict["sex_Male"] = 1 if sex == "Male" else 0

            # Create DF
            input_df = pd.DataFrame([input_dict])
            for col in df_data.columns:
                if col not in input_df:
                    input_df[col] = 0
            input_df = input_df[df_data.drop("income", axis=1).columns]

            # Scale
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            # Model select
            model = log_model if st.session_state.get("model_choice") == "Logistic Regression" else knn_model
            pred = model.predict(input_df)[0]

            st.session_state.prediction = pred
            st.session_state.input_data = input_df

            # Auto-switch tab
            js = """
            <script>
                var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length > 1) { tabs[1].click(); }
            </script>
            """
            time.sleep(0.1)
        components.html(js, height=0)
