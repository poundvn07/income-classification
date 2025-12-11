import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# PAGE CONFIG

st.set_page_config(
    page_title="Income Predictor",
    page_icon="💵",
    layout="wide",
)

# CSS

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stButton>button {
    width: 100%; 
    border-radius: 8px; 
    height: 3em; 
    font-weight: 600;
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
    font-size: 16px;
    border: none;
    padding: 12px 0;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #5AA0FF, #3A7CE0);
    color: white;
}

.metric-container {
    background: #E8EBF0;
    padding: 18px 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    border-left: 4px solid #4A90E2;
}
.metric-title {
    font-size: 15px;
    font-weight: 600;
    color: #333;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #2b2b2b;
    margin: 0;
}
.metric-diff {
    font-size: 13px;
    font-weight: 600;
    margin-top: -2px;
    display: inline-block;
}
.metric-diff.positive { color: #2ecc71; }
.metric-diff.negative { color: #e74c3c; }
.progress-container {
    margin-top: 8px;
    margin-bottom: 2px;
}
progress {
    accent-color: #4A90E2;
    border-radius: 10px;
    overflow: hidden;
    height: 8px;
}
.percentile-text {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
    font-style: italic;
}

.cat-card {
    background: #E8EBF0;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: 0.2s;
    border: 1px solid #e6ecf5;
}
.cat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}
.cat-title {
    font-size: 14px;
    font-weight: 600;
    color: #444;
}
.cat-value {
    font-size: 18px;
    font-weight: 700;
    color: #2c3e50;
    margin-top: 5px;
}
.cat-share {
    font-size: 12px;
    color: #777;
    margin-top: 4px;
}

.input-card {
    background: #F3F4F6;
    padding: 25px 25px 15px 25px;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}
.field-label {
    font-weight: 600;
    color: #333;
    font-size: 15px;
    margin-bottom: 4px;
}
.edu-label {
    margin-top: -8px;
    font-size: 13px;
    color: #777;
    font-style: italic;
}
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

TEXT = {
    "title": "Income Insights and Prediction",
    "sidebar_info": "AI system predicting personal income based on census data (education, demographics and occupation).",
    "tabs": ["📥 Input", "📤 AI Result", "📊 Data & Model Analysis"],
    "model_select": "Select Model",
    "input_header": "Enter Information",
    "age": "Age",
    "Marital_Status": "Marial Status",
    "Occupation": "Occupation",
    "Workclass": "Workclass",
    "Sex": "Gender",
    "Race": "Race",
    "edu": "Education Level (1–16)",
    "hours": "Hours per week",
    "gain": "Capital Gain (0-99999)",
    "loss": "Capital Loss  (0-4356)",
    "country": "Country",
    "predict": "Analyze & Predict",
    "not_ready": "Please enter your information first.",
    "compare": "User vs Dataset",
    "income_dist": "Income Distribution",
    "hours_chart": "Working Hours by Income",
    "age_chart": "Age Distribution by Income",
    "result_title": "Prediction Result",
    "high": "🟢 Predicted income: > 50K",
    "low": "🔴 Predicted income: ≤ 50K",
    "Age": "Age",
    "Capital Gain": "Capital Gain",
    "Capital Loss": "Capital Loss",
    "Hours per Week": "Hours per Week",
    "Education Num": "Education Level",
    "vs Avg": "vs Avg",
    "Higher than": "Higher than"
}

# LOAD & TRAIN MODELS

@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/adult.csv", na_values="?", skipinitialspace=True)

    df = df.drop(["fnlwgt", "education", "relationship"], axis=1)
    df_original = df.copy()
    df["native.country"] = df["native.country"].apply(lambda x: "US" if x == "United-States" else "Non-US")
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["income"] = le.fit_transform(df["income"])

    df = pd.get_dummies(df, drop_first=True)

    num_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=15)
    knn_model.fit(X_train, y_train)

    return df, df_original, scaler, log_model, knn_model, X_test, y_test, num_cols


df_data, df_original, scaler, log_model, knn_model, X_test, y_test, num_cols = load_and_train()

# SIDEBAR

with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSo5DkXoUd3qoU9bMA0jlVZmfiv1yFwZqop8w&s", width=200)

    st.title(TEXT["title"])
    st.markdown("---")

    t = TEXT
    
    model_choice = st.selectbox(
        t["model_select"],
        ["Logistic Regression", "K-Nearest Neighbours"]
    )

    st.info(t["sidebar_info"])

# TABS

tab1, tab2, tab3 = st.tabs(t["tabs"])

# STATE
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "input_data" not in st.session_state:
    st.session_state.input_data = None

# TAB 1

with tab1:
    st.header("📝 Personal Infomation")
    col1, col2 = st.columns(2)

 
    # LEFT COLUMN (col1)
 
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


    # RIGHT COLUMN (col2)
 
    with col2:
        st.markdown("### Numeric Data")

        gain = st.number_input(t["gain"], 0, 99999, 0, key="gain")
        loss = st.number_input(t["loss"], 0, 4356, 0, key="loss")
        age = st.slider(t["age"], 17, 90, 30, key="age")
        hours = st.slider(t["hours"], 1, 80, 40, key="hours")
        edu = st.slider(t["edu"], 1, 16, 10, key="edu")

        education_dict = {
            1: "Preschool", 2: "Grade 1st–4th", 3: "Grade 5th–6th",
            4: "Grade 7th–8th", 5: "Grade 9th", 6: "Grade 10th",
            7: "Grade 11th", 8: "Grade 12th", 9: "High School Graduate",
            10: "Some-college", 11: "Associate (vocational)",
            12: "Associate (academic)", 13: "Bachelors",
            14: "Masters", 15: "Professional degree", 16: "Doctorate"
        }

        st.markdown(
            f"<div class='edu-label'>🎓 {edu} — {education_dict[edu]}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


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

            for m in marital_options[1:]:
                input_dict[f"marital.status_{m}"] = 1 if marital_status == m else 0

            for o in occupation_options[1:]:
                input_dict[f"occupation_{o}"] = 1 if occupation == o else 0

            for w in workclass_options[1:]:
                input_dict[f"workclass_{w}"] = 1 if workclass == w else 0

            for r in race_options[1:]:
                input_dict[f"race_{r}"] = 1 if race == r else 0

            input_dict["sex_Male"] = 1 if sex == "Male" else 0

            input_df = pd.DataFrame([input_dict])
            for col in X_test.columns:
                input_df[col] = input_df.get(col, 0)
            input_df = input_df[X_test.columns]
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            model = log_model if model_choice == "Logistic Regression" else knn_model
            pred = model.predict(input_df)[0]

            st.session_state.prediction = pred
            st.session_state.input_data = input_df

            js = """
            <script>
                var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length > 1) {
                    tabs[1].click();
                }
            </script>
            """
            time.sleep(0.2)
        components.html(js, height=0)


# TAB 2
with tab2:
    if st.session_state.input_data is None:
        st.info(t["not_ready"])
    else:
        st.header('Comparative Analysis 📊')

        st.subheader('Numerical Data')
        def render_metric(name, user_val, avg_val, percentile=None):
            diff = user_val - avg_val
            positive = diff >= 0
            diff_class = "positive" if positive else "negative"
            arrow = "↑" if positive else "↓"
            diff_text = f"{arrow} {abs(diff):.2f} vs Avg"
            progress_value = percentile if percentile is not None else user_val
            progress_max = 100 if percentile is not None else max(user_val, avg_val) * 1.1

            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-title">{name}</p>
                <p class="metric-value">{user_val}</p>
                <span class="metric-diff {diff_class}">{diff_text}</span>
                <div class="progress-container">
                    <progress value="{progress_value}" max="{progress_max}"></progress>
                </div>
                {f'<p class="percentile-text">Higher than {percentile:.1f}%</p>' if percentile else ''}
            </div>
            """, unsafe_allow_html=True)

        user_metrics = {
            "Age": age,
            "Capital Gain": gain,
            "Capital Loss": loss,
            "Hours per week": hours,
            "Education Num": edu
        }

        avg_metrics = {
            "Age": df_data['age'].mean(),
            "Capital Gain": df_data['capital.gain'].mean(),
            "Capital Loss": df_data['capital.loss'].mean(),
            "Hours per week": df_data['hours.per.week'].mean(),
            "Education Num": df_data['education.num'].mean()
        }

        def calc_percentile(col, user_val, df):
            return (df[col] < user_val).mean() * 100

        percentiles = {
            "Age": calc_percentile("age", user_metrics["Age"], df_data),
            "Capital Gain": calc_percentile("capital.gain", user_metrics["Capital Gain"], df_data),
            "Capital Loss": calc_percentile("capital.loss", user_metrics["Capital Loss"], df_data),
            "Hours per week": calc_percentile("hours.per.week", user_metrics["Hours per week"], df_data),
            "Education Num": calc_percentile("education.num", user_metrics["Education Num"], df_data),
        }

        cols = st.columns(2)
        for i, (key, user_val) in enumerate(user_metrics.items()):
            with cols[i % 2]:
                render_metric(key, user_val, avg_metrics[key], percentiles[key])

        st.subheader('Categorical Data')

        def category_percentage(column_name, value):
            total = len(df_original)
            count = len(df_original[df_original[column_name] == value])
            return round((count / total) * 100, 1)

        categorical_items = [
            ("Gender", sex, "sex"),
            ("Country", country, "native.country"),
            ("Marital Status", marital_status, "marital.status"),
            ("Occupation", occupation, "occupation"),
            ("Workclass", workclass, "workclass"),
            ("Race", race, "race"),
        ]

        cols = st.columns(3)
        for i, (label, value, col_name) in enumerate(categorical_items):
            with cols[i % 3]:
                share = category_percentage(col_name, value)
                st.markdown(f"""
                <div class="cat-card">
                    <div class="cat-title">{label}</div>
                    <div class="cat-value">{value}</div>
                    <div class="cat-share">Shared by {share}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.subheader(t["result_title"])

        if st.session_state.prediction == 1:
            st.success(t["high"])
        else:
            st.error(t["low"])
        st.subheader("📘 Dataset")
        st.dataframe(df_original, use_container_width=True)

# TAB 3
with tab3:

    if st.session_state.prediction is None:
        st.info(t["not_ready"])
    else:     
        st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba)
            }

        log_metrics = evaluate_model(log_model, X_test, y_test)
        knn_metrics = evaluate_model(knn_model, X_test, y_test)

        metrics_df = pd.DataFrame([log_metrics, knn_metrics], index=["Logistic Regression", "KNN"])

        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        # BAR CHART

        metrics_melt = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        metrics_melt.rename(columns={"index": "Model"}, inplace=True)

        fig_bar = px.bar(
            metrics_melt,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            height=380,
            text_auto=".3f",
            color_discrete_sequence=["#0074D9", "#7FDBFF"]
        )
        fig_bar.update_layout(title="", margin=dict(t=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # CONFUSION MATRIX

        st.markdown('<div class="section-title">🧩 Confusion Matrix</div>', unsafe_allow_html=True)

        def plot_cm(model, title):
            cm = confusion_matrix(y_test, model.predict(X_test))
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred ≤50K", "Pred >50K"],
                y=["Actual ≤50K", "Actual >50K"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False
            ))
            fig.update_layout(
                title=title,
                height=330,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig

        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(plot_cm(log_model, "Logistic Regression"), use_container_width=True)
        with colB:
            st.plotly_chart(plot_cm(knn_model, "K-Nearest Neighbors"), use_container_width=True)

        # ROC

        st.markdown('<div class="section-title">📈 ROC-AUC Comparison</div>', unsafe_allow_html=True)

        y_proba_log = log_model.predict_proba(X_test)[:, 1]
        y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

        fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
        fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_log, y=tpr_log,
                                    mode='lines',
                                    name=f"Logistic Regression (AUC={auc(fpr_log, tpr_log):.3f})",
                                    line=dict(color="#0074D9", width=3)))
        fig_roc.add_trace(go.Scatter(x=fpr_knn, y=tpr_knn,
                                    mode='lines',
                                    name=f"KNN (AUC={auc(fpr_knn, tpr_knn):.3f})",
                                    line=dict(color="#2ECC40", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                    mode='lines',
                                    name='Random Baseline',
                                    line=dict(color='gray', dash='dash')))

        fig_roc.update_layout(
            height=400,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=20, b=30)
        )

        st.plotly_chart(fig_roc, use_container_width=True)

        # INCOME DISTRIBUTION
        st.subheader("📌 Income Distribution")

        # Lấy phân phối income
        income_counts = df_data["income"].value_counts().reset_index()
        income_counts.columns = ["income", "count"]
        income_counts["income"] = income_counts["income"].map({0: "≤50K", 1: ">50K"})

        # Biểu đồ
        fig_income = px.bar(
            income_counts,
            x="income",
            y="count",
            text="count",
            color="income",
            color_discrete_map={
                "≤50K": "#3498db",
                ">50K": "#2ecc71"
            },
            height=500
        )

        fig_income.update_traces(
            marker=dict(opacity=0.9),
            textposition="outside",
            width=0.45
        )

        fig_income.update_layout(
            xaxis_title="Income Group",
            yaxis_title="Count",
            bargap=0.25,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(size=14)
        )

        st.plotly_chart(fig_income, use_container_width=True)
