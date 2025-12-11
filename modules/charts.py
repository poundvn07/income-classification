import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(model, X_test, y_test):
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

    fig.update_layout(height=300, margin=dict(t=30))
    return fig

def plot_roc_curve(log_model, knn_model, X_test, y_test):
    y_proba_log = log_model.predict_proba(X_test)[:, 1]
    y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

    fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_log, y=tpr_log,
                             mode='lines', name="Logistic Regression"))
    fig.add_trace(go.Scatter(x=fpr_knn, y=tpr_knn,
                             mode='lines', name="KNN"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                             mode='lines', name="Random", line=dict(dash="dash")))
    fig.update_layout(height=350)
    return fig
