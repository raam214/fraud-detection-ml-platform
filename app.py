import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG
st.set_page_config(
    page_title="Real-Time Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# ── CUSTOM CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 100%); }
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.3rem;
    }
    .hero-sub { text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #1e2436);
        border-radius: 16px; padding: 1.5rem;
        border: 1px solid #2d3450; text-align: center;
    }
    .approve { color: #00f5a0; font-size: 2rem; font-weight: 900; }
    .review { color: #ffa726; font-size: 2rem; font-weight: 900; }
    .block { color: #ff4757; font-size: 2rem; font-weight: 900; }
    .score-big {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER
st.markdown('<div class="hero-title">🛡️ Real-Time Fraud Detection Platform</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">XGBoost ML engine — real-time transaction risk scoring with APPROVE / REVIEW / BLOCK decisions</div>', unsafe_allow_html=True)

# ── GENERATE DATA


@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'amount': np.random.exponential(200, n),
        'hour': np.random.randint(0, 24, n),
        'day_of_week': np.random.randint(0, 7, n),
        'merchant_category': np.random.randint(0, 10, n),
        'distance_from_home': np.random.exponential(50, n),
        'velocity_24h': np.random.randint(1, 20, n),
        'device_score': np.random.uniform(0, 1, n),
        'location_risk': np.random.uniform(0, 1, n),
        'card_age_days': np.random.randint(1, 2000, n),
        'failed_attempts': np.random.randint(0, 5, n),
    })
    fraud_prob = (
        (df['amount'] > 500).astype(int) * 0.25 +
        (df['hour'] < 6).astype(int) * 0.2 +
        (df['distance_from_home'] > 100).astype(int) * 0.2 +
        (df['velocity_24h'] > 10).astype(int) * 0.15 +
        (df['failed_attempts'] > 2).astype(int) * 0.15 +
        (df['location_risk'] > 0.7).astype(int) * 0.1 +
        np.random.uniform(0, 0.15, n)
    )
    df['is_fraud'] = (fraud_prob > 0.6).astype(int)
    return df

# ── TRAIN MODEL


@st.cache_resource
def train_model(df):
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns.tolist()


df = generate_data()
model, X_test, y_test, feature_names = train_model(df)

# ── DECISION LOGIC


def get_decision(prob):
    if prob < 0.3:
        return "✅ APPROVE", "approve", "#00f5a0"
    elif prob < 0.6:
        return "⚠️ REVIEW", "review", "#ffa726"
    else:
        return "🚫 BLOCK", "block", "#ff4757"


# ── TABS
tab1, tab2, tab3 = st.tabs(
    ["🔍 Transaction Analyser", "📊 Risk Dashboard", "🤖 Model Performance"])

# ════════════════════════════════════
# TAB 1 — TRANSACTION ANALYSER
# ════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Analyse Transaction Risk")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount ($)", 1.0, 10000.0, 250.0)
        hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

    with col2:
        merchant_category = st.selectbox("Merchant Category", list(range(10)),
                                         format_func=lambda x: ['Grocery', 'Electronics', 'Travel', 'Restaurant',
                                                                'Gas', 'Entertainment', 'Healthcare', 'Online',
                                                                'ATM', 'Other'][x])
        distance_from_home = st.slider("Distance from Home (km)", 0, 500, 25)
        velocity_24h = st.slider("Transactions in last 24h", 1, 20, 3)

    with col3:
        device_score = st.slider("Device Trust Score", 0.0, 1.0, 0.8)
        location_risk = st.slider("Location Risk Score", 0.0, 1.0, 0.2)
        card_age_days = st.number_input("Card Age (days)", 1, 2000, 365)
        failed_attempts = st.slider("Failed Attempts", 0, 5, 0)

    if st.button("🚀 Analyse Transaction", use_container_width=True):
        input_data = np.array([[amount, hour, day_of_week, merchant_category,
                                distance_from_home, velocity_24h, device_score,
                                location_risk, card_age_days, failed_attempts]])

        prob = model.predict_proba(input_data)[0][1]
        decision, css_class, color = get_decision(prob)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888; margin-bottom:0.5rem">RISK DECISION</div>
                <div style="color:{color}; font-size:2.2rem; font-weight:900;">{decision}</div>
                <div class="score-big">{prob*100:.1f}%</div>
                <div style="color:#888">Fraud Probability</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Risk factors
        st.markdown("#### 🧠 Risk Factor Analysis")
        factors = {
            'High Amount': 1 if amount > 500 else 0,
            'Odd Hours': 1 if hour < 6 else 0,
            'Far from Home': 1 if distance_from_home > 100 else 0,
            'High Velocity': 1 if velocity_24h > 10 else 0,
            'Failed Attempts': 1 if failed_attempts > 2 else 0,
            'High Location Risk': 1 if location_risk > 0.7 else 0,
            'Low Device Trust': 1 if device_score < 0.3 else 0,
        }
        factor_df = pd.DataFrame({
            'Risk Factor': list(factors.keys()),
            'Triggered': list(factors.values()),
            'Status': ['🔴 HIGH RISK' if v == 1 else '🟢 NORMAL' for v in factors.values()]
        })
        st.dataframe(factor_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════
# TAB 2 — RISK DASHBOARD
# ════════════════════════════════════
with tab2:
    st.markdown("### 📊 Transaction Risk Dashboard")

    # KPIs
    total = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_rate = fraud_count / total * 100

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Transactions", f"{total:,}", "#00d9f5"),
        ("Fraud Detected", f"{fraud_count:,}", "#ff4757"),
        ("Fraud Rate", f"{fraud_rate:.1f}%", "#ffa726"),
        ("Safe Transactions", f"{total-fraud_count:,}", "#00f5a0"),
    ]
    for col, (label, val, color) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888; font-size:0.9rem">{label}</div>
                <div style="color:{color}; font-size:2rem; font-weight:900;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fraud_dist = df['is_fraud'].value_counts().reset_index()
        fraud_dist.columns = ['Type', 'Count']
        fraud_dist['Type'] = fraud_dist['Type'].map(
            {0: 'Legitimate', 1: 'Fraudulent'})
        fig1 = px.pie(fraud_dist, values='Count', names='Type',
                      title='Transaction Distribution',
                      color_discrete_map={'Legitimate': '#00f5a0', 'Fraudulent': '#ff4757'})
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        hourly = df.groupby('hour')['is_fraud'].mean().reset_index()
        fig2 = px.line(hourly, x='hour', y='is_fraud',
                       title='Fraud Rate by Hour of Day',
                       color_discrete_sequence=['#ff4757'])
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(df, x='amount', color=df['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'}),
                            title='Transaction Amount Distribution',
                            color_discrete_map={
                                'Legitimate': '#00f5a0', 'Fraudulent': '#ff4757'},
                            barmode='overlay', nbins=50)
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font_color='white')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig4 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                      title='Feature Importance',
                      color='Importance', color_continuous_scale='viridis')
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font_color='white')
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Performance Metrics")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Accuracy", accuracy_score(y_test, y_pred)),
        ("Precision", precision_score(y_test, y_pred)),
        ("Recall", recall_score(y_test, y_pred)),
        ("F1 Score", f1_score(y_test, y_pred)),
        ("ROC AUC", roc_auc_score(y_test, y_prob)),
    ]
    for col, (name, val) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888">{name}</div>
                <div class="score-big">{val:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'ROC Curve (AUC={roc_auc_score(y_test, y_prob):.3f})',
                              line=dict(color='#00f5a0', width=2)))
    fig5.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              name='Random', line=dict(color='#888', dash='dash')))
    fig5.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                       yaxis_title='True Positive Rate',
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       font_color='white')
    st.plotly_chart(fig5, use_container_width=True)

# ── FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.85rem;'>
    Built by <b>Ram Dukare</b> · Powered by XGBoost · FastAPI · PostgreSQL · Docker · Streamlit
</div>
""", unsafe_allow_html=True)
