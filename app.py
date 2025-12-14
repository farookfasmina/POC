import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import preprocess

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="AI Study Group Matching", layout="wide")

# ==================================================
# UI POLISH (SAFE – NO LOGIC CHANGE)
# ==================================================
st.markdown("""
<style>

/* Remove Streamlit decorations */
div[data-testid="stDecoration"],
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Remove top gap */
.block-container {
    padding-top: 0.5rem !important;
}

/* DARK MODE CARD */
.card {
    background: linear-gradient(180deg, #0f172a, #020617);
    padding: 1.5rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}

/* Section titles */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #4ade80;
    margin-bottom: 0.6rem;
}

/* Login card */
.login-card {
    max-width: 420px;
    margin: auto;
    padding: 2rem;
    border-radius: 16px;
}
 .chat-container {
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    background: #f9fafb;
    border-radius: 10px;
    margin-bottom: 10px;
}

.chat-bubble {
    padding: 8px 12px;
    border-radius: 14px;
    margin-bottom: 8px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.4;
}

.chat-you {
    background-color: #DCF8C6;
    margin-left: auto;
    text-align: right;
}

.chat-other {
    background-color: #E5E7EB;
    margin-right: auto;
    text-align: left;

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ==================================================
# SESSION STATE
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

# ==================================================
# AUTHENTICATION (DEMO)
# ==================================================
if not st.session_state.logged_in:

    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(180deg, #0f172a, #020617);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card login-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>🎓 AI Study Group Matching</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sign in to continue</p>", unsafe_allow_html=True)

    role = st.selectbox("Select role", ["Student", "Admin"])
    if st.button("🔐 Login", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.role = role
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.success(f"Logged in as: {st.session_state.role}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Student Matching",
        "Visual Insights",
        "Model & Data Insights",
        "Admin Dashboard",
        "About"
    ]
)

# ==================================================
# MATCHING SCORE FUNCTION
# ==================================================
def calculate_matching_score(student, group):
    score = 0
    total = 4

    def overlap(a, b):
        return len(set(a.split(", ")) & set(b.split(", "))) > 0

    if overlap(student["Subjects"], ", ".join(group["Subjects"])):
        score += 1
    if overlap(student["Time_Slots"], ", ".join(group["Time_Slots"])):
        score += 1
    if overlap(student["Communication_Methods"], ", ".join(group["Communication_Methods"])):
        score += 1
    if student["Year_of_Study"] in group["Year_of_Study"].values:
        score += 1

    return int((score / total) * 100)

# ==================================================
# HOME
# ==================================================
# ==================================================
# HOME
# ==================================================
if page == "Home":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        "<h1>🎓 AI-Powered Adaptive Study Group Matching</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:1.05rem;'>"
        "Intelligently forms <b>study buddies or study groups</b> "
        "based on <b>student-selected preferences</b> using "
        "<b>AI-driven clustering</b>.</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("🤖 AI Model", "K-Means Clustering")
    c2.metric("👥 Group Size", "User Selected")
    c3.metric("☁ Deployment", "Cloud Ready")

    st.markdown("---")

    st.markdown("### ✨ Key Capabilities")
    st.write("• Preference-based study group formation")
    st.write("• Real-time clustering and visualization")
    st.write("• Admin analytics & KPI dashboard")
    st.write("• Scalable and cloud-ready design")

    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# STUDENT MATCHING
# ==================================================
if page == "Student Matching":

    # Header
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👩‍🎓 Student Matching</div>', unsafe_allow_html=True)
    st.write(
        "Enter your preferences below. "
        "The system will use AI to form the most compatible study buddy or group for you."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Progress
    st.progress(0.6, text="Preference Selection in Progress")

    year = st.selectbox("Year of Study", ['1st Year','2nd Year','3rd Year','4th Year or above'])
    program = st.selectbox("Academic Program", ['IT','SE','CS','Business','Engineering'])
    subjects = st.multiselect("Subjects", ['AI','Networking','Databases','Software Quality','Research','Algorithms'])
    days = st.multiselect("Available Days", ['Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday'])
    times = st.multiselect("Time Slots", ['Morning','Afternoon','Evening','Night'])
    comms = st.multiselect("Communication Methods", ['WhatsApp','Zoom','Teams','Telegram','Face-to-face'])
    competency = st.slider("Academic Competency (1–5)", 1, 5, 3)

    group_type = st.radio(
        "Preferred Study Group Type",
        ["Buddy (2 students)", "Small Group (3–4 students)", "Large Group (5–6 students)"]
    )

    GROUP_SIZE = 2 if group_type.startswith("Buddy") else 4 if group_type.startswith("Small") else 6

    # Initialize chat state safely
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if st.button("🔍 Generate Optimal Study Group", use_container_width=True):

        # Reset chat when new group is generated
        st.session_state.chat = []

        if not subjects:
            st.warning("Please select at least one subject to generate a meaningful match.")
            st.stop()

        with st.spinner("Analyzing preferences and forming the best group for you..."):
            time.sleep(1.2)

        student = pd.DataFrame([{
            "Year_of_Study": year,
            "Academic_Program": program,
            "Subjects": ", ".join(subjects),
            "Available_Days": ", ".join(days),
            "Time_Slots": ", ".join(times),
            "Communication_Methods": ", ".join(comms),
            "Competency": competency,
            "Satisfaction": 3
        }])

        X_student = preprocess(student).reindex(columns=features, fill_value=0)
        cluster = int(kmeans.predict(scaler.transform(X_student))[0])

        st.success(f"✅ Assigned to Cluster {cluster}")
        st.metric("Selected Group Size", f"{GROUP_SIZE} Students")

        df = pd.read_csv("synthetic_students.csv")
        X_all = preprocess(df).reindex(columns=features, fill_value=0)
        df["Cluster"] = kmeans.predict(scaler.transform(X_all))

        group = df[df["Cluster"] == cluster].sample(
            min(GROUP_SIZE - 1, len(df[df["Cluster"] == cluster])),
            random_state=42
        )

        group = pd.concat([group, student], ignore_index=True)

        st.subheader("📌 Your Study Group")
        st.dataframe(group)

        st.download_button(
            "⬇️ Download My Study Group (CSV)",
            group.to_csv(index=False),
            file_name="study_group.csv",
            mime="text/csv"
        )

        score = calculate_matching_score(student.iloc[0], group.iloc[:-1])
        st.subheader("🎯 Matching Strength")
        st.progress(score / 100)
        st.metric("Matching Score", f"{score}%")

        if score >= 80:
            st.success("🟢 High Compatibility Group")
        elif score >= 50:
            st.info("🟡 Moderate Compatibility Group")
        else:
            st.warning("🔴 Low Compatibility – consider adjusting preferences")

        # ================================
        # 🧠 WHY THIS GROUP? (AI EXPLANATION)
        # ================================
        st.subheader("🧠 Why this group?")

        reasons = []

        if any(s in ", ".join(group["Subjects"]) for s in subjects):
            reasons.append("📘 Shared academic subjects with group members")

        if any(t in ", ".join(group["Time_Slots"]) for t in times):
            reasons.append("⏰ Compatible study time availability")

        if any(c in ", ".join(group["Communication_Methods"]) for c in comms):
            reasons.append("💬 Preferred communication methods match")

        if year in group["Year_of_Study"].values:
            reasons.append("🎓 Similar academic year among group members")

        reasons.append(f"👥 Matches your preferred group size ({GROUP_SIZE} students)")
        reasons.append(f"🧠 AI clustering placed you in Cluster {cluster}")

        for r in reasons:
            st.write("•", r)

        # ================================
        # 💬 MOCK REAL-TIME GROUP CHAT
        # ================================
        st.subheader("💬 Group Chat (Prototype)")

        msg = st.text_input("Type a message")

        if st.button("Send"):
            if msg:
                st.session_state.chat.append(("You", msg))

        for sender, message in st.session_state.chat:
            st.markdown(f"**{sender}:** {message}")

        # ================================
        # 📊 CLUSTER VISUALIZATION
        # ================================
        st.subheader("📊 Cluster Visualization")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(scaler.transform(X_all))
        student_pca = pca.transform(scaler.transform(X_student))

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"], cmap="tab10", alpha=0.5)
        ax.scatter(student_pca[0,0], student_pca[0,1], c="red", s=150, label="You")
        ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.legend()
        st.pyplot(fig)


# ==================================================
# VISUAL INSIGHTS (UNCHANGED)
# ==================================================
if page == "Visual Insights":
    st.title("📊 Visual Insights")

    df = pd.read_csv("synthetic_students.csv")
    X = preprocess(df).reindex(columns=features, fill_value=0)
    df["Cluster"] = kmeans.predict(scaler.transform(X))

    st.plotly_chart(px.pie(df, names="Cluster", hole=0.4, title="Cluster Distribution"),
                     use_container_width=True)

    prog_counts = df["Academic_Program"].value_counts().reset_index()
    prog_counts.columns = ["Academic_Program", "count"]

    st.plotly_chart(
        px.bar(prog_counts, x="Academic_Program", y="count",
               title="Academic Program Distribution", color="Academic_Program"),
        use_container_width=True
    )

    crosstab = pd.crosstab(df["Cluster"], df["Academic_Program"])
    st.plotly_chart(
        px.imshow(crosstab, text_auto=True, aspect="auto",
                  title="Cluster vs Academic Program Heatmap"),
        use_container_width=True
    )

# ==================================================
# MODEL & DATA INSIGHTS (UNCHANGED)
# ==================================================
if page == "Model & Data Insights":
    st.title("📊 Model & Data Insights")

    df = pd.read_csv("synthetic_students.csv")
    X = preprocess(df).reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X)
    df["Cluster"] = kmeans.predict(X_scaled)

    sizes = df["Cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["Cluster", "Students"]

    st.plotly_chart(px.bar(sizes, x="Cluster", y="Students",
                           title="Cluster Size Distribution"),
                    use_container_width=True)

    pca_tmp = PCA(n_components=2)
    pca_tmp.fit(X_scaled)
    var = pca_tmp.explained_variance_ratio_ * 100

    st.plotly_chart(px.bar(x=["PC1","PC2"], y=var,
                           labels={"x":"Component","y":"Variance (%)"},
                           title="PCA Explained Variance"),
                    use_container_width=True)

    profile = df.groupby("Cluster")[["Competency","Satisfaction"]].mean()
    st.plotly_chart(px.imshow(profile, text_auto=".2f",
                              title="Cluster Profile Heatmap"),
                    use_container_width=True)

    inertia = []
    for k in range(2,8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    st.plotly_chart(px.line(x=list(range(2,8)), y=inertia,
                            title="Elbow Method"),
                    use_container_width=True)

    sil = silhouette_score(X_scaled, df["Cluster"])
    st.metric("Silhouette Score", f"{sil:.3f}")

# ==================================================
# ADMIN DASHBOARD
# ==================================================
if page == "Admin Dashboard":

    # ================================
    # LOAD DATA & CLUSTERS (REQUIRED)
    # ================================
    df = pd.read_csv("synthetic_students.csv")
    X_admin = preprocess(df).reindex(columns=features, fill_value=0)
    df["Cluster"] = kmeans.predict(scaler.transform(X_admin))

    # ================================
    # 📌 KPI CARDS
    # ================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3)

    kpi1.metric("👥 Total Students", len(df))
    kpi2.metric("🧠 AI Clusters (K)", kmeans.n_clusters)
    kpi3.metric("🎓 Avg Competency", f"{df['Competency'].mean():.2f}")

    cluster_counts = df["Cluster"].value_counts()

    kpi4, kpi5, kpi6 = st.columns(3)

    kpi4.metric("😊 Avg Satisfaction", f"{df['Satisfaction'].mean():.2f}")
    kpi5.metric("📈 Largest Cluster", int(cluster_counts.max()))
    kpi6.metric("📉 Smallest Cluster", int(cluster_counts.min()))

    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # 🧑‍💼 ADMIN OVERVIEW
    # ================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧑‍💼 Admin Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Clusters (K)", kmeans.n_clusters)
    col3.metric("Most Common Program", df["Academic_Program"].mode()[0])

    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # 📊 ADMIN ANALYTICS
    # ================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Admin Analytics</div>', unsafe_allow_html=True)

    # Cluster size distribution
    cluster_sizes = df["Cluster"].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["Cluster", "Students"]

    fig_cluster = px.bar(
        cluster_sizes,
        x="Cluster",
        y="Students",
        title="Cluster Size Distribution",
        color="Cluster"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Program distribution per cluster
    program_cluster = pd.crosstab(df["Cluster"], df["Academic_Program"])

    fig_heatmap = px.imshow(
        program_cluster,
        text_auto=True,
        aspect="auto",
        title="Academic Program Distribution per Cluster"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Cluster performance
    performance = df.groupby("Cluster")[["Competency", "Satisfaction"]].mean().reset_index()

    fig_perf = px.bar(
        performance,
        x="Cluster",
        y=["Competency", "Satisfaction"],
        barmode="group",
        title="Average Competency & Satisfaction per Cluster"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# ABOUT
# ==================================================
if page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ️ About the System</div>', unsafe_allow_html=True)

    st.write("""
    **AI-Powered Adaptive Study Group Matching System**

    Uses unsupervised learning to dynamically form study groups
    based on student preferences.
    """)

    st.markdown("**Evaluation Metrics:**")
    st.write("- Elbow Method")
    st.write("- Silhouette Score")

    st.caption("System Status: Active | Model Version: v1.0")

    st.markdown('</div>', unsafe_allow_html=True)

