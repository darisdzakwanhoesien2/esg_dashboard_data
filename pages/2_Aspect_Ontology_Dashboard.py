import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio

# Fix plotly theme
pio.templates.default = "plotly_white"

st.title("🔍 Aspect & Ontology Visualization Dashboard")
st.write("Analyze aspect categories, ontology URIs, sentiment and tone at sentence level.")

# ------------------------------------------------
# 📥 File Upload (shared across pages)
# ------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload ESG CSV", type=["csv"], key="aspect_file")

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

required = {"aspect", "aspect_category", "ontology_uri", "sentiment", "tone"}

if not required.issubset(df.columns):
    st.error(f"Dataset must contain: {required}")
    st.stop()

st.success("File loaded successfully!")
st.dataframe(df.head())

# ==============================================
# 🔹 1 — Aspect Category Frequency
# ==============================================
st.subheader("1️⃣ Aspect Category Distribution")

fig1 = px.bar(
    df["aspect_category"].value_counts().reset_index(),
    x="index",
    y="aspect_category",
    labels={"index": "Aspect Category", "aspect_category": "Count"},
    title="Aspect Category Frequency",
)
st.plotly_chart(fig1, use_container_width=True)

# ==============================================
# 🔹 2 — Ontology URI Frequency
# ==============================================
st.subheader("2️⃣ Ontology URI Distribution")

fig2 = px.bar(
    df["ontology_uri"].value_counts().reset_index(),
    x="index",
    y="ontology_uri",
    labels={"index": "Ontology URI", "ontology_uri": "Count"},
    title="Ontology URI Frequency",
)
st.plotly_chart(fig2, use_container_width=True)

# ==============================================
# 🔹 3 — Sentiment by Aspect Category
# ==============================================
st.subheader("3️⃣ Sentiment by Aspect Category")

sent_aspect = (
    df.groupby(["aspect_category", "sentiment"])
    .size()
    .reset_index(name="count")
)

fig3 = px.bar(
    sent_aspect,
    x="aspect_category",
    y="count",
    color="sentiment",
    barmode="group",
    title="Sentiment Distribution by Aspect Category",
)
st.plotly_chart(fig3, use_container_width=True)

# ==============================================
# 🔹 4 — Tone by Aspect Category
# ==============================================
st.subheader("4️⃣ Tone by Aspect Category")

tone_aspect = (
    df.groupby(["aspect_category", "tone"])
    .size()
    .reset_index(name="count")
)

fig4 = px.bar(
    tone_aspect,
    x="aspect_category",
    y="count",
    color="tone",
    barmode="group",
    title="Tone Distribution by Aspect Category",
)
st.plotly_chart(fig4, use_container_width=True)

# ==============================================
# 🔹 5 — Heatmap: Aspect Category × Sentiment/Tone
# ==============================================
st.subheader("5️⃣ Aspect Category vs Sentiment/Tone Heatmap")

# Pivot for heatmap (sentiment)
pivot_sent = pd.pivot_table(
    df, values="sentence", index="aspect_category", columns="sentiment",
    aggfunc="count", fill_value=0
)

# Pivot for heatmap (tone)
pivot_tone = pd.pivot_table(
    df, values="sentence", index="aspect_category", columns="tone",
    aggfunc="count", fill_value=0
)

# Plot side-by-side
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(pivot_sent, annot=True, cmap="Blues", ax=ax[0])
ax[0].set_title("Sentiment Heatmap")

sns.heatmap(pivot_tone, annot=True, cmap="Greens", ax=ax[1])
ax[1].set_title("Tone Heatmap")

st.pyplot(fig)
