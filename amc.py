import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open("feature_col.pkl", 'rb') as file:
    feature_col = joblib.load(file)

st.set_page_config(page_title="🎵 Amazon Music Clustering Dashboard", layout="wide")
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_music_clusters.csv")
    summary = pd.read_csv("amazon_music_fs.csv")
    return df, summary

df, summary = load_data()

st.title("🎧 Amazon Music Clustering Dashboard")
st.markdown("Visualize how songs are grouped into clusters based on audio features like energy, danceability, and tempo.")


st.sidebar.header("🔍 Cluster Selection")
cluster_ids = sorted(df['cluster'].unique())
selected_cluster = st.sidebar.selectbox("Select a Cluster:", cluster_ids)

st.sidebar.header("🎚️ Feature Filters")
features = feature_col

selected_features = st.sidebar.multiselect("Select Features to View:", features, default=['danceability','energy','tempo'])


st.subheader(f"📊 Cluster {selected_cluster} Summary")

col1, col2 = st.columns(2)

with col1:
    cluster_data = df[df['cluster'] == selected_cluster]
    cluster_label = cluster_data['cluster_label'].iloc[0]
    st.subheader(f"🎵{cluster_label}",width="content")
    st.metric("Number of Songs", len(cluster_data))
    for i in selected_features:
     st.metric(f"Average {i}", f"{cluster_data[i].mean():.2f}")
    
with col2:
    st.write("**Feature Means for Cluster:**")
    st.dataframe(summary[summary['cluster'] == selected_cluster].T)


st.subheader("🔥 Feature Comparison Across Clusters")
if 'cluster_label' in summary.columns:
    summary_display = summary.set_index('cluster_label')
else:
    summary_display = summary.set_index('cluster')

numeric_summary = summary_display.select_dtypes(include=['float64', 'int64'])
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_summary, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🎨 PCA Visualization of Clusters")
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(df[features]) 
pca = PCA(n_components=2, random_state=42) 
X_pca = pca.fit_transform(X_scaled) 
plt.figure(figsize=(8,6)) 
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["cluster"], palette="tab10", s=8, alpha=0.7) 
plt.title("PCA 2D Projection of Songs by Cluster") 
plt.xlabel("Principal Component 1") 
plt.ylabel("Principal Component 2") 
st.pyplot(plt)

if 'name_song' in df.columns:
    st.subheader(f"🎶 Top 10 Songs in Cluster {selected_cluster}")
    cluster_data = df[df['cluster'] == selected_cluster]

    display_cols = ['name_song']
    if 'genres' in df.columns:
        display_cols.append('genres')

    display_cols += [col for col in selected_features if col in df.columns]
    sort_col = st.selectbox("Sort by:", selected_features)

    
    # Get top 10 songs by chosen sort feature
    top_songs = cluster_data.nlargest(10, sort_col)
    st.caption(f"Sorted by {sort_col}")

    # Display dataframe
    st.dataframe(top_songs[display_cols].reset_index(drop=True))

else:
    st.warning("⚠️ Song names not found in the dataset. Ensure 'name_song' column exists.")