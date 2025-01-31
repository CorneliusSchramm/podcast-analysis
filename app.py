import streamlit as st
import plotly.express as px
import pandas as pd
import spacy
import numpy as np
from collections import Counter, defaultdict
import umap
import hdbscan
import plotly.graph_objects as go
from typing import Optional
from spacy.lang.de.stop_words import STOP_WORDS as de_stop_words

# Set page to wide mode
st.set_page_config(layout="wide")

# Load spacy model at app startup
@st.cache_resource
def load_spacy_model():
    """Loads the SpaCy model with caching.

    Returns:
    -------
        spacy.lang: The loaded SpaCy model.

    """
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.error("SpaCy model 'en_core_web_md' not found. Please download it using:\n"
                "python -m spacy download en_core_web_md")
        st.stop()

# Load your data
@st.cache_data
def load_data():
    """Loads and caches the DataFrame.

    Returns:
    -------
        pd.DataFrame: The loaded DataFrame.

    """
    return pd.read_csv('data/full_df.csv')

def create_timeline_plot(df):
    # Convert to datetime
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df_2024 = df[df['publish_date'].dt.year == 2024]
    
    # Count snips per episode
    snips_per_episode = df_2024.groupby(['publish_date', 'show', 'episode_title']).size().reset_index(name='snip_count')
    
    # Create the plot
    fig = px.scatter(snips_per_episode,
                    x='publish_date', 
                    y='show',
                    color='show',
                    size='snip_count',
                    hover_data=['episode_title', 'snip_count'],
                    title='Timeline of Snips by Show (2024)',
                    template='plotly_dark')
    
    fig.update_layout(
        height=800,
        width=2000  # Added width parameter
    )
    fig.update_traces(showlegend=False)
    return fig

def create_show_keyword_visualization(
    df: pd.DataFrame,
    min_freq: int = 3,
    dimensionality: int = 3,
    method: str = 'UMAP',
    clustering: bool = True,
    cluster_selection_method: str = 'eom',
    max_words_per_show: Optional[int] = None
) -> None:
    """
    Creates an interactive visualization of keywords clustered by show based on semantic meaning.
    
    Parameters:
    - df: pd.DataFrame containing at least 'show' and 'summary' columns.
    - min_freq: Minimum total frequency for a word to be included.
    - dimensionality: 2 for 2D plot, 3 for 3D plot.
    - method: Dimensionality reduction method ('UMAP', 't-SNE', 'MDS').
    - clustering: Whether to apply HDBSCAN clustering to define dedicated regions.
    - cluster_selection_method: Method for HDBSCAN clustering ('eom' or 'leaf').
    - max_words_per_show: If set, only keep the top N words per show (by that word's freq in the show).
    """
    if dimensionality not in [2, 3]:
        raise ValueError("dimensionality must be either 2 or 3")
    
    if method not in ['MDS', 't-SNE', 'UMAP']:
        raise ValueError("method must be one of 'MDS', 't-SNE', or 'UMAP'")
    
    # Load a SpaCy model with word vectors
    try:
        nlp = spacy.load("en_core_web_md")  # Use 'en_core_web_lg' for larger vectors
    except OSError:
        raise OSError("SpaCy model 'en_core_web_md' not found. Please download it using:\n"
                      "python -m spacy download en_core_web_md")
    
    # Gather all show names
    all_shows = sorted(df['show'].dropna().unique())
    
    # Combine English/German stopwords and any domain-specific ones
    custom_stops = {
        "people", "thing", "year", "lot", "point", "das", "man", "der", "way",
        "stuff", "sort", "mal", "speaker", "time", "going", "yeah", "gonna",
        "think", "know", "mean", "actually", "right", "really"
    }
    custom_stops.update(de_stop_words)
    for w in custom_stops:
        nlp.vocab[w].is_stop = True
    
    # Count frequencies
    show_word_freq = defaultdict(Counter)
    total_word_freq = Counter()
    for show_name in all_shows:
        show_text = ' '.join(df.loc[df['show'] == show_name, 'summary'].dropna())
        doc = nlp(show_text)
        keywords = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ['NOUN', 'PROPN']
            and not token.is_stop
            and len(token.text) > 2
            and token.has_vector  # Ensure the token has a vector representation
        ]
        show_word_freq[show_name].update(keywords)
        total_word_freq.update(keywords)
    
    # Filter out words below min_freq overall
    common_words = {w for w, c in total_word_freq.items() if c >= min_freq}
    if not common_words:
        print(f"No words after min_freq={min_freq} filtering.")
        return
    
    # Prepare lists to hold data for each word
    words = []
    word_vectors = []
    dominant_shows = []
    hover_texts = []
    word_total_freq = []
    
    for word in sorted(common_words):
        # Skip if no vector
        if not nlp.vocab.has_vector(word):
            continue
        vector = nlp.vocab[word].vector
        
        # Compute total freq + dominant show
        word_total = total_word_freq[word]
        show_counts = {s: show_word_freq[s][word] for s in all_shows}
        show_max = max(show_counts, key=show_counts.get)
        
        words.append(word)
        word_vectors.append(vector)
        word_total_freq.append(word_total)
        dominant_shows.append(show_max)
        
        # Build hover text for frequencies
        nonzero = [(s, v) for s, v in show_counts.items() if v > 0]
        nonzero.sort(key=lambda x: x[1], reverse=True)
        hover_html = (
            f'<b style="color: black">{word}</b><br>' + 
            '<span style="color: black">' + 
            "<br>".join(f"{s}: {v}" for s, v in nonzero) +
            '</span>'
        )
        hover_texts.append(hover_html)
    
    if not words:
        print("No words to display after vector checks. Adjust 'min_freq' or check data.")
        return
    
    # -------------------------------
    # NEW: limit to top N words/show
    # -------------------------------
    if max_words_per_show is not None and max_words_per_show > 0:
        # Group word indices by their dominant show
        show_to_word_info = defaultdict(list)
        for i, w in enumerate(words):
            show_name = dominant_shows[i]
            freq_in_that_show = show_word_freq[show_name][w]
            show_to_word_info[show_name].append((freq_in_that_show, i))
        
        # For each show, keep only top N by freq in that show
        keep_indices = set()
        for show_name, freq_and_inds in show_to_word_info.items():
            freq_and_inds.sort(key=lambda x: x[0], reverse=True)  # sort by freq desc
            top_n = freq_and_inds[:max_words_per_show]
            for _, idx in top_n:
                keep_indices.add(idx)
        
        # Filter out everything else
        if not keep_indices:
            print(f"No words left after limiting to max_words_per_show={max_words_per_show}.")
            return
        
        # Rebuild the final arrays
        words = [words[i] for i in range(len(words)) if i in keep_indices]
        word_vectors = [word_vectors[i] for i in range(len(word_vectors)) if i in keep_indices]
        word_total_freq = [word_total_freq[i] for i in range(len(word_total_freq)) if i in keep_indices]
        dominant_shows = [dominant_shows[i] for i in range(len(dominant_shows)) if i in keep_indices]
        hover_texts = [hover_texts[i] for i in range(len(hover_texts)) if i in keep_indices]
    
    # Convert to arrays
    word_vectors = np.array(word_vectors, dtype=float)
    word_total_freq = np.array(word_total_freq, dtype=float)
    
    # Normalize vectors (helps with similarity/distance)
    word_vectors_norm = word_vectors / np.linalg.norm(word_vectors, axis=1, keepdims=True)
    
    # Dimensionality reduction
    if method == 'MDS':
        from sklearn.manifold import MDS
        similarities = cosine_similarity(word_vectors_norm)
        dist_matrix = 1 - similarities
        reducer = MDS(n_components=dimensionality, dissimilarity='precomputed', random_state=42)
        coords = reducer.fit_transform(dist_matrix)
    elif method == 't-SNE':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=dimensionality,
            metric='cosine',
            random_state=42,
            perplexity=30,
            n_iter=1000,
            init='random'  # required if metric='cosine'
        )
        coords = reducer.fit_transform(word_vectors_norm)
    elif method == 'UMAP':
        reducer = umap.UMAP(
            n_components=dimensionality,
            metric='cosine',
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        coords = reducer.fit_transform(word_vectors_norm)
    
    # Optional: Clustering with HDBSCAN
    if clustering:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric='euclidean',
            cluster_selection_method=cluster_selection_method
        )
        cluster_labels = clusterer.fit_predict(coords)
        # Replace -1 with 'Noise'
        cluster_labels = np.where(cluster_labels == -1, 'Noise', cluster_labels)
    else:
        cluster_labels = None
    
    # Scale coords
    scaling_factor = 20
    coords_scaled = coords * scaling_factor
    
    # Slight random jitter
    jitter_strength = 0.3
    coords_jittered = coords_scaled + np.random.normal(0, jitter_strength, coords_scaled.shape)
    
    # Color mapping by show
    unique_shows = sorted(set(dominant_shows))
    colors = px.colors.qualitative.Set3
    color_cycle = (colors * ((len(unique_shows) // len(colors)) + 1))[:len(unique_shows)]
    show_color_map = dict(zip(unique_shows, color_cycle))
    
    # If clustering, create cluster color mapping
    if clustering:
        unique_clusters = sorted(set(cluster_labels))
        cluster_colors = px.colors.qualitative.G10
        # extend if needed
        if len(unique_clusters) > len(cluster_colors):
            cluster_colors *= (len(unique_clusters) // len(cluster_colors) + 1)
        cluster_color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}
    else:
        cluster_color_map = {}
    
    # Marker sizes based on total frequency
    freq_arr = word_total_freq
    size_min = 10
    size_max = 30
    freq_norm = (freq_arr - freq_arr.min()) / (freq_arr.max() - freq_arr.min() + 1e-6)
    size_values = size_min + freq_norm * (size_max - size_min)
    
    fig = go.Figure()
    
    # Plot each show's words
    for show_name in unique_shows:
        mask = [ (dom_show == show_name) for dom_show in dominant_shows ]
        if not any(mask):
            continue
        
        # gather subsets
        x_vals = coords_jittered[mask, 0]
        y_vals = coords_jittered[mask, 1]
        if dimensionality == 3:
            z_vals = coords_jittered[mask, 2]
        text_subset = np.array(words)[mask]
        hover_subset = np.array(hover_texts)[mask]
        size_subset = size_values[mask]
        color_choice = show_color_map.get(show_name, "gray")
        
        if dimensionality == 2:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                text=text_subset,
                hovertext=hover_subset,
                mode='markers+text',
                name=show_name,
                textposition='top center',
                textfont=dict(size=size_subset, color=color_choice),
                marker=dict(size=size_subset, color=color_choice, opacity=0.7,
                            line=dict(width=1, color='DarkSlateGrey')),
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                text=text_subset,
                hovertext=hover_subset,
                mode='markers+text',
                name=show_name,
                textposition='top center',
                textfont=dict(size=size_subset, color=color_choice),
                marker=dict(size=size_subset, color=color_choice, opacity=0.7,
                            line=dict(width=1, color='DarkSlateGrey')),
                hoverinfo='text'
            ))
    
    # Optionally add cluster "legend" items if clustering is True
    if clustering:
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == 'Noise':
                continue
            cluster_mask = (cluster_labels == cluster_id)
            if not np.any(cluster_mask):
                continue
            
            cx = coords_jittered[cluster_mask, 0]
            cy = coords_jittered[cluster_mask, 1]
            if dimensionality == 3:
                cz = coords_jittered[cluster_mask, 2]
            cluster_color = cluster_color_map[cluster_id]
            
            # Just add an invisible scatter for the cluster label to appear in the legend
            if dimensionality == 2:
                fig.add_trace(go.Scatter(
                    x=[cx.mean()],
                    y=[cy.mean()],
                    mode='markers',
                    marker=dict(size=0, color=cluster_color, opacity=0),
                    name=f"Cluster {cluster_id}",
                    showlegend=True
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[cx.mean()],
                    y=[cy.mean()],
                    z=[cz.mean()],
                    mode='markers',
                    marker=dict(size=0, color=cluster_color, opacity=0),
                    name=f"Cluster {cluster_id}",
                    showlegend=True
                ))
    
    # Layout
    if dimensionality == 2:
        layout = dict(
            title=f'Clusters by Show (2D) | max_words_per_show={max_words_per_show}',
            template='plotly_dark',
            height=800,
            width=1200,
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="auto",
                bgcolor="rgba(0,0,0,0)",
                itemsizing='constant',
                itemwidth=30
            )
        )
    else:
        layout = dict(
            title=f'Clusters by Show (3D) | max_words_per_show={max_words_per_show}',
            template='plotly_dark',
            height=800,
            width=1200,
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="auto",
                bgcolor="rgba(0,0,0,0)",
                itemsizing='constant',
                itemwidth=30
            )
        )
    
    fig.update_layout(layout)
    
    # Enhance interactivity
    if dimensionality == 3:
        fig.update_traces(
            selector=dict(type='scatter3d'),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
    else:
        fig.update_traces(
            selector=dict(type='scatter'),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
    
    return fig


def main():
    """Main function for the Streamlit app."""
    st.title("Podcast Analysis Dashboard")
    
    # Load data
    df = load_data()
    nlp = load_spacy_model()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Timeline View", "Keyword Analysis", "Raw Data"])

    with tab1:
        st.header("Show Timeline Analysis")
        fig_timeline = create_timeline_plot(df)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab2:
        st.header("Keyword Visualization")
        
        # Sidebar controls for keyword visualization
        with st.sidebar:
            st.header("Visualization Settings")
            
            min_freq = st.slider(
                "Minimum Word Frequency",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum number of times a word must appear to be included"
            )
            
            dimensionality = st.radio(
                "Dimensions",
                options=[3, 2],  # Changed order to make 3D default
                format_func=lambda x: f"{x}D",
                help="Number of dimensions for visualization"
            )
            
            method = st.selectbox(
                "Dimensionality Reduction Method",
                options=['UMAP', 't-SNE', 'MDS'],
                help="Method to reduce word vectors to 2D/3D"
            )
            
            clustering = st.checkbox(
                "Enable Clustering",
                value=True,
                help="Apply HDBSCAN clustering to words"
            )
            
            cluster_method = st.selectbox(
                "Cluster Selection Method",
                options=['eom', 'leaf'],
                help="Method for HDBSCAN clustering"
            ) if clustering else 'eom'
            
            max_words = st.slider(
                "Max Words per Show",
                min_value=5,
                max_value=50,
                value=7,
                help="Maximum number of words to display per show"
            )
        
        # Create and display keyword visualization
        with st.spinner("Generating keyword visualization..."):
            fig_keywords = create_show_keyword_visualization(
                df,
                min_freq=min_freq,
                dimensionality=dimensionality,
                method=method,
                clustering=clustering,
                cluster_selection_method=cluster_method,
                max_words_per_show=max_words
            )
            st.plotly_chart(fig_keywords, use_container_width=True)

    with tab3:
        st.header("Raw Data")
        sorted_df = df.sort_values('show', ascending=True)
        st.dataframe(
            sorted_df,
            use_container_width=True,
            column_config={
                col: st.column_config.TextColumn(
                    col,
                    width="large",
                    help=f"Click cell to see full {col} content",
                    max_chars=1000
                ) for col in sorted_df.columns
            },
            height=800,
            hide_index=False
        )

    # Clear sidebar when not on keyword analysis tab
    if tab1.id or tab3.id:
        with st.sidebar:
            st.empty()

if __name__ == "__main__":
    main()
