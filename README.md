# Podcast Analysis Dashboard

A Streamlit-based dashboard for analyzing podcast content using natural language processing and interactive visualizations.

## Features

### 1. Timeline View
- Interactive scatter plot showing podcast episodes over time
- Size of points indicates number of snippets per episode
- Color-coded by show
- Hover functionality for detailed episode information

### 2. Keyword Analysis
- Interactive 2D/3D visualization of keywords across different shows
- Semantic clustering using word embeddings
- Customizable visualization parameters:
  - Minimum word frequency
  - Dimensionality (2D/3D)
  - Dimensionality reduction method (UMAP, t-SNE, MDS)
  - Clustering options (HDBSCAN)
  - Words per show limit

### 3. Raw Data View
- Full dataset exploration
- Sortable columns
- Expandable cells for full content viewing

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **NLP**: SpaCy (en_core_web_md model)
- **Visualization**: Plotly
- **Machine Learning**: 
  - UMAP (Dimensionality Reduction)
  - HDBSCAN (Clustering)
  - scikit-learn (t-SNE, MDS)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/podcast-analysis.git
cd podcast-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the SpaCy model:
```bash
python -m spacy download en_core_web_md
```

5. Set up Streamlit configuration:
```bash
mkdir -p ~/.streamlit/
echo "[theme]\nbase='dark'\n" > ~/.streamlit/config.toml
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar controls in the Keyword Analysis tab to customize visualizations:
   - Adjust minimum word frequency
   - Switch between 2D and 3D views
   - Change dimensionality reduction methods
   - Enable/disable clustering
   - Modify the number of words per show

## Data Format

The dashboard expects a CSV file (`data/full_df.csv`) with the following columns:
- `show`: Name of the podcast show
- `episode_title`: Title of the episode
- `publish_date`: Publication date
- `summary`: Text content for analysis

## Project Structure

```
podcast-analysis/
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── data/
│   ├── visualization/   # Visualization modules
│   └── utils/          # Utility functions
├── data/
│   └── full_df.csv     # Dataset
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── setup.sh           # Setup script
```


## License

This project is licensed under the MIT License 