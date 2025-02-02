# Podcast Analysis Dashboard

A Streamlit-based dashboard for analyzing podcast content using natural language processing and interactive visualizations. This repository includes data wrangling code (in Jupyter notebooks) to extract and structure raw markdown data and a web app (in `app.py`) that presents the cleaned data through interactive timelines and semantic keyword analyses.



https://github.com/user-attachments/assets/7be2b4b0-c39b-4518-82ec-40c3cc0b6eb5



## Overview

This project takes raw markdown exports of podcast episodes and uses a combination of NLP, custom data models, and OpenAI’s GPT-powered extraction to transform unstructured content into a structured CSV file. The processed data is then fed into a Streamlit dashboard that provides:
- **Timeline View:** A time-based scatter plot of episodes/snippets.
- **Keyword Analysis:** A 2D/3D semantic mapping of keywords extracted from episode summaries.
- **Raw Data View:** An interactive table for detailed data exploration.

## Data Wrangling & Extraction

The raw data (e.g., `data/snipd_export_2024-12-24_14-55.md`) contains markdown-formatted podcast episodes with metadata, show notes, snips, and transcripts directly expoted from [Snipd](https://www.snipd.com/), a podcast player I use that allows you to save insights and notes conveniently. In the `notebooks/essential_code.ipynb` (and supporting code):

- **Pydantic Models:**  
  Two models—`EpisodeData` and `Snip`—are defined to represent the structure of each podcast episode and its snippets.
  
- **Episode Extraction:**  
  A function (`extract_episode_info`) sends the raw markdown to OpenAI’s GPT-based model (using a specified system prompt) to extract and structure the episode information in JSON format.
  
- **Data Aggregation:**  
  The notebook splits the raw markdown into episode blocks, processes each block using the extraction function, and saves the structured results to CSV files:
  - `responses_parsed_final.csv`: Contains full episode data.
  - `results_parsed_data.csv`: Contains each snippet merged with its parent episode’s metadata.
  
This wrangling step ensures that the dashboard works with a clean, normalized dataset.

## Visualizations

The Streamlit app (in `app.py`) leverages Plotly for interactive visualizations and includes three main views:

### 1. Timeline View
- **What it does:**  
  Displays a scatter plot of podcast episodes over time (filtered to 2024).  
- **How it works:**  
  Episodes are grouped by publish date, show, and title. The marker size represents the number of snippets per episode, and points are color-coded by show. Hovering over a point reveals detailed episode information.

### 2. Keyword Analysis
- **What it does:**  
  Provides an interactive 2D/3D visualization of keywords extracted from podcast summaries.
- **How it works:**  
  - **NLP Processing:**  
    Uses SpaCy (with the `en_core_web_md` model) to tokenize, lemmatize, and filter keywords (nouns and proper nouns) while excluding common English/German and custom stop words.
  - **Frequency & Filtering:**  
    Words below a set frequency threshold are removed, and an optional limit can restrict the number of keywords per show.
  - **Dimensionality Reduction:**  
    Word vectors are reduced to 2D or 3D coordinates using UMAP (or t-SNE/MDS as alternatives), positioning semantically similar words near each other.
  - **Clustering:**  
    Optionally applies HDBSCAN to group similar keywords into clusters.
  - **Visual Encoding:**  
    Marker sizes are proportional to word frequency, and colors indicate the dominant show (or cluster). Hover text provides detailed per-show frequency counts.
  
This visualization enables users to explore the semantic landscape of podcast topics and see at a glance which keywords are most prominent in each show.

### 3. Raw Data View
- **What it does:**  
  Displays the full dataset in an interactive table.
- **How it works:**  
  The table supports sorting and expandable cells, allowing users to inspect every detail of the processed podcast data.

## Technical Stack

- **Frontend:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **NLP:** SpaCy (`en_core_web_md` model)  
- **Visualization:** Plotly  
- **Machine Learning:**  
  - UMAP (Dimensionality Reduction)  
  - HDBSCAN (Clustering)  
  - scikit-learn (t-SNE, MDS)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/podcast-analysis.git
   cd podcast-analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the SpaCy model:**
   ```bash
   python -m spacy download en_core_web_md
   ```

5. **Set up Streamlit configuration (optional for dark theme):**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[theme]\nbase='dark'\n" > ~/.streamlit/config.toml
   ```

## Usage

1. **Process Raw Data:**  
   Run the notebook or the wrangling script to extract and structure raw markdown data. This will generate CSV files (`responses_parsed_final.csv` and `results_parsed_data.csv`) that the dashboard uses.

2. **Start the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. **Explore the Dashboard:**  
   - **Timeline Tab:** Review episodes over time with interactive scatter plots.
   - **Keyword Analysis Tab:** Use sidebar controls to adjust parameters (word frequency, dimensionality, reduction method, clustering, words per show) and explore semantic clusters.
   - **Raw Data Tab:** Inspect the full dataset interactively.

## Demo

A quick demo video is available (link or embedded video) that walks through the data wrangling process and demonstrates how the dashboard visualizes podcast data—from timeline insights to semantic keyword clusters.

## Data Format

The dashboard expects a CSV file (`data/full_df.csv`) with the following columns:
- `show`: Name of the podcast show
- `episode_title`: Title of the episode
- `publish_date`: Publication date
- `summary`: Text content for analysis

*The raw markdown extraction process further enriches this data with detailed metadata and snippet-level information.*

## License

MIT
