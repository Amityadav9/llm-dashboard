## LLM Dashboard

## About
The LLM Dashboard is an interactive web application designed to provide comprehensive information and analysis tools for various Large Language Models (LLMs). Built with Streamlit, this dashboard aims to be a valuable resource for researchers, developers, and AI enthusiasts interested in exploring and comparing different LLMs.

## Features

### LLM List
Browse through a comprehensive list of LLMs with quick filters. This feature allows users to get a broad overview of the available models in the field.

### Ask LLM
Search for specific LLM models and get detailed information. This section provides in-depth details about individual models, including:
- Model Name
- Maintainer
- Size
- Score
- Context Length
- A brief description of the model's capabilities

### LLM Stats
This powerful section offers various tools for analyzing and comparing LLMs:
- Select and compare multiple models based on various metrics
- Visualize data using interactive bar charts
- Compare models using radar charts for multi-metric analysis
- View detailed statistics for selected models
- Download comparison data for further analysis

### Leaderboard
Explore top-performing models across different benchmarks. This section includes:
- Customizable filters for viewing leaderboard data
- Interactive visualizations including bar charts, pie charts, and scatter plots
- Options to view and download leaderboard statistics

### Word Cloud
Visualize the prevalence of different LLM models in the dataset using an interactive word cloud.

### Searchable and Sortable Table
Quickly find and sort LLMs based on various criteria. This feature allows users to:
- Search for models by name or maintainer
- Select specific columns to display
- Sort the table by any selected column
- Download filtered data for offline analysis

## Data and Sources
The information presented in this dashboard is compiled from various publicly available sources and is regularly updated. However, please note that the AI field is rapidly evolving, and some information may not always be up to date.

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/Amityadav9/llm-dashboard.git
   cd llm-dashboard
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run dashboard.py
   ```

4. Open your web browser and go to the address provided in the terminal (typically http://localhost:8501).

## Contributing
We welcome contributions to improve this dashboard! Please feel free to submit issues, feature requests, or pull requests through GitHub.

## Disclaimer
This dashboard is for informational purposes only. Performance metrics and other data should be used as reference points, not as definitive measures of a model's capabilities.

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [pandas](https://pandas.pydata.org/)
- [WordCloud](https://github.com/amueller/word_cloud)