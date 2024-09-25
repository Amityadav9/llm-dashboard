import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objects as go
import time


def load_data(data):
    return pd.read_csv(data)


def stream_text(text: str, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def stream_df(df, delay: float = 0.02):
    for i in range(len(df)):
        yield df.iloc[[i]]
        time.sleep(delay)


df = load_data("data/llm_clean_list.csv")
df_leaderboard = load_data("data/llm_clean_leaderboard.csv")


def get_top_models(df, metric, n=10):
    """
    Get the top n models based on a given metric.

    :param df: DataFrame containing the model data
    :param metric: The metric to sort by (e.g., 'Downloads', 'Likes', 'Context Length')
    :param n: Number of top models to return (default is 10)
    :return: DataFrame with the top n models
    """
    return df.nlargest(n, columns=metric)[["Model Name", "Maintainer", metric]]


def create_word_cloud(df):
    st.subheader("Word Cloud of LLM Model Names")

    if "Model Name" not in df.columns:
        st.warning(
            "Word cloud generation requires a 'Model Name' column. This feature is not available with the current dataset."
        )
        return

    # Combine all model names
    text = " ".join(df["Model Name"].dropna())

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def compare_models(df):
    st.subheader("Model Comparison")

    # Select models to compare
    all_models = df["Model Name"].unique().tolist()
    selected_models = st.multiselect(
        "Select models to compare", all_models, default=all_models[:3]
    )

    if len(selected_models) < 2:
        st.warning("Please select at least two models to compare.")
        return

    # Select metrics for comparison
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    selected_metrics = st.multiselect(
        "Select metrics for comparison", numeric_columns, default=numeric_columns[:3]
    )

    if not selected_metrics:
        st.warning("Please select at least one metric for comparison.")
        return

    # Filter data for selected models and metrics
    comparison_data = df[df["Model Name"].isin(selected_models)][
        ["Model Name"] + selected_metrics
    ]

    # Create radar chart
    fig = go.Figure()

    for model in selected_models:
        model_data = comparison_data[comparison_data["Model Name"] == model]
        fig.add_trace(
            go.Scatterpolar(
                r=model_data[selected_metrics].values[0],
                theta=selected_metrics,
                fill="toself",
                name=model,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    st.plotly_chart(fig)

    # Display comparison table
    st.dataframe(comparison_data)

    # Option to download comparison data
    csv = comparison_data.to_csv(index=False)
    st.download_button(
        label="Download comparison data as CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
    )


def create_searchable_table(df):
    st.subheader("Searchable and Sortable Model Table")

    # Text input for search
    search_term = st.text_input("Search for a model", "")

    # Multi-select for columns to display
    all_columns = df.columns.tolist()
    default_columns = ["Model Name", "Maintainer", "Size", "Context Length"]
    selected_columns = st.multiselect(
        "Select columns to display", all_columns, default=default_columns
    )

    # Filter dataframe based on search term
    if search_term:
        filtered_df = df[
            df["Model Name"].str.contains(search_term, case=False)
            | df["Maintainer"].str.contains(search_term, case=False)
        ]
    else:
        filtered_df = df

    # Display filtered dataframe with selected columns
    if not filtered_df.empty:
        st.dataframe(filtered_df[selected_columns], height=400)
        st.write(f"Showing {len(filtered_df)} out of {len(df)} models")
    else:
        st.warning("No models found matching your search criteria.")

    # Option to download filtered data
    csv = filtered_df[selected_columns].to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_models.csv",
        mime="text/csv",
    )


def about_page():
    st.title("About LLM Dashboard")

    st.markdown("""
    Welcome to the LLM (Large Language Model) Dashboard!

    This interactive dashboard provides comprehensive information and analysis tools for various Large Language Models. Our goal is to help researchers, developers, and AI enthusiasts explore and compare different LLMs easily.

    Key Features:
    - **LLM List**: Browse through a comprehensive list of LLMs with quick filters.
    - **Ask LLM**: Search for specific LLM models and get detailed information.
    - **LLM Stats**: Analyze and compare LLMs based on various metrics, visualize data, and download comparison results.
    - **Leaderboard**: View top-performing models across different benchmarks.

    Data Sources:
    The information presented in this dashboard is compiled from various publicly available sources and is regularly updated. However, please note that the AI field is rapidly evolving, and some information may not always be up to date.

    Feedback and Contributions:
    We welcome feedback and contributions to improve this dashboard. Please visit our GitHub repository to report issues, suggest features, or contribute to the project.

    Disclaimer:
    This dashboard is for informational purposes only. Performance metrics and other data should be used as reference points, not as definitive measures of a model's capabilities.
    """)

    st.markdown("---")
    st.markdown("Created by Amit Yadav ")


def ask_llm_page():
    st.title("Ask LLM")

    with st.container():
        st.markdown("### Search for an LLM Model")
        search_model = st.text_input("Enter model name", key="search_model")
        search_button = st.button("Search")

    if search_button and search_model:
        result_df = df[df["Model Name"].str.contains(search_model.title(), case=False)]

        if result_df.empty:
            st.warning("No models found matching your search.")
        else:
            for _, row in result_df.iterrows():
                with st.expander(f"Details for {row['Model Name']}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Name", row["Model Name"])
                        st.metric("Maintainer", row["Maintainer"])
                        st.metric("Size", row["Size"])
                    with col2:
                        st.metric("Score", row["Score"])
                        st.metric("Context Length", row["Context Length"])

                    st.markdown("### Model Description")
                    description = f"""
                    {row['Model Name']} is the latest version of {row['Maintainer']}'s large language model (LLM). 
                    It is designed to handle a more extensive array of tasks, including text, image, and video processing. 
                    {row['Model Name']} has a context length of {row['Context Length']}, allowing for more nuanced understanding and generation of content.
                    It has {row['Size']} parameters.
                    """
                    st.markdown(description)

                    st.markdown("### Model Summary")
                    st.dataframe(pd.DataFrame([row]))


def llm_list_page():
    st.title("LLM List")
    quick_filters = st.multiselect("Filters", df.columns.tolist())
    st.dataframe(df[quick_filters])


def llm_stats_page():
    st.title("LLM Stats and Comparison")

    # Metric selection
    metrics = ["Downloads", "Likes", "Context Length"]
    selected_metric = st.selectbox("Select Primary Metric", metrics)

    st.subheader(f"Top {10} Models by {selected_metric}")
    top_models = get_top_models(df, selected_metric)
    st.dataframe(top_models)

    top_models_fig = px.bar(
        top_models,
        x="Model Name",
        y=selected_metric,
        color="Maintainer",
        title=f"Top {10} Models by {selected_metric}",
    )
    st.plotly_chart(top_models_fig, use_container_width=True)

    # Model selection for comparison
    all_models = df["Model Name"].unique().tolist()
    selected_models = st.multiselect(
        "Select Models to Compare", all_models, default=all_models[:5]
    )

    if not selected_models:
        st.warning("Please select at least one model to display stats.")
        return

    # Filter data based on selected models
    filtered_df = df[df["Model Name"].isin(selected_models)]

    # Create interactive bar chart
    bar_fig = px.bar(
        filtered_df,
        x="Model Name",
        y=selected_metric,
        color="Maintainer",
        hover_data=["Size", "Score"],
        labels={"Model Name": "Model", selected_metric: f"{selected_metric} Value"},
        title=f"Comparison of {selected_metric} Across Selected Models",
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    # Create radar chart for multi-metric comparison
    if len(selected_models) > 0:
        st.subheader("Multi-Metric Comparison")
        radar_metrics = st.multiselect(
            "Select Metrics for Radar Chart", metrics, default=metrics[:3]
        )

        if radar_metrics:
            radar_fig = go.Figure()

            for model in selected_models:
                model_data = filtered_df[filtered_df["Model Name"] == model]
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=[model_data[metric].values[0] for metric in radar_metrics],
                        theta=radar_metrics,
                        fill="toself",
                        name=model,
                    )
                )

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Multi-Metric Model Comparison",
            )

            st.plotly_chart(radar_fig, use_container_width=True)

    # Display detailed stats table
    st.subheader("Detailed Stats")
    st.dataframe(filtered_df)

    compare_models(df)

    create_searchable_table(df)

    create_word_cloud(df)


def llm_leaderboard_page():
    st.title("LLM Leaderboard")

    quick_filters = st.multiselect(
        "Filters ",
        [
            "Model Name",
            "Maintainer",
            "License",
            "Context Length",
            "Mt Bench",
            "Humaneval",
            "Input Priceusd/1M Tokens",
        ],
        default=["Model Name", "Maintainer", "License", "Context Length", "Humaneval"],
    )

    st.dataframe(df_leaderboard[quick_filters])

    metric = st.selectbox("Metric", ["Context Length", "Humaneval", "MT Bench"])

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("LLM Leaderboard Stats "):
            show_stats_as_table_for_leaderboard()

    # Top N Models

    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]

    # Bar Chart
    bar_fig = px.bar(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Bar Chart")
    st.plotly_chart(bar_fig)

    # Pie Chart
    pie_fig = px.pie(top_models, names="Model Name", values=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Pie Chart")
    st.plotly_chart(pie_fig)

    # Scatter Chart
    scatter_fig = px.scatter(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Scatter Plot")
    st.plotly_chart(scatter_fig)


@st.experimental_dialog("LLM List - Stats")
def show_llm_stats_as_table():
    metric = st.selectbox("Metric ", ["Downloads", "Likes", "Context Length"])
    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model Name", "Maintainer", metric]]

    st.dataframe(top_models)


@st.experimental_dialog("LLM List ")
def show_llm_list_as_table():
    st.dataframe(df)


@st.experimental_dialog("LLM Leaderboard - Stats")
def show_stats_as_table_for_leaderboard():
    metric = st.selectbox("Metric  ", ["Context Length", "Humaneval", "MT Bench"])
    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]

    st.dataframe(top_models)


about = st.Page(about_page, title="About", icon=":material/info:")
ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")
llm_stats = st.Page(llm_stats_page, title="LLM Stats", icon=":material/list:")
llm_list = st.Page(llm_list_page, title="LLM List", icon=":material/analytics:")
llm_leaderboard = st.Page(
    llm_leaderboard_page, title="Leaderboard", icon=":material/favorite:"
)


# Navigations
pg = st.navigation(
    {"Home": [llm_list, ask_llm, llm_stats, llm_leaderboard], "About": [about]}
)


pg.run()
