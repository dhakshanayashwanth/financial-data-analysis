import openai
import streamlit as st
import pandas as pd
import re
import numpy as np
import json
import uuid
import time
import os
from dotenv import load_dotenv
import plotly.express as px

# Load environment variables from .env file (only for local development)
load_dotenv()

# Load OpenAI API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

# Load existing mappings from a JSON file
def load_mappings():
    try:
        with open('mappings.json', 'r') as file:
            mappings = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Could not load mappings: {e}")
        mappings = {}
    return mappings

# Save mappings to a JSON file
def save_mappings(mappings):
    with open('mappings.json', 'w') as file:
        json.dump(mappings, file)

# Function to categorize values based on mean and standard deviation
def categorize_values(series):
    mean = series.mean()
    std = series.std()
    bins = [-np.inf, mean - 2*std, mean - std, mean + std, mean + 2*std, np.inf]
    labels = ['Much Below Avg', 'Below Avg', 'Avg', 'Above Avg', 'Much Above Avg']
    
    if len(np.unique(bins)) != len(bins):
        bins = np.unique(bins)
        labels = labels[:len(bins)-1]

    categories = pd.cut(
        series,
        bins=bins,
        labels=labels,
        duplicates='drop'
    )
    return categories

# Function to generate unique code for any non-numeric column
def generate_code(value, mapping_dict):
    if value not in mapping_dict:
        if isinstance(value, str) and '|' not in value:
            mapping_dict[value] = uuid.uuid4().hex[:8].upper()
        else:
            mapping_dict[value] = str(value) if value is not None else 'None'
    return mapping_dict[value]

# Function to reverse the mapping
def reverse_mappings(text, mappings):
    for col, mapping_dict in mappings.items():
        reverse_dict = {v: str(k) for k, v in mapping_dict.items() if '|' not in str(k)}
        for code, original in reverse_dict.items():
            if isinstance(code, str) and isinstance(original, str):
                text = text.replace(code, original)
    return text

# Function to get insights from OpenAI with exponential backoff
def get_insights_from_openai(data_str, model, insight_type, scenario=None, retries=5):
    prompt = f"Here is the dataset:\n{data_str}\nPlease provide the insights from this data in concise bullet points. Focus on actionable insights and exclude any basic descriptive analytics or trivial observations. Please don't include anything tied to descriptive analytics."

    if insight_type == "General Insights":
        prompt += (" Provide general insights from the dataset."
                   " Focus on actionable insights and exclude any basic descriptive analytics or trivial observations."
                   " Ensure the insights are concise and relevant to a senior finance audience, focusing on strategic decisions and business impact.")

    elif insight_type == "Trend Analysis":
        prompt += (" Perform a detailed trend analysis focusing on identifying and analyzing trends over time."
                   " Include details such as monthly or quarterly trends, seasonal patterns, and year-over-year comparisons."
                   " Specifically, identify key drivers of revenue growth or decline, highlight significant changes in key metrics,"
                   " and provide insights on sales performance by segment and product."
                   " Provide specific insights based on the identified trends."
                   " Ensure the insights are concise and relevant to a senior finance audience, focusing on strategic decisions and business impact."
                   " Please avoid any basic descriptive analytics.")

    elif insight_type == "Key Drivers of Revenue Growth":
        prompt += (" Analyze the dataset to identify the key drivers of revenue growth last month."
                   " Provide detailed insights on which factors contributed most significantly to the increase or decrease in revenue."
                   " Include specific segments, products, or discounts that had the largest impact.")
        
    elif insight_type == "What-If Scenario Analysis":
        change = scenario['change']
        factor = scenario['factor']
        prompt += (f" Perform a what-if scenario analysis to evaluate the impact of decreasing the {factor} by {change}%."
                       " Provide specific and tangible insights on how this decrease would affect the {factor} or any of the other fields in the dataset."
                       " Avoid generic statements and focus on concrete data-driven insights."
                       " Ensure the analysis is relevant to a CFO, VP of Finance or Director of Finance, focusing on strategic decisions and business impact."
                       " Please avoid any basic descriptive analytics."
                       " Focus only on the impacts without providing any conclusions, recommendations, or key metrics."
                       " Do not include conclusions, recommendations, or key metrics.")

    attempt = 0
    while attempt < retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            st.warning(f"Rate limit reached for {model}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    st.error("Failed to get insights after multiple attempts. Please try again later.")
    return None

# Function to get visualization recommendations from OpenAI
def get_visualization_recommendations(data_str):
    prompt = f"Here is the dataset:\n{data_str}\nPlease suggest some potential visualizations from this data. Provide recommendations with brief descriptions of the visuals."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI data analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Load existing mappings
mappings = load_mappings()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "AI Insights", 
    "AI Visualizations", 
    "ChatGPT", 
    "Intro to ChatGPT", 
    "AI in Finance",
    "Interactive AI Tutorials"
])

# AI Insights Section
if section == "AI Insights":
    st.title("Hi, I'm Beam 🐳 - your data insights companion. I can generate insights 🔍 and recommendations from your data in 60 seconds or less.")
    st.markdown('<p style="font-size:14px; margin-top: 20px;">Beam is powered by the latest version of ChatGPT: 4o and does not train on any of your data.</p>', unsafe_allow_html=True)
    st.subheader("To begin, please upload 📑 your Google Sheet or CSV file below.")
    st.markdown("**Note:** The most data ChatGPT can analyze is 500 rows or around 103KB.")

    uploaded_files = st.file_uploader("Please choose CSV or Google Sheet files.", type=['csv', 'xlsx'], accept_multiple_files=True, key="ai_insights_upload")

    dataframes = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'xlsx':
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheets = st.multiselect(f"Select the sheets to analyze from {uploaded_file.name}", sheet_names)
                if st.button(f"Load selected sheets from {uploaded_file.name}", key=f"load_{uploaded_file.name}"):
                    for selected_sheet in selected_sheets:
                        with st.spinner(f'Loading {selected_sheet} from {uploaded_file.name}...'):
                            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                            dataframes.append(df)
                            st.session_state[f'df_{uploaded_file.name}_{selected_sheet}'] = df
            elif file_extension == 'csv':
                with st.spinner(f'Loading data from {uploaded_file.name}...'):
                    df = pd.read_csv(uploaded_file)
                    dataframes.append(df)
                    st.session_state[f'df_{uploaded_file.name}'] = df

    if len(dataframes) > 1:
        st.subheader("Join Datasets")
        common_columns = list(set.intersection(*(set(df.columns) for df in dataframes)))
        if common_columns:
            join_column = st.selectbox("Select the column to join on", common_columns)
            how = st.selectbox("Select join type", ["inner", "outer", "left", "right"])
            if st.button("Join Datasets", key="join_datasets"):
                joined_df = dataframes[0]
                for df in dataframes[1:]:
                    joined_df = joined_df.merge(df, on=join_column, how=how)
                st.session_state['joined_df'] = joined_df
        else:
            st.warning("No common columns found to join the datasets.")

    if 'joined_df' in st.session_state:
        df = st.session_state['joined_df']
    else:
        df = dataframes[0] if dataframes else None

    if df is not None:
        if len(df) > 500:
            st.warning(f"Dataset contains {len(df)} rows. ChatGPT can only analyze up to 500 rows.")
            
            auto_filter = st.radio("Do you want us to automatically filter the most recent 500 rows?", ("Yes", "No"), key="auto_filter")
            
            if auto_filter == "Yes":
                df = df.iloc[::-1].head(500).reset_index(drop=True)
                st.session_state['df'] = df
                st.success("Automatically filtered to the most recent 500 rows.")
            else:
                st.write("Please filter the data to 500 rows or less and re-upload the file.")
                
                st.subheader("Filter Data")
                filter_columns = st.multiselect("Select columns to filter by", df.columns.tolist(), key="filter_columns")
                filtered_df = df.copy()

                for col in filter_columns:
                    unique_values = df[col].unique().tolist()
                    selected_values = st.multiselect(f"Select values for {col}", unique_values, default=unique_values, key=f"filter_{col}")
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

                if len(filtered_df) <= 500:
                    st.session_state['df'] = filtered_df
                    st.write("Filtered Data")
                    st.dataframe(filtered_df)
                else:
                    st.warning("Filtered data still exceeds 500 rows. Please adjust your filters or manually reduce the dataset size.")
        else:
            st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.columns = df.columns.str.strip()

        date_column = None
        for col in df.columns:
            if re.search(r'date|month|year', col, re.IGNORECASE):
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_column = col
                    break
                except (ValueError, TypeError):
                    continue

        if date_column:
            df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%m/%d/%Y')

        exclude_columns = [date_column] if date_column else []

        numeric_columns = []
        for col in df.columns:
            if col not in exclude_columns:
                first_value = df[col].iloc[0]
                if not any(c.isalpha() for c in str(first_value)):
                    df[col] = df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)).split('.')[0])
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_columns.append(col)

        original_df = df.copy()

        st.subheader("Optional: Filter your data.")
        st.markdown("**Note:** If you don't need to filter your dataset, please scroll to the bottom, select the analysis you need and click on the Perform Analysis button.")

        filter_columns = st.multiselect("Select columns to filter by", df.columns.tolist(), key="optional_filter_columns")
        filtered_df = df.copy()

        for col in filter_columns:
            unique_values = df[col].unique().tolist()
            selected_values = st.multiselect(f"Select values for {col}", unique_values, default=unique_values, key=f"optional_filter_{col}")
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

        st.write("Original Data")
        st.dataframe(filtered_df)

        try:
            categorized_df = filtered_df.copy()
            for col in numeric_columns:
                categorized_df[col] = categorize_values(filtered_df[col])

            encoded_df = filtered_df.copy()
            for col in filtered_df.columns:
                if col not in numeric_columns and col not in exclude_columns:
                    if col not in mappings:
                        mappings[col] = {}
                    encoded_df[col] = filtered_df[col].apply(lambda x: generate_code(x, mappings[col]))

            save_mappings(mappings)

            combined_df = encoded_df.copy()
            for col in numeric_columns:
                combined_df[col] = categorized_df[col]

            st.write("Mapped and Categorized Data")
            st.dataframe(combined_df)

            available_insights = ["General Insights"]

            trend_columns = {'Date', 'Month Number', 'Month Name', 'Year'}
            if any(col in df.columns for col in trend_columns):
                available_insights.append("Trend Analysis")

            key_drivers_columns = {'Gross Sales', 'Sales', 'Date', 'Profit'}
            if any(col in df.columns for col in key_drivers_columns):
                available_insights.append("Key Drivers of Revenue Growth")

            exclude_factors = {'Month Number', 'Month Name', 'Year'}
            numeric_factors = [col for col in categorized_df.columns if set(categorized_df[col].unique()).issubset({'Below Avg', 'Avg', 'Above Avg'}) and col not in exclude_factors]

            if numeric_factors:
                available_insights.append("What-If Scenario Analysis")

            selected_insight = st.selectbox("Please select the type of insights you would like to see and wait 20-30 seconds for the insights to generate:", available_insights, key="selected_insight")

            scenario = None
            if selected_insight == "What-If Scenario Analysis":
                factor = st.selectbox("Select the factor to change:", numeric_factors, key="what_if_factor")
                change = st.number_input("Enter the percentage change (You can enter negative numbers like -2 or positive numbers like 2):", min_value=-100, max_value=100, value=5, key="what_if_change")
                scenario = {"factor": factor, "change": change}

            if selected_insight and (selected_insight != "What-If Scenario Analysis" or scenario):
                if st.button("Perform Analysis", key="perform_analysis"):
                    progress_bar = st.progress(0)
                    with st.spinner('Generating insights, please wait...'):
                        combined_df_str = combined_df.to_csv(index=False)
                        for i in range(0, 50, 5):
                            time.sleep(0.5)
                            progress_bar.progress(i)
                        insights = get_insights_from_openai(combined_df_str, model="gpt-4o", insight_type=selected_insight, scenario=scenario)
                        for i in range(50, 100, 5):
                            time.sleep(0.5)
                            progress_bar.progress(i)
                        if selected_insight != "What-If Scenario Analysis":
                            st.markdown("**Conclusions and Recommendations**")
                        decoded_insights = reverse_mappings(insights, mappings)
                        st.markdown(decoded_insights)
                        progress_bar.progress(100)
        except ValueError as e:
            st.error(f"Please provide Beam with more data so he can give you your insights.")
    else:
        st.write("Please upload a file to proceed.")

# Visualize Data Section
elif section == "AI Visualizations":
    st.title("Visualize Your Data with AI")
    st.subheader("To begin, please upload 📑 your Google Sheet or CSV file below.")
    st.markdown("**Note:** Ensure your dataset is properly prepared. ChatGPT can not create visualizations for you but it can offer recommendations on which charts can be created with your data. This can assist with exploratory analysis, gathering insights and creating ppts.")

    uploaded_files = st.file_uploader("Please choose CSV or Google Sheet files.", type=['csv', 'xlsx'], accept_multiple_files=True, key="visualize_data_upload")

    dataframes = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'xlsx':
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheets = st.multiselect(f"Select the sheets to analyze from {uploaded_file.name}", sheet_names)
                if st.button(f"Load selected sheets from {uploaded_file.name}", key=f"load_viz_{uploaded_file.name}"):
                    for selected_sheet in selected_sheets:
                        with st.spinner(f'Loading {selected_sheet} from {uploaded_file.name}...'):
                            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                            dataframes.append(df)
                            st.session_state[f'viz_df_{uploaded_file.name}_{selected_sheet}'] = df
            elif file_extension == 'csv':
                with st.spinner(f'Loading data from {uploaded_file.name}...'):
                    df = pd.read_csv(uploaded_file)
                    dataframes.append(df)
                    st.session_state[f'viz_df_{uploaded_file.name}'] = df

    if 'viz_df' not in st.session_state and dataframes:
        if len(dataframes) == 1:
            df = dataframes[0]
            st.session_state['viz_df'] = df
        elif len(dataframes) > 1:
            st.subheader("Join Datasets")
            common_columns = list(set.intersection(*(set(df.columns) for df in dataframes)))
            if common_columns:
                join_column = st.selectbox("Select the column to join on", common_columns, key="viz_join_column")
                how = st.selectbox("Select join type", ["inner", "outer", "left", "right"], key="viz_join_type")
                if st.button("Join Datasets", key="viz_join_datasets"):
                    joined_df = dataframes[0]
                    for df in dataframes[1:]:
                        joined_df = joined_df.merge(df, on=join_column, how=how)
                    st.session_state['viz_df'] = joined_df
            else:
                st.warning("No common columns found to join the datasets.")
    else:
        df = st.session_state.get('viz_df', None)

    if df is not None:
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.columns = df.columns.str.strip()

        st.subheader("Preview Data")
        st.write("Here's a preview of your data:")
        st.dataframe(df)

        # Encode and categorize data
        categorized_df = df.copy()
        numeric_columns = []
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                categorized_df[col] = categorize_values(df[col])
                numeric_columns.append(col)

        encoded_df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                if col not in mappings:
                    mappings[col] = {}
                encoded_df[col] = df[col].apply(lambda x: generate_code(x, mappings[col]))

        save_mappings(mappings)

        combined_df = encoded_df.copy()
        for col in numeric_columns:
            combined_df[col] = categorized_df[col]

        st.subheader("Mapped and Categorized Data")
        st.dataframe(combined_df)

        # Get visualization recommendations once and store them
        if 'visualization_recommendations' not in st.session_state:
            progress_bar = st.progress(0)
            combined_df_str = combined_df.to_csv(index=False)
            with st.spinner('Generating visualization recommendations, please wait...'):
                recommendations = get_visualization_recommendations(combined_df_str)
                st.session_state['visualization_recommendations'] = recommendations
                progress_bar.progress(100)

        st.subheader("Visualization Recommendations")
        st.markdown(st.session_state['visualization_recommendations'])

        # Create Visuals
        st.subheader("Create Visuals")
        st.write("Choose a type of chart and columns to visualize your data.")

        chart_type = st.selectbox("Select chart type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"], key="chart_type")
        x_column = st.selectbox("Select X-axis column", df.columns, key="x_column")
        y_column = st.selectbox("Select Y-axis column", df.columns, key="y_column")
        color_column = st.selectbox("Select column for color (optional)", [None] + list(df.columns), key="color_column")
        chart_title = st.text_input("Enter chart title", key="chart_title")
        x_axis_title = st.text_input("Enter X-axis title", key="x_axis_title")
        y_axis_title = st.text_input("Enter Y-axis title", key="y_axis_title")

        if st.button("Create Visual", key="create_visual"):
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=chart_title, color_discrete_sequence=px.colors.qualitative.Bold)
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x_column, y=y_column, color=color_column, title=chart_title, color_discrete_sequence=px.colors.qualitative.Bold)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=chart_title, color_discrete_sequence=px.colors.qualitative.Bold)
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=x_column, values=y_column, title=chart_title, color_discrete_sequence=px.colors.qualitative.Bold)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_column, y=y_column, color=color_column, title=chart_title, color_discrete_sequence=px.colors.qualitative.Bold)

            fig.update_layout(
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Create Multiple Charts")
        if st.button("Add Another Chart", key="add_another_chart"):
            chart_type2 = st.selectbox("Select chart type for second chart", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"], key="chart2_type")
            x_column2 = st.selectbox("Select X-axis column for second chart", df.columns, key="x_column2")
            y_column2 = st.selectbox("Select Y-axis column for second chart", df.columns, key="y_column2")
            color_column2 = st.selectbox("Select column for color for second chart (optional)", [None] + list(df.columns), key="color_column2")
            chart_title2 = st.text_input("Enter chart title for second chart", key="chart_title2")
            x_axis_title2 = st.text_input("Enter X-axis title for second chart", key="x_axis_title2")
            y_axis_title2 = st.text_input("Enter Y-axis title for second chart", key="y_axis_title2")

            if st.button("Create Second Visual", key="create_second_visual"):
                if chart_type2 == "Bar Chart":
                    fig2 = px.bar(df, x=x_column2, y=y_column2, color=color_column2, title=chart_title2, color_discrete_sequence=px.colors.qualitative.Bold)
                elif chart_type2 == "Line Chart":
                    fig2 = px.line(df, x=x_column2, y=y_column2, color=color_column2, title=chart_title2, color_discrete_sequence=px.colors.qualitative.Bold)
                elif chart_type2 == "Scatter Plot":
                    fig2 = px.scatter(df, x=x_column2, y=y_column2, color=color_column2, title=chart_title2, color_discrete_sequence=px.colors.qualitative.Bold)
                elif chart_type2 == "Pie Chart":
                    fig2 = px.pie(df, names=x_column2, values=y_column2, title=chart_title2, color_discrete_sequence=px.colors.qualitative.Bold)
                elif chart_type2 == "Histogram":
                    fig2 = px.histogram(df, x=x_column2, y=y_column2, color=color_column2, title=chart_title2, color_discrete_sequence=px.colors.qualitative.Bold)

                fig2.update_layout(
                    xaxis_title=x_axis_title2,
                    yaxis_title=y_axis_title2
                )

                st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No data available. Please upload data to create visuals.")

# ChatGPT Section
elif section == "ChatGPT":
    st.title("ChatGPT")
    st.markdown("""
    - [ChatGPT](https://www.openai.com/chatgpt)
    """)

# Intro to ChatGPT Section
elif section == "Intro to ChatGPT":
    st.title("Intro to ChatGPT")
    st.markdown("Here are some recommended free courses to get started with ChatGPT:")
    st.markdown("""
    - [Introduction to ChatGPT by Codecademy](https://www.codecademy.com/learn/intro-to-chatgpt)
    - [edX: Introduction to ChatGPT](https://www.edx.org/learn/chatgpt/edx-introduction-to-chatgpt)
    - [datacamp: Introduction to ChatGPT](https://www.datacamp.com/courses/introduction-to-chatgpt)
    - [The Neuron: Intro to ChatGPT](https://www.theneuron.ai/courses/intro-to-chatgpt-training-course)
    """)

# AI in Finance Section
elif section == "AI in Finance":
    st.title("AI in Finance")
    st.markdown("Learn about the latest uses of AI in finance. Here are some recent articles:")
    
    try:
        with open('latest_ai_in_finance_links.json', 'r') as file:
            data = json.load(file)
            last_updated = data["last_updated"]
            links = data["links"]
            st.markdown(f"**Last Updated:** {last_updated}")
            for link in links:
                st.markdown(f"- [{link['title']}]({link['url']})")
    except FileNotFoundError:
        st.warning("The latest AI in Finance links have not been fetched yet. Please run the fetch_latest_links.py script.")
    except json.JSONDecodeError:
        st.warning("There was an error reading the latest AI in Finance links. Please check the JSON file.")

# Interactive AI Tutorials Section
elif section == "Interactive AI Tutorials":
    st.title("Interactive AI Tutorials")

    st.markdown("""
    ## Interactive AI Tutorials in Finance

    1. **ChatGPT/AI for Finance Professionals: Investing & Analysis**
        - [AI For Everyone](https://www.udemy.com/course/chatgpt-for-finance-professionals-investing-analysis/?couponCode=LETSLEARNNOWPP)
                
    2. **Coursera - AI in Finance**
       - [AI For Everyone](https://www.coursera.org/learn/ai-for-everyone)

    3. **Udacity - AI in Trading**
       - [AI for Trading](https://www.udacity.com/course/ai-for-trading--nd880)

    4. **DataCamp - Machine Learning for Finance in Python**
       - [Machine Learning for Finance in Python](https://www.datacamp.com/courses/machine-learning-for-finance-in-python)

    ## Additional Interactive Tutorials

    1. **Google's Machine Learning Crash Course**
       - [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

    2. **Kaggle Learn**
       - [Kaggle Learn](https://www.kaggle.com/learn)

    3. **Fast.ai**
       - [Fast.ai Courses](https://course.fast.ai/)

    4. **Codecademy's Data Science and Machine Learning Courses**
       - [Codecademy AI Courses](https://www.codecademy.com/catalog/subject/data-science)

    5. **Coursera**
       - [Coursera AI Courses](https://www.coursera.org/browse/data-science/machine-learning)

    6. **DataCamp**
       - [DataCamp AI Courses](https://www.datacamp.com/)
    """)
