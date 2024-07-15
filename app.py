import openai
import streamlit as st
import pandas as pd
import re
import numpy as np
import json
import uuid
import time
import os
import io
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
def categorize_values(series, exclude_columns):
    if series.name in exclude_columns:
        return series
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
        duplicates='drop',
        include_lowest=True
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
        prompt += (
            " You are a sr. data analyst with 15 years of analytics experience."
            " Provide general insights from the dataset."
            " Remove duplicate data, if any exist. Check for any inconsistencies or errors in the data and correct them."
            " If necessary, aggregate or group the data by relevant variables for analysis."
            " Create additional features from the dataset if you feel they can help with the analysis."
            " Focus on actionable insights and exclude any basic descriptive analytics or trivial observations."
            " Ensure the insights are concise and relevant, focusing on strategic decisions and impact."
            " Do not provide the Data Quality Observations section, Data Consistency, Duplication Check or descriptive analytics."
        )

    elif insight_type == "Trend Analysis":
        prompt += (
            " You are a sr. data analyst with 15 years of analytics experience."
            " Perform a detailed trend analysis focusing on identifying and analyzing trends over time."
            " Remove duplicate data, if any exist. Check for any inconsistencies or errors in the data and correct them."
            " If necessary, aggregate or group the data by relevant variables for analysis."
            " Create additional features from the dataset if you feel they can help with the analysis."
            " Include details such as monthly or quarterly trends, yearly, seasonal patterns, and year-over-year comparisons."
            " Specifically, identify key drivers of revenue growth or decline, highlight significant changes in key metrics."
            " Provide specific insights based on the identified trends."
            " Ensure the insights are concise and relevant, focusing on strategic decisions and impact."
            " Please avoid any basic descriptive analytics."
            " Avoid redundancy in insights and ensure all words are spelled correctly."
            " Do not provide the Data Quality Observations section, Data Consistency, Duplication Check or descriptive analytics."
        )

    elif insight_type == "Key Drivers of Revenue Growth":
        prompt += (
            " You are a sr. data analyst with 15 years of analytics experience."
            " Remove duplicate data, if any exist. Check for any inconsistencies or errors in the data and correct them."
            " If necessary, aggregate or group the data by relevant variables for analysis."
            " Create additional features from the dataset if you feel they can help with the analysis."
            " Analyze the dataset to identify the key drivers of revenue growth."
            " Provide detailed insights on which factors contributed most significantly to the increase or decrease in revenue."
            " Include specific segments, products, or discounts that had the largest impact."
            " Do not provide the Data Quality Observations section, Data Consistency, Duplication Check or descriptive analytics."
        )

    elif insight_type == "What-If Scenario Analysis":
        change = scenario['change']
        factor = scenario['factor']
        prompt += (
            f" Perform a what-if scenario analysis to evaluate the impact of decreasing the {factor} by {change}%."
            " Remove duplicate data, if any exist. Check for any inconsistencies or errors in the data and correct them."
            " If necessary, aggregate or group the data by relevant variables for analysis."
            " Create additional features from the dataset if you feel they can help with the analysis."
            f" Provide specific and tangible insights on how this decrease would affect the {factor} or any of the other fields in the dataset."
            " Avoid generic statements and focus on concrete data-driven insights."
            " Ensure the analysis is relevant, focusing on strategic decisions and impact."
            " Please avoid any basic descriptive analytics."
            " Focus only on the impacts without providing any conclusions, recommendations, or key metrics."
            " Do not include conclusions, recommendations, or key metrics."
            " You are the Sr. Director of Finance at Indeed with 15 years of experience in finance, strategy, and analytics."
            " Do not provide the Data Quality Observations section, Data Consistency, Duplication Check or descriptive analytics."
        )

    elif insight_type == "Region Analysis":
        prompt += (
            " Perform a detailed region analysis focusing on the different fields such as City, State, District, Country, Municipal, Zone, and Municipality."
            " Remove duplicate data, if any exist. Check for any inconsistencies or errors in the data and correct them."
            " If necessary, aggregate or group the data by relevant regional variables for analysis."
            " Create additional features from the dataset if you feel they can help with the analysis."
            " Include details such as regional performance, regional trends, and significant differences between regions."
            " Specifically, identify key areas of high and low performance, highlight significant changes in key metrics across regions,"
            " and provide insights on how regional differences impact overall performance."
            " Provide specific insights based on the identified regional differences."
            " Ensure the insights are concise and relevant, focusing on strategic decisions and impact."
            " Please avoid any basic descriptive analytics."
            " Do not provide the Data Quality Observations section, Data Consistency, Duplication Check or descriptive analytics."
        )

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

# Load existing mappings
mappings = load_mappings()

st.title("Hi, I'm Bheem 🐳 - your data insights companion. I can generate insights 🔍 and recommendations from your data in 60 seconds or less.")
st.markdown('<p style="font-size:14px; margin-top: 20px;">Bheem is powered by the latest version of ChatGPT: 4o and does not train on any of your data.</p>', unsafe_allow_html=True)
st.subheader("To begin, please upload 📑 your Google Sheet or CSV file below.")
st.markdown("**Note:** The most data ChatGPT can analyze is 700 rows or around 103KB.")

uploaded_files = st.file_uploader("Please choose CSV or Google Sheet files.", type=['csv', 'xlsx'], accept_multiple_files=True, key="ai_insights_upload")

# Clear session state if new files are uploaded
if 'uploaded_files' in st.session_state:
    if st.session_state['uploaded_files'] != uploaded_files:
        st.session_state.clear()

st.session_state['uploaded_files'] = uploaded_files

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
                # Load the CSV file
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
    # Convert date fields to datetime and sort
    date_column = None

    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
        df = df.sort_values(by='Year')
        date_column = 'Year'
    else:
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            # Sort by the first date column found
            date_column = date_columns[0]
            df = df.sort_values(by=date_column)
    # Identify columns with 'Timestamp' in the name
    timestamp_columns = [col for col in df.columns if 'timestamp' in col.lower()]
    if timestamp_columns:
        for col in timestamp_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        date_column = timestamp_columns[0]
        df = df.sort_values(by=date_column)
    
    if len(df) > 700:
        df = df.head(700)
        st.session_state['df'] = df
    else:
        st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    exclude_columns = ['Year', 'Date', 'Month Number', 'Month Name', 'Timestamp'] + date_columns if 'date_columns' in locals() else []

    numeric_columns = []
    for col in df.columns:
        if col not in exclude_columns:
            first_value = df[col].iloc[0]
            if not any(c.isalpha() for c in str(first_value)):
                df[col] = df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)))
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
        exclude_columns_for_labels = ['Year'] 
        for col in numeric_columns:
            categorized_df[col] = categorize_values(filtered_df[col], exclude_columns_for_labels)

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

        if 'selected_insight' not in st.session_state:
            st.session_state['selected_insight'] = None
        if 'analysis_requested' not in st.session_state:
            st.session_state['analysis_requested'] = False
        if 'data_confirmed' not in st.session_state:
            st.session_state['data_confirmed'] = False

        available_insights = ["General Insights"]

        trend_columns = {'Date', 'date',
                         'Month', 'month', 
                         'Month Number', 'month number', 
                         'Month Name', 'month name', 
                         'Year', 'year', 
                         'Timestamp', 'timestamp'}
        if any(any(col in column_name for col in trend_columns) for column_name in df.columns):
            available_insights.append("Trend Analysis")

        key_drivers_columns = {'Gross Sales', 'gross sales', 
                               'Sales', 'sales',
                               'Date','date',
                               'Profit', 'profit'}
        if any(any(col in column_name for col in key_drivers_columns) for column_name in df.columns):
            available_insights.append("Key Drivers of Revenue Growth")

        regions_columns = {'State', 'state', 
                           'City', 'city', 
                           'District','district',  
                           'Country', 'country',
                           'Region', 'region',
                           'Zone', 'zone'}
        if any(any(col in column_name for col in regions_columns) for column_name in df.columns):
            available_insights.append("Region Analysis")

        exclude_factors = {'Month Number', 'Month Name', 'Year'}
        numeric_factors = [col for col in categorized_df.columns if set(categorized_df[col].unique()).issubset({'Much Below Avg', 'Below Avg', 'Avg', 'Above Avg', 'Much Above Avg'}) and col not in exclude_factors]

        if numeric_factors:
            available_insights.append("What-If Scenario Analysis")

        selected_insight = st.selectbox("Please select the type of insights you would like to see and wait 20-30 seconds for the insights to generate:", available_insights, key="selected_insight_widget")

        scenario = None
        if selected_insight == "What-If Scenario Analysis":
            factor = st.selectbox("Select the factor to change:", numeric_factors, key="what_if_factor")
            change = st.number_input("Enter the percentage change (You can enter negative numbers like -2 or positive numbers like 2):", min_value=-100, max_value=100, value=5, key="what_if_change")
            scenario = {"factor": factor, "change": change}

        if st.button("Perform Analysis", key="perform_analysis"):
            st.session_state['analysis_requested'] = True
            st.session_state['data_str'] = combined_df.to_csv(index=False)
            st.session_state['selected_insight'] = selected_insight
            st.session_state['scenario'] = scenario

    except ValueError as e:
        st.error(f"Please provide Bheem with more data so he can give you your insights.")

if 'analysis_requested' in st.session_state and st.session_state['analysis_requested']:
    st.write("Confirm the data below before proceeding with the analysis.")
    st.dataframe(pd.read_csv(io.StringIO(st.session_state['data_str'])))

    top_row = pd.read_csv(io.StringIO(st.session_state['data_str'])).head(1)
    data_str = top_row.to_csv(index=False)
    display_data_prompt = f"Can you please produce a tabular version of the dataset you can see?\n{data_str}"
    with st.spinner('Verifying the dataset with ChatGPT, please wait...'):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI data analyst."},
                {"role": "user", "content": display_data_prompt}
            ]
        )
    data_as_seen_by_chatgpt = response['choices'][0]['message']['content']
    st.write("Data as seen by ChatGPT:")
    st.text(data_as_seen_by_chatgpt)

    if st.button("Confirm Data", key="confirm_data"):
        st.session_state['data_confirmed'] = True

if 'data_confirmed' in st.session_state and st.session_state['data_confirmed']:
    progress_bar = st.progress(0)
    with st.spinner('Generating insights, please wait...'):
        for i in range(0, 50, 5):
            time.sleep(0.5)
            progress_bar.progress(i)
        insights = get_insights_from_openai(st.session_state['data_str'], model="gpt-4o", insight_type=st.session_state['selected_insight'], scenario=st.session_state['scenario'])
        for i in range(50, 100, 5):
            time.sleep(0.5)
            progress_bar.progress(i)
        if st.session_state['selected_insight'] != "What-If Scenario Analysis":
            st.markdown("**Conclusions and Recommendations**")
        decoded_insights = reverse_mappings(insights, mappings)
        st.markdown(decoded_insights)
        progress_bar.progress(100)

else:
    st.write("Please upload a file to proceed.")
