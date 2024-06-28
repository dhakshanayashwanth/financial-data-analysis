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
from openai.error import RateLimitError

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
    bins = [-np.inf, mean - std, mean + std, np.inf]
    if len(np.unique(bins)) != len(bins):
        bins = np.unique(bins)
    categories = pd.cut(
        series,
        bins=bins,
        labels=['Below Avg', 'Avg', 'Above Avg'],
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

# Function to get insights from OpenAI
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
        except RateLimitError as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            st.warning(f"Rate limit reached for {model}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    st.error("Failed to get insights after multiple attempts. Please try again later.")
    return None

# Load existing mappings
mappings = load_mappings()

st.title("Hi, I'm Beam 🐳 - your data insights companion. I can generate insights 🔍 and recommendations from your data in 60 seconds or less.")
st.markdown('<p style="font-size:14px; margin-top: 20px;">Beam is powered by the latest version of ChatGPT: 4o and does not train on any of your data.</p>', unsafe_allow_html=True)
st.subheader("To begin, please upload 📑 your Google Sheet or CSV file below.")
st.markdown("**Note:** The most data ChatGPT can analyze is 500 rows or around 103KB.")

uploaded_file = st.file_uploader("Please choose a CSV or Google Sheet file.", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Check the file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'xlsx':
        # Load the Excel file and display sheet options
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.selectbox("Select the sheet to analyze", sheet_names)

        if st.button("Select Sheet"):
            st.session_state['selected_sheet'] = selected_sheet

    elif file_extension == 'csv':
        with st.spinner('Loading data...'):
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df

if 'selected_sheet' in st.session_state and file_extension == 'xlsx':
    with st.spinner('Loading data...'):
        df = pd.read_excel(uploaded_file, sheet_name=st.session_state['selected_sheet'])
        st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Remove leading/trailing spaces from string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    # Detect the date column
    date_column = None
    for col in df.columns:
        if re.search(r'date|month|year', col, re.IGNORECASE):
            try:
                pd.to_datetime(df[col], errors='raise')
                date_column = col
                break
            except (ValueError, TypeError):
                continue

    # Format the date column to show only the date part
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%m/%d/%Y')

    # Exclude date columns from further processing
    exclude_columns = [date_column] if date_column else []

    # Remove symbols from numeric columns (excluding date columns)
    numeric_columns = []
    for col in df.columns:
        if col not in exclude_columns:
            first_value = df[col].iloc[0]
            if not any(c.isalpha() for c in str(first_value)):
                df[col] = df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)).split('.')[0])  # Remove decimals
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_columns.append(col)

    # Create a copy for displaying the original data
    original_df = df.copy()

    # Allow user to filter the dataset using multiselects
    st.subheader("Optional: Filter your data.")
    st.markdown("**Note:** If you don't need to filter your dataset, please scroll to the bottom, select the analysis you need and click on the Perform Analysis button.")

    filter_columns = st.multiselect("Select columns to filter by", df.columns.tolist())
    filtered_df = df.copy()

    for col in filter_columns:
        unique_values = df[col].unique().tolist()
        selected_values = st.multiselect(f"Select values for {col}", unique_values, default=unique_values)
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    st.write("Original Data")
    st.dataframe(filtered_df)

    try:
        # Apply categorization rules
        categorized_df = filtered_df.copy()
        for col in numeric_columns:
            categorized_df[col] = categorize_values(filtered_df[col])

        # Apply code mappings to the original DataFrame
        encoded_df = filtered_df.copy()
        for col in filtered_df.columns:
            if col not in numeric_columns and col not in exclude_columns:
                if col not in mappings:
                    mappings[col] = {}
                encoded_df[col] = filtered_df[col].apply(lambda x: generate_code(x, mappings[col]))

        # Save the updated mappings
        save_mappings(mappings)

        # Combine mapped and categorized data
        combined_df = encoded_df.copy()
        for col in numeric_columns:
            combined_df[col] = categorized_df[col]

        # Display the combined data
        st.write("Mapped and Categorized Data")
        st.dataframe(combined_df)

        # Determine available insights based on the columns present in the dataset
        available_insights = ["General Insights"]

        trend_columns = {'Date', 'Month Number', 'Month Name', 'Year'}
        if any(col in df.columns for col in trend_columns):
            available_insights.append("Trend Analysis")

        key_drivers_columns = {'Gross Sales', 'Sales', 'Date', 'Profit'}
        if any(col in df.columns for col in key_drivers_columns):
            available_insights.append("Key Drivers of Revenue Growth")

        # Determine numeric columns for What-If Scenario Analysis
        exclude_factors = {'Month Number', 'Month Name', 'Year'}
        numeric_factors = [col for col in categorized_df.columns if set(categorized_df[col].unique()).issubset({'Below Avg', 'Avg', 'Above Avg'}) and col not in exclude_factors]

        if numeric_factors:
            available_insights.append("What-If Scenario Analysis")

        # Show the dropdown for insight selection
        selected_insight = st.selectbox("Please select the type of insights you would like to see and wait 20-30 seconds for the insights to generate:", available_insights)

        scenario = None
        if selected_insight == "What-If Scenario Analysis":
            factor = st.selectbox("Select the factor to change:", numeric_factors)
            change = st.number_input("Enter the percentage change (You can enter negative numbers like -2 or positive numbers like 2):", min_value=-100, max_value=100, value=5)
            scenario = {"factor": factor, "change": change}

        if selected_insight and (selected_insight != "What-If Scenario Analysis" or scenario):
            if st.button("Perform Analysis"):
                # Initialize progress bar
                progress_bar = st.progress(0)

                # Prepare the data to send to ChatGPT for insights
                with st.spinner('Generating insights, please wait...'):
                    combined_df_str = combined_df.to_csv(index=False)

                    for i in range(0, 50, 5):
                        time.sleep(0.5)  # Simulating data processing time
                        progress_bar.progress(i)

                    insights = get_insights_from_openai(combined_df_str, model="gpt-4o", insight_type=selected_insight, scenario=scenario)
                    
                    for i in range(50, 100, 5):
                        time.sleep(0.5)  # Simulating data processing time
                        progress_bar.progress(i)

                    if selected_insight != "What-If Scenario Analysis":
                        st.markdown("**Conclusions and Recommendations**")

                    # Reverse the mappings in the insights
                    decoded_insights = reverse_mappings(insights, mappings)
                    st.markdown(decoded_insights)
                    
                    # Complete the progress bar
                    progress_bar.progress(100)
    except ValueError as e:
        st.error(f"Please provide Beam with more data so he can give you your insights.")

else:
    st.write("Please upload a file to proceed.")
