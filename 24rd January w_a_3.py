import cufflinks as cf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objs as go



# rest of the script
from dateutil.parser import parse

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

cf.set_config_file(colorscale='plotly', world_readable=True)
st.set_page_config(page_title='My Plotly Dashboard', page_icon=':chart_with_upwards_trend:', layout='wide')


# Extra options


def duplicate_clean ():
    if (not df.duplicated().any()):
        st.write("There are no duplicate rows in the data.")
    else:
        st.write("Duplicate rows found, what would you like do? (pick one of the following options)")
        st.write("(1) Keep all duplicates")
        st.write("(2) Drop all duplicates")
        st.write("(3) Drop all rows except the first one")
        dup_option = st.selectbox("Select option", ["(1) Keep all duplicates", "(2) Drop all duplicates", "(3) Drop all rows except the first one"])
        if dup_option == "(1) Keep all duplicates":
            return
        elif dup_option == "(2) Drop all duplicates":
            df.drop_duplicates(inplace=True)
            return
        elif dup_option == "(3) Drop all rows except the first one":
            df.drop_duplicates(keep='first',inplace=True)
            return


def clean_non_numerical (col):
    def is_date(string):
        try: 
            parse(string)
            return True
        except ValueError:
            return False
    
    if (len(df[col].unique()) > 30):
        duplicate_clean(col)
    
    for i, row_value in df[col].head(5).iteritems():
        if is_date(df[col][i]):
            return
    
    if ('country' in col) or ('COUNTRY' in col) or ('Country' in col):
        invalid_df_us = df[(df[col] == 'United States') | (df[col] == 'United States of America') | (df[col] == 'USA') | (df[col] == 'usa') | (df[col] == 'us')]
        if invalid_df_us.shape[0] > 0:
            st.write("There are " + str(invalid_df_us.shape[0])+ " alternate references to the US. These age entries will be all be set to US.")
            df.loc[(df[col] == 'United States') | (df[col] == 'United States of America') | (df[col] == 'USA') | (df[col] == 'usa') | (df[col] == 'us'), [col]] = 'US'
        invalid_df_uk = df[(df[col] == 'United Kingdom') | (df[col] == 'uk')]
        if invalid_df_us.shape[0] > 0:
            st.write
            st.write("There are " + str(invalid_df_us.shape[0])+ " alternate references to the UK. These age entries will be all be set to UK.")
            df.loc[(df[col] == 'United Kingdom') | (df[col] == 'uk'), [col]] = 'UK'
        
    unique_vals = ['None' if x is np.nan else x for x in df[col].unique()]
    unique_vals = ['None' if v is None else v for v in unique_vals]
    st.write("Initial set of unqiue entries in '" + col + "' column: "+ ', '.join(unique_vals))
    df[col] = df[col].replace(r"\s+$", "", regex=True)
    df[col] = df[col].replace(r"^\s+", "", regex=True)
    unique_vals = ['None' if x is np.nan else x for x in df[col].unique()]
    unique_vals = ['None' if v is None else v for v in unique_vals]
    st.write("Final set of unqiue entries in '" + col + "' column: "+ ', '.join(unique_vals))
    standardize = st.selectbox("Would you like to standardize the entries in this column?", ["Yes", "No"])
    if (standardize == "Yes"):
        standard_map = {}
        for value in unique_vals:
            new_val = st.text_input(f"Enter the standardized value for {value}")
            standard_map[value] = new_val
        df[col].replace(standard_map, inplace=True)



def handle_missing_values(df):
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.shape[0] == 0:
        st.write("There are no missing values in the data.")
    else:
        st.write("Missing values per column:")
        st.write(missing_values)
        strategy = st.selectbox("Select strategy to handle missing values", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode","KNN Imputation"])
        if strategy == "Drop rows":
            df.dropna(inplace=True)
        elif strategy == "KNN Imputation":
            df_numerical = df.select_dtypes(include=np.number)
            if df_numerical.shape[1] == 0:
                st.warning("KNN Imputation cannot be applied as all columns are non-numerical.")
                return
            st.write("Please enter number of nearest neighbour to use for imputation")
            k = int(st.number_input('Number of nearest neighbour :'))
            imputer = KNNImputer(n_neighbors=k)
            df_numerical = pd.DataFrame(imputer.fit_transform(df_numerical), columns = df_numerical.columns)
            df.update(df_numerical)
        else:
            for column in missing_values.index:
                if strategy == "Fill with mean":
                    mean = df[column].mean()
                    df[column].fillna(mean, inplace=True)
                elif strategy == "Fill with median":
                    median = df[column].median()
                    df[column].fillna(median, inplace=True)
                elif strategy == "Fill with mode":
                    mode = df[column].mode()[0]
                    df[column].fillna(mode, inplace=True)




def create_plots(df):


    # Histogram
    if st.checkbox("Histogram"):
        all_columns_names = df.columns.tolist()
        selected_column = st.selectbox("Select a column to plot histogram", all_columns_names)
        fig = px.histogram(df, x=selected_column)
        st.plotly_chart(fig)

    # Bar Chart
    if st.checkbox("Bar Chart"):
        all_columns_names = df.columns.tolist()
        selected_column_x = st.selectbox("Select a column for x-axis", all_columns_names)
        selected_column_y = st.selectbox("Select a column for y-axis", all_columns_names)
        fig = px.bar(df, x=selected_column_x, y=selected_column_y)
        st.plotly_chart(fig)

    # Scatter Plot
    if st.checkbox("Scatter Plot"):
        all_columns_names = df.columns.tolist()
        selected_column_x = st.selectbox("Select a column for x-axis ", all_columns_names)
        selected_column_y = st.selectbox("Select a column for y-axis ", all_columns_names)
        fig = px.scatter(df, x=selected_column_x, y=selected_column_y)
        st.plotly_chart(fig)

    # Correlation Heatmap
    if st.checkbox("Correlation Heatmap"):
        corr = df.corr()
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Heatmap(z=corr, 
                                 x=corr.columns,
                                 y=corr.columns,
                                 colorscale='Viridis'))
        fig.update_layout(title='Correlation Heatmap')
        st.plotly_chart(fig)

def perform_EDA(df):
    st.subheader("Exploratory Data Analysis")
    if st.checkbox("Show columns"):
        all_columns = df.columns.tolist()
        st.write("Columns:")
        st.write(all_columns)
    if st.checkbox("Show shape"):
        st.write("Shape:", df.shape)
    if st.checkbox("Show description"):
        st.write(df.describe())
    if st.checkbox("Show value counts"):
        st.write(df.iloc[:, 0].value_counts())
    if st.checkbox("Show missing values"):
        st.write(df.isna().sum())
    if st.checkbox("Show value range"):
        st.write(df.agg([min, max]))


def main():
    global df
    st.title("Automated Data Cleaning Web App")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        try:
            handle_missing_values(df)
        except Exception as e:
            st.error(e)
        st.write("Original data shape:", df.shape)
        columns = st.multiselect("Select the columns you would like to clean", df.columns)
        for col in columns:
            if (df[col].dtype == 'object'):
                clean_non_numerical(col)
            else:
                duplicate_clean()
        st.dataframe(df)
        create_plots(df)
        perform_EDA(df)


if __name__ == "__main__":
    main()
