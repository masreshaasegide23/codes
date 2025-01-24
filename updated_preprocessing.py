"""
# Libraries

## ## Install missing packages
"""

import os
import cv2
import sys
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel

from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define a variable for the dataset of sample (directory number)
''' full dataset size'''
CT = 326 
CXR = 557

''' test dataset size '''
# CT =  20 
# CXR = 20 

"""# Path setup"""

dataset_base_directory = f"/home/masresha/dataset/all_dataset"
generated_data_directory = f"/home/masresha/dataset/generatedfile/{CT}_CT_{CXR}_CXR_generated"

# Function to check if a directory exists
def ensure_directory_exists(directory_path, create_if_missing=False):
    if not os.path.exists(directory_path):
        if create_if_missing:  # Create the directory if flagged
            os.makedirs(directory_path)
            print(f"📁 Created directory: {directory_path}")
        else:  # Show error and exit if not allowed to create
            print(f"❌ ERROR: Directory does not exist: {directory_path}. Exiting...")
            sys.exit(1)  # Abort execution
    else:
        print(f"✅ Directory exists: {directory_path}")

# Check directories
ensure_directory_exists(dataset_base_directory, create_if_missing=False)  # Must exist
ensure_directory_exists(generated_data_directory, create_if_missing=True)  # Create if missing

"""# Defined function for Excel loading

### drop columns
"""

def drop_columns_case_insensitive(df, columns_to_drop):
    # Convert columns to lowercase for case-insensitive comparison
    for column in columns_to_drop:
        # Find the actual column name with correct capitalization
        col_to_drop = next((col for col in df.columns if col.lower() == column.lower()), None)
        if col_to_drop:
            df.drop(col_to_drop, axis=1, inplace=True)

"""### split and validate"""

def split_and_validate(value):
    """
    Splits a string containing two numeric parts separated by '/' and validates them.

    Args: value (any):
        The input value to be processed.
        - If the input is a string containing '/', the string will be split into two parts.
        - If the input is not a string or does not contain '/', the function will return NaN for both parts.

    Returns:  A pandas Series containing two elements:
        - sbp (float): The first numeric part (e.g., systolic blood pressure).
        - dbp (float): The second numeric part (e.g., diastolic blood pressure).

        If the input is invalid or conversion to float fails, both elements will be NaN.
    """
    # Check if the value is a string and contains '/'
    if isinstance(value, str) and '/' in value:
        parts = value.split('/')
        try:
            # Convert parts to float if possible, otherwise return NaN
            sbp = float(parts[0].strip()) if parts[0].strip().isdigit() else np.nan
            dbp = float(parts[1].strip()) if parts[1].strip().isdigit() else np.nan
            return pd.Series([sbp, dbp])
        except ValueError:
            return pd.Series([np.nan, np.nan])
    else:
        return pd.Series([np.nan, np.nan])

"""### data cleaning"""

def clean_data(df):
    """
    Cleans and preprocesses the given DataFrame by performing the following:
    1. Standardizes case for categorical columns.
    2. Fixes data inconsistencies and substitutes invalid values.
    3. Extracts numeric values from specific columns and renames them.
    4. Splits 'SBP/DBP' into separate columns and validates the data.
    5. Converts datetime values to day numbers where applicable.
    6. Replace ',' with '.'  and convert them to float64
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned and preprocessed DataFrame.
    """

    # 1. Solve by matching case
    df['marital status'] = df['marital status'].str.capitalize()
    # Rename the column if it exists, using a case-insensitive comparison
    if 'family history' in map(str.lower, df.columns):
      df.rename(columns={'family history': 'family history of lung disease'}, inplace=True)

    if 'family history of lung disease' in df.columns:
      df['family history of lung disease'] = df['family history of lung disease'].str.capitalize()
    df['employment'] = df['employment'].str.capitalize()
    df['anxiety'] = df['anxiety'].str.capitalize()
    df['comorbidity'] = df['comorbidity'].str.capitalize()
    df['chest pain'] = df['chest pain'].str.capitalize()
    df['fever'] = df['fever'].str.capitalize()
    df['alcohol consumption'] = df['alcohol consumption'].str.capitalize()

    # 2. Need Substitution
    df['chest pain'] = df['chest pain'].replace({'Non': 'No'})
    df['comorbidity'] = df['comorbidity'].replace({'Non': 'No'})
    df['sputum test'] = df['sputum test'].replace({'positvie': 'Positive', 'positive': 'Positive'})
    df['rr'] = df['rr'].replace({'na': np.nan, 'Na': np.nan})
    df['rr'] = df['rr'].replace({'21+AO3:AO19': 21}) # '21+AO3:AO19 -> 21 value in TB_CXR(1)
    df['rr'] = df['rr'].astype(float)
    df['spo2'] = df['spo2'].replace({'na': np.nan})
    df['spo2'] = df['spo2'].astype(float)


    # 3. Extract number and change header
    if 'duration illness' in df.columns:
      df['duration illness'] = df['duration illness'].str.extract(r'(\d+)').astype(float)
      df.rename(columns={'duration illness': 'duration illness (day)'}, inplace=True)

    if 'oxegen' in df.columns:
      df['oxegen'] = df['oxegen'].apply( lambda x: pd.to_numeric(str(x).split('l')[0].strip(), errors='coerce') if pd.notna(x) else np.nan)
      df.rename(columns={'oxegen': 'oxygen (lit/min)'}, inplace=True)

    df['glascocoma scale(gcs)'] = df['glascocoma scale(gcs)'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # 4. Apply the function to split and validate the 'sbp/dbp' column
    if 'sbp/dbp' in df.columns:
      df[['sbp', 'dbp']] = df['sbp/dbp'].apply(split_and_validate)
      df.drop('sbp/dbp', axis=1, inplace=True)

    # 5. Convert some value encoded in DATETIME format to a number
    # Change datetime.datetime to day value
    def extract_day(value):
            if isinstance(value, datetime):
                return value.day
            return value
    df['pdw(gsd)'] = df['pdw(gsd)'].apply(extract_day)

    # 6. Replace ',' with '.' in both columns and convert them to float64
    df['temp(°c)'] = df['temp(°c)'].astype(str).str.replace(',', '.').astype(float)
    df['rbs profile(mg/dl)'] = df['rbs profile(mg/dl)'].astype(str).str.replace(',', '.').astype(float)

    return df

"""### Preprocessing and cleaning of the excel data"""

def process_excel_data(file_path):
    """
    Processes an Excel file by performing various steps, including renaming columns,
    cleaning data, and updating the 'label' column with the name derived from the file name.
    """
    # Extract the Excel file name without the extension
    excel_name = os.path.splitext(os.path.basename(file_path))[0]

    # Extract the label name before the last underscore (_CT or _CXR)
    label_name = excel_name.rsplit('_', 1)[0]  # This ensures we handle cases like "Lung_Cancer_CT"
    folder_type = excel_name.rsplit('_', 1)[1]  # "CT" or "CXR"

    # Load the Excel file without using the first row as a header
    excel_data = pd.read_excel(file_path, header=None, engine='openpyxl')

    # Get the last three columns of the DataFrame
    last_two_columns = excel_data.columns[-3:]

    # Loop through the last two columns to check and update values in row 1
    for col in last_two_columns:
        # Check if the value in row 1 (second row) is NaN
        if pd.isna(excel_data.at[1, col]):
            # If NaN, copy the value from row 0 (first row) to row 1 (second row)
            excel_data.at[1, col] = excel_data.at[0, col]

    # Replace the value in the last column of row 1 with 'label'
    last_column = excel_data.columns[-1]
    excel_data.at[1, last_column] = 'label'

    # Replace all values in the 'label' column with the extracted label name
    excel_data.iloc[2:, last_column] = label_name

    # Replace the first column's header with 'image_name'
    first_column = excel_data.columns[0]
    excel_data.at[1, first_column] = 'image_name'

    # Update values in the first column with the file name as a prefix
    excel_data[first_column] = excel_data[first_column].apply(
        lambda x: f"{excel_name}{x[x.find('('):]}" if isinstance(x, str) and '(' in x else x
    )

    # Drop the first row (row 0), as it is no longer needed
    excel_data = excel_data.drop(index=0)

    # Set the second row (row 1) as the new header
    new_header = excel_data.iloc[0]  # Get the first row after dropping row 0
    excel_data = excel_data[1:]  # Remove the first row, which is now the header
    excel_data.columns = new_header  # Assign the new header to the columns

    # Reset the index of the DataFrame after dropping the first row
    excel_data.reset_index(drop=True, inplace=True)

    # Remove leading or trailing spaces from all header cells
    excel_data.columns = excel_data.columns.str.strip()
    # Remove any leading or trailing spaces or tabs from all cells
    excel_data = excel_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    # Remove escape characters like '\n' from headers
    excel_data.columns = excel_data.columns.str.replace(r'\n', '', regex=True)

    # Remove rows where all values are NaN (entire rows)
    excel_data = excel_data.dropna(how='all')

    # Remove rows where 'image_name' is NaN, empty string, or contains only whitespace
    excel_data = excel_data[excel_data['image_name'].notnull() & (excel_data['image_name'].str.strip() != '')]

    # Rename the column 'Ches pain' to 'Chest pain'
    excel_data.rename(columns={'Ches pain': 'chest pain'}, inplace=True)

    #convert all columns name to lower
    excel_data.columns = excel_data.columns.str.lower()

    '''
      DELETE COLUMNS
    '''
    if folder_type == 'CT':
      # print(excel_data.columns)
      '''
        [Alcohol Usage] : Dropped because of having 10312 NaN values
        [Histopathology] : Dropped because relevance since it is obtained from cell sample
        [Radiologic Findings] : Dropped since it is textual data
      '''
      columns_to_drop = ['Alcohol Usage', 'Histopathology', 'Radiologic Findings']
      drop_columns_case_insensitive(excel_data, columns_to_drop)

    elif folder_type == 'CXR':
      '''
        [Alcohol Usage] : Dropped because of having 13748 NaN values
        [HIV screen result] : Dropped because of having 13191 NaN values
        [Histopathology] : Dropped because relevance since it is obtained from cell sample
        [Radiologic Findings] : Dropped since it is textual data
      '''
      columns_to_drop = ['Alcohol Usage', 'HIV screen result', 'Histopathology', 'Radiologic Findings']
      drop_columns_case_insensitive(excel_data, columns_to_drop)

    # clean the data
    excel_data = clean_data(excel_data)

    # move label column to the end
    column_to_move = 'label'
    excel_data = excel_data[[col for col in excel_data.columns if col != column_to_move] + [column_to_move]]
    # Return the processed DataFrame
    return excel_data

"""### Load Excel data"""

def load_data(base_path):
    """
    Loads clinical information for specified categories.

    The function performs the following:
    1. Iterates through categories and subfolders (CT and CXR).
    2. Reads the corresponding Excel file and counts its rows.
    3. Counts unique rows in the 'image_name' column of the Excel data.

    Args:
        base_path (str): The base directory containing category folders.

    Returns:
         clinical_data: DataFrame of clinical data
    """
    data = defaultdict(dict)

    # Get all parent categories like Asthma, COPD, etc.
    categories = [category for category in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, category))]
    total_categories = len(categories)  # Total number of categories

    for index, category in enumerate(categories, start=1):  # Use enumerate to get the index
        category_path = os.path.join(base_path, category)
        print(f"[{index}/{total_categories}] Processing... {category}")
        # if category != 'Lung_Cancer':
        #     continue
        # Load Excel files for both CT and CXR for the category
        for folder in ['CT', 'CXR']:
            folder_path = os.path.join(category_path, f'{category}_{folder}')
            print(f"\tWorking in folder... {category}_{folder}")

            # Read Excel file for this category (CT or CXR)
            excel_file = os.path.join(category_path, f'{category}_{folder}.xlsx')

            if os.path.exists(excel_file):
                print(f"\t\tLoading Excel... {category}_{folder}.xlsx")
                excel_data = process_excel_data(excel_file)

                # Count total rows and unique rows in 'image_name' column
                excel_row_count = len(excel_data)  # Total row count
                if 'image_name' in excel_data.columns:
                    unique_row_count = excel_data['image_name'].nunique()
                else:
                    print(f"\t\t⚠️ Warning: 'image_name' column not found in {category}_{folder}.xlsx")
                    unique_row_count = 0

                print(f"\t\tRows in Excel: {excel_row_count}, Unique 'image_name' rows: {unique_row_count}")
            else:
                print(f"❌❌ Error: Excel file not found for {category} {folder}")
                continue

            # Store clinical data for the current category
            data[category][folder] = {
                'clinical_data': excel_data,
            }

    return data

"""# Apply data loading"""

# Load the data
data = load_data(dataset_base_directory)

"""# Drop rows with corroputed image"""

# Filter out the row with 'Asthma_CT(17)' in the 'image_name' column
data['Asthma']['CT']['clinical_data'] = data['Asthma']['CT']['clinical_data'][
    data['Asthma']['CT']['clinical_data']['image_name'] != 'Asthma_CT(17)'
]

# Filter out rows with specified values in the 'image_name' column
data['Lung_Cancer']['CT']['clinical_data'] = data['Lung_Cancer']['CT']['clinical_data'][
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(74)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(76)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(1021)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(192)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(424)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(487)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(491)') &
    (data['Lung_Cancer']['CT']['clinical_data']['image_name'] != 'Lung_Cancer_CT(548)')
]

# Filter out the row with 'Pneumonia_CXR(2768)' in the 'image_name' column
data['Pneumonia']['CXR']['clinical_data'] = data['Pneumonia']['CXR']['clinical_data'][
    data['Pneumonia']['CXR']['clinical_data']['image_name'] != 'Pneumonia_CXR(2768)'
]

"""# Select a given number of rows based on the given CT and CXR value"""

for condition, modalities in data.items():
    for folder, folder_data in modalities.items():
        clinical_data = folder_data.get('clinical_data')
        if isinstance(clinical_data, pd.DataFrame):
            max_rows = len(clinical_data)
            n  = globals()[folder]
            if n > max_rows:
                n = max_rows  # Avoid errors by limiting to available rows
            data[condition][folder]['clinical_data'] = clinical_data.iloc[:n]
            # data[condition][folder]['clinical_data'] = clinical_data.sample(n)

"""## print number of rows for each data"""

# Loop through each top-level key in the data dictionary
for disease in data.keys():
    # Loop through each sub-key (e.g., 'CT' and 'CXR') under each disease
    for modality in data[disease].keys():
        # Extract the clinical_data DataFrame
        clinical_data = data[disease][modality]['clinical_data']

        # Print the disease, modality, and number of rows in the clinical_data
        print(f"{disease} - {modality}: {len(clinical_data)} rows")

"""# Save clinical data as csv"""

# Loop through each key in the dictionary structure
for disease, modalities in data.items():
    for modality, details in modalities.items():
        # Extract the clinical_data DataFrame
        clinical_data = details['clinical_data']

        # Define the output file name (e.g., 'TB_CT_clinical_data.csv')
        output_dir = os.path.join(generated_data_directory, 'raw_clinical_data_csv',f'{modality}')
        ensure_directory_exists(output_dir, create_if_missing=True)  # Create if missing
        filename = f"{disease}_{modality}_clinical_data.csv"
        filepath = os.path.join(output_dir, filename)

        # Save the DataFrame to a CSV file
        clinical_data.to_csv(filepath, index=False)

        print(f"Saved {filename}")

"""# Load clinical data from CSV"""

# Define the directory where the CSV files are stored
clinical_data_csv_dir = os.path.join(generated_data_directory, 'raw_clinical_data_csv')

# Initialize a new dictionary to store the loaded data
data = {}

# Loop through each file in the clinical_data_csv directory
for modality in os.listdir(clinical_data_csv_dir):
    csv_dir = os.path.join(clinical_data_csv_dir,modality)

    for filename in os.listdir(csv_dir):
        print(modality, filename)
        if filename.endswith(".csv"):
            # Parse the filename to get the disease and modality
            disease_modality, _, _ = filename.rsplit('_', 2)  # e.g., "Lung_Cancer_CT_clinical_data.csv"
            disease, modality = disease_modality.rsplit('_', 1)  # Split the last underscore for modality


            # Ensure the disease key exists in the dictionary
            if disease not in data:
                data[disease] = {}

            # Load the CSV file into a DataFrame
            filepath = os.path.join(csv_dir, filename)
            clinical_data = pd.read_csv(filepath)

            # Add the DataFrame to the appropriate location in the dictionary
            data[disease][modality] = {'clinical_data': clinical_data}

            print(f"Loaded {filename} into data[{disease}][{modality}]")

"""# define functions to preprocess the clinical data

### Normalize or Standardize Numerical Data
"""

def normalize_numerical_data(df):
  # Define numerical and categorical columns
  numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
  categorical_cols = df.select_dtypes(include=['object']).columns

  # Exclude Glascocoma scale(GCS) since it is already normalized
  numerical_cols = numerical_cols.difference(['glascocoma scale(gcs)'])

  # # Standardize numerical columns
  scaler = StandardScaler()
  df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
  return df

"""### handle missing values"""

def handle_missing_values(excel_data):
    """
    Handle missing values by imputing them with the most frequent value for both
    numerical and categorical columns.

    Parameters:
    - excel_data (DataFrame): DataFrame to process.

    Returns:
    - DataFrame: DataFrame with missing values imputed.
    """
    numerical_cols = excel_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = excel_data.select_dtypes(include=['object']).columns.difference(['image_name', 'label'])

    num_imputer = SimpleImputer(strategy="most_frequent")
    excel_data[numerical_cols] = num_imputer.fit_transform(excel_data[numerical_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    excel_data[categorical_cols] = cat_imputer.fit_transform(excel_data[categorical_cols])

    return excel_data

"""### map values"""

def map_values(value, no_categories, yes_categories):
      """
      Maps values in a column based on the provided mappings.

      Args:
          value: The current value in the column.
          no_categories: A list of categories to map "No"-like values.
          yes_categories: A list of categories to map "Yes"-like values.

      Returns:
          The mapped value.
      """
      if value == 'No':
          return np.random.choice(no_categories)  # Randomly select from No categories
      elif value == 'Yes':
          return np.random.choice(yes_categories)  # Randomly select from Yes categories
      else:
          return str(value).capitalize()

"""### encode
#### CT
"""

def encode_ct(df_ct):
  """
    Encode Categorical Variables
  """

  # Normalize case for binary categorical columns: Yes or No columns (20 columns)
  binary_columns = ['air pollution', 'allergy(penicillin)', 'anxiety', 'biomass fuel exposure', 'clubbing of the fingers', 'coughing','diarrhea', 'family history of lung disease', 'fatigue', 'headache', 'hemoptysis/coughing up blood','hoarseness of voice', 'loss of appetite', 'night sweats','shortness of breath/dyspnea','sore throat', 'swallowing difficulty', 'throat pain', 'weight loss', 'wheezing']
  df_ct[binary_columns] = df_ct[binary_columns].apply(lambda x: x.str.lower().map({'yes': 1.0, 'no': 0.0}))

  one_hot_encode_columns = df_ct.columns[df_ct.columns.isin(['gender','marital status', 'address' ,'employment','smoking history',  'sputum test', 'rh factor test', 'occult blood', 'comorbidity'])]
  df_ct = pd.get_dummies(df_ct, columns=one_hot_encode_columns, drop_first=True)

  # Define the mappings for Chest Pain
  no_categories = ['Asymptomatic']
  yes_categories = ['Mild', 'Moderate', 'Severe', 'Very severe']
  df_ct['chest pain'] = df_ct['chest pain'].apply(map_values, args=(no_categories, yes_categories))  # Apply the function to the Chest Pain column

  # Define the mappings for Fever
  no_categories = ['Normal', 'Low-grade']
  yes_categories = ['Moderate-grade', 'High-grade', 'Very high']
  df_ct['fever'] = df_ct['fever'].apply(map_values, args=(no_categories, yes_categories)) # Apply the function to the Fever column

  encoder = OrdinalEncoder(categories=[['Non', 'Mild', 'Heavy']])
  df_ct['alcohol consumption'] = encoder.fit_transform(df_ct[['alcohol consumption']])

  encoder = OrdinalEncoder(categories=[['No eduaction', 'Elementary School','High School',  'College and Above']])
  df_ct['educational status'] = encoder.fit_transform(df_ct[['educational status']])

  encoder = OrdinalEncoder(categories=[['Asymptomatic', 'Mild','Moderate', 'Severe', 'Very severe', ]])
  df_ct['chest pain'] = encoder.fit_transform(df_ct[['chest pain']])

  encoder = OrdinalEncoder(categories=[['Normal', 'Low-grade',  'Moderate-grade', 'High-grade', 'Very high']])
  df_ct['fever'] = encoder.fit_transform(df_ct[['fever']])

  class_mapping = { 'Asthma': 0, 'COPD': 1, 'COVID-19': 2, 'Lung_Cancer': 3, 'Normal': 4, 'Pneumonia': 5, 'Pneumothorax': 6, 'TB': 7 }
  df_ct['label'] = df_ct['label'].map(class_mapping)

  # move label to end
  column_to_move = 'label'
  df_ct = df_ct[[col for col in df_ct.columns if col != column_to_move] + [column_to_move]]

  return df_ct

"""#### CXR"""

def encode_cxr(df_cxr):
      """
        Encode Categorical Variables
      """

      # Normalize case for binary categorical columns # Yes or No columns
      binary_columns = [ 'family history of lung disease', 'air pollution', 'biomass fuel exposure', 'clubbing of the fingers', 'throat pain',   'diarrhea', 'sore throat', 'headache', 'hoarseness of voice', 'loss of appetite', 'anxiety', 'fatigue', 'allergy(penicillin)', 'wheezing', 'coughing', 'shortness of breath/dyspnea', 'hemoptysis/coughing up blood', 'swallowing difficulty', 'night sweats', 'weight loss']
      df_cxr[binary_columns] = df_cxr[binary_columns].apply(lambda x: x.str.lower().map({'yes': 1.0, 'no': 0.0}))

      # Encode categorical columns using one-hot encoding
      one_hot_encode_columns = df_cxr.columns[df_cxr.columns.isin(['address' , 'alcohol consumption', 'comorbidity','employment', 'gender','marital status', 'rh factor test', 'smoking history', 'sputum test', 'occult blood' ])]
      df_cxr = pd.get_dummies(df_cxr, columns=one_hot_encode_columns, drop_first=True)

      # Define the mappings for Chest Pain
      no_categories = ['Asymptomatic']
      yes_categories = ['Mild', 'Moderate', 'Severe', 'Very severe']
      df_cxr['chest pain'] = df_cxr['chest pain'].apply(map_values, args=(no_categories, yes_categories))  # Apply the function to the Chest Pain column

      # Define the mappings for Fever
      no_categories = ['Normal', 'Low-grade']
      yes_categories = ['Moderate-grade', 'High-grade', 'Very high']
      df_cxr['fever'] = df_cxr['fever'].apply(map_values, args=(no_categories, yes_categories)) # Apply the function to the Fever column

      encoder = OrdinalEncoder(categories=[['No eduaction', 'Elementary School','High School',  'College and Above']])
      df_cxr['educational status'] = encoder.fit_transform(df_cxr[['educational status']])

      encoder = OrdinalEncoder(categories=[['Asymptomatic', 'Mild','Moderate', 'Severe', 'Very severe', ]])
      df_cxr['chest pain'] = encoder.fit_transform(df_cxr[['chest pain']])

      encoder = OrdinalEncoder(categories=[['Normal', 'Low-grade',  'Moderate-grade', 'High-grade', 'Very high']])
      df_cxr['fever'] = encoder.fit_transform(df_cxr[['fever']])

     # Create a dictionary to map the classes to integer values
      class_mapping = { 'Asthma': 0, 'COPD': 1, 'COVID-19': 2, 'Lung_Cancer': 3, 'Normal': 4, 'Pneumonia': 5, 'Pneumothorax': 6, 'TB': 7 }
      df_cxr['label'] = df_cxr['label'].map(class_mapping)


      # move label column to the end
      column_to_move = 'label'
      df_cxr = df_cxr[[col for col in df_cxr.columns if col != column_to_move] + [column_to_move]]

      return df_cxr

"""### Shuffle"""

def shuffle_df(df):
      """
        Shuffle the dataset
      """
      # Shuffle the balanced dataset
      df = df.sample(frac=1, random_state=42).reset_index(drop=True)
      return df

"""# Apply data preprocessing on the each data"""

# Get the total number of classes-modality combinations
total_combinations = sum(len(modalities) for modalities in data.values())

# Initialize data for progress tracking
current_progress = 0

# Apply the function to each DataFrame
for classes, modalities in data.items():
    for modality, details in modalities.items():
        # Extract the clinical_data DataFrame
        clinical_data = details['clinical_data']

        # Apply the normalization function
        clinical_data = normalize_numerical_data(clinical_data)

        # Apply the missing value handle function
        clinical_data = handle_missing_values(clinical_data)

        # Apply data shuffle
        clinical_data = shuffle_df(clinical_data)

        # Update the nested dictionary with the transformed DataFrame
        data[classes][modality]['clinical_data'] = clinical_data

        # # Save the processed clinical data for each modality
        # output_dir = os.path.join(generated_data_directory, 'individual_preprocessed_csv')
        # ensure_directory_exists(output_dir, create_if_missing=True)  # Create if missing
        # file_path = os.path.join(output_dir, f"{classes}_{modality}_preprocessed_data.csv")
        # clinical_data.to_csv(file_path, index=False)
        # print(f"Preprocessed {modality} clinical data for {classes} saved to '{file_path}'.")

        # Progress tracking
        current_progress += 1
        print(f"Progress: {current_progress}/{total_combinations} "
              f"({(current_progress / total_combinations) * 100:.2f}%) - Processing data for {classes} - {modality}")

"""# Merge classes"""

# Create a new dictionary to hold merged data
merged_clinical_data = {
    'CT': pd.DataFrame(),
    'CXR': pd.DataFrame()
}

# Iterate through each category in the original data
for category in data.keys():
    for modality in ['CT', 'CXR']:
        if modality in data[category]:

            # Merge clinical_data (concatenate DataFrames)
            merged_clinical_data[modality] = pd.concat(
                [merged_clinical_data[modality], data[category][modality]['clinical_data']],
                ignore_index=True, sort=False
            )

# Iterate through the keys in the merged_clinical_data dictionary (CT, CXR)
for modality in merged_clinical_data.keys():
    # Get the corresponding DataFrame for the modality
    clinical_data = merged_clinical_data[modality]

    # Apply encoding based on modality
    if modality == 'CT':  # for CT
        # Apply the encode_ct function for CT modality
        clinical_data = encode_ct(clinical_data)
    else:
        # Apply the encode_cxr function for CXR modality
        clinical_data = encode_cxr(clinical_data)

    # Update the DataFrame with the encoded data
    merged_clinical_data[modality] = clinical_data

    print(f"Encoding applied to {modality} modality.")

# Optionally, print the updated columns to verify the changes
for modality in merged_clinical_data.keys():
    print(f"Columns in {modality} modality after encoding: {merged_clinical_data[modality].columns.tolist()}")

# Iterate through the keys in the merged_clinical_data dictionary (CT, CXR)
for modality in merged_clinical_data.keys():
    # Get the corresponding DataFrame for the modality
    data = merged_clinical_data[modality]

    # Check if there are any NaN values in the DataFrame
    if data.isnull().values.any():
        print(f"NaN values detected in {modality} modality.")

        # Print rows with NaN values
        print(f"Rows with NaN values in {modality}:")
        print(data[data.isnull().any(axis=1)])
    else:
        print(f"No NaN values detected in {modality} modality.")

"""### Save merged clinical data as csv"""

# Loop through the keys in the merged_clinical_data dictionary
for modality in merged_clinical_data.keys():
    # # Save the Preprocessed Data
    output_dir = os.path.join(generated_data_directory, 'merged_preprocessed_csv')
    ensure_directory_exists(output_dir, create_if_missing=True)  # Create if missing
    merged_clinical_data[modality].to_csv(os.path.join(output_dir, f"{modality}_preprocessed_data.csv"), index=False)
    print(f"Preprocessed {modality} clinical data saved to '{output_dir}{modality}_preprocessed_data.csv'.")

"""# load merged CSV data"""

file_path = {
    modality: os.path.join(generated_data_directory, 'merged_preprocessed_csv', f"{modality}_preprocessed_data.csv")
    for modality in ['CT', 'CXR']
}

for modality, path in file_path.items():
    if not os.path.exists(path):
        print(f"⛔ERROR⛔: {modality} CSV file does not exist at {path}.")

"""### Set image size"""

# Define a target dimension
TARGET_DIMENSIONS = (224, 224)

"""# define function for image data

### image preprocessing
"""

def preprocess_image(image, target_dimensions):
    """
    Resizes the image to the target dimensions and converts to grayscale.

    Args:
        image (numpy.ndarray): Input image (pixel array).
        target_dimensions (tuple): Target dimensions (width, height).

    Returns:
        numpy.ndarray: Processed image.
    """
    # Check if the image is already grayscale
    if len(image.shape) == 3:  # If it has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize the image
    resized_image = cv2.resize(image, target_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

"""### load image function"""

def load_image_data(path, image_type):
    """
    Loads and aligns DICOM images with clinical data from a merged CSV file.

    Args:
        path (str): The path to the CSV file with clinical data.
        image_type (str): The type of images to load ('CT' or 'CXR').

    Returns:
        dict: A dictionary containing:
            - 'images': NumPy array of preprocessed image arrays.
            - 'labels': NumPy array of image labels (image_name from the CSV).
            - 'clinical_data': DataFrame of clinical data rows corresponding to the returned images.
    """
    # Load the clinical data CSV
    clinical_data = pd.read_csv(path)

    # Ensure the image_name column exists
    if 'image_name' not in clinical_data.columns:
        raise ValueError("The CSV file must contain an 'image_name' column.")

    # Create a set of valid image names from the CSV
    valid_image_names = set(clinical_data['image_name'])

    # Initialize storage
    images = []
    labels = []
    aligned_clinical_data = []

    # Get all parent categories like COVID-19, TB, etc.
    categories = [category for category in os.listdir(dataset_base_directory) if os.path.isdir(os.path.join(dataset_base_directory, category))]
    total_categories = len(categories)

    for index, category in enumerate(categories, start=1):
        category_path = os.path.join(dataset_base_directory, category)
        print(f"[{index}/{total_categories}] Processing... {category}")

        # Only process the specified image type (CT or CXR)
        folder = image_type
        folder_path = os.path.join(category_path, f"{category}_{folder}")
        if not os.path.exists(folder_path):
            print(f"\t⚠️ Folder not found: {folder_path}. Skipping...")
            continue

        # Read DICOM files in the folder
        dicom_files = glob(os.path.join(folder_path, "*.dcm"))
        dicom_file_count = len(dicom_files)
        print(f"\t\tNumber of DICOM files in {folder}: {dicom_file_count}")

        # Process each DICOM file
        for dicom_file in tqdm(dicom_files, desc=f"\t\tProcessing {folder} DICOM files"):
            # Extract image_name
            image_name = os.path.basename(dicom_file).replace('.dcm', '')

            if image_name in valid_image_names:
                # Load the DICOM file
                dicom_data = pydicom.dcmread(dicom_file)
                image = preprocess_image(dicom_data.pixel_array, TARGET_DIMENSIONS)

                # Append to results
                images.append(image)
                labels.append(category)

                # Add the corresponding clinical data row
                aligned_clinical_data.append(clinical_data[clinical_data['image_name'] == image_name].iloc[0])
            # else:
            #     print(f"\t\t⚠️ No matching clinical data for image '{image_name}'. Skipping...")

    # Convert aligned clinical data to a DataFrame
    aligned_clinical_data = pd.DataFrame(aligned_clinical_data)

    # Reset the index of the DataFrame
    aligned_clinical_data.reset_index(drop=True, inplace=True)

    # Convert images and labels to NumPy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Normalize the Images
    images_np = images_np / 255.0

    # Print summary
    print(f"Final counts:")
    print(f"  Total Images Processed: {len(images_np)}")
    print(f"  Total Labels: {len(labels_np)}")
    print(f"  Total Clinical Data Rows: {len(aligned_clinical_data)}")

    # Return results in a dictionary
    return {
        'image': images_np,
        'label': labels_np,
        'clinical_data': aligned_clinical_data
    }

"""# Apply image load and preprocessing"""

# Create a new dictionary to hold merged data
data = {
    'CT': pd.DataFrame(),
    'CXR': pd.DataFrame()
}

for modality, path in file_path.items():
    print(f"👉👉{modality}👈👈")
    data[modality] = load_image_data(path, modality)
    print("\n\n")

"""### save the preprocessing image as npy file"""

# Save features and labels for later use
for modality in data.keys():
    save_dir = os.path.join(generated_data_directory, 'merged_preprocessed_image_npy',modality)
    ensure_directory_exists(save_dir, create_if_missing=True)

    np.save(os.path.join(save_dir,f"{modality}_image_data.npy"), data[modality]['image'])
    np.save(os.path.join(save_dir, f"{modality}_image_labels.npy"), data[modality]['label'])

    print(f"{modality} images saved to '{save_dir}{modality}_image_data.npy'.")
    print(f"{modality} Labels saved to '{save_dir}{modality}_image_labels.npy'.")

"""# Feature extraction

### clinical data feature extraction
"""

for modality, path in file_path.items():
    print(f"👉👉 Processing modality: {modality.upper()} 👈👈")

    clinical_data = data[modality]['clinical_data']
    print(f"✅ Clinical data loaded for {modality}.")

    # Separate features and target
    print("🔄 Separating features and target variables...")
    X = clinical_data.drop(columns=['image_name', 'label'])
    y = clinical_data['label']
    print(f"✅ Features and target separated. Number of samples: {len(X)}, Number of features: {X.shape[1]}")

    """#### Feature Selection Using Mutual Information"""
    print("🔍 Performing feature selection using Mutual Information...")
    k = 10  # Number of top features to select
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected_mi = selector.fit_transform(X, y)
    selected_features_mi = X.columns[selector.get_support()]
    print(f"✅ Top {k} features selected using Mutual Information: {list(selected_features_mi)}")

    """#### Feature Importance Using Random Forest"""
    print("🌲 Training Random Forest model for feature importance...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    importances = rf_model.feature_importances_
    threshold = 0.01  # Define a threshold for feature importance
    important_features = X.columns[importances > threshold]
    X_selected_rf = X[important_features]
    print(f"✅ Features selected using Random Forest (threshold > {threshold}): {list(important_features)}")

    """#### Dimensionality Reduction Using PCA"""
    print("📉 Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print(f"✅ Dimensionality reduction complete. Original features: {X.shape[1]}, Features after PCA: {X_reduced.shape[1]}")

    """#### Save Extracted Features"""
    print("💾 Saving extracted features...")
    save_dir = os.path.join(generated_data_directory, 'selected_features_csv',f'{modality}')
    ensure_directory_exists(save_dir, create_if_missing=True)

    pd.concat([pd.DataFrame(X_selected_mi, columns=selected_features_mi), clinical_data['label']], axis=1).to_csv(
    os.path.join(save_dir, f"{modality}_clinical_data_selected_features_mi.csv"), index=False)
    
    pd.concat([pd.DataFrame(X_selected_rf), clinical_data['label']], axis=1).to_csv(
    os.path.join(save_dir, f"{modality}_clinical_data_selected_features_rf.csv"), index=False)

    pd.concat([pd.DataFrame(X_reduced), clinical_data['label']], axis=1).to_csv(
    os.path.join(save_dir, f"{modality}_clinical_data_selected_features_pca.csv"), index=False)

    print(f"✅ Data saved:")
    print(f"   - Mutual Information features: {save_dir}{modality}_clinical_data_selected_features_mi.csv")
    print(f"   - Random Forest features: {save_dir}{modality}_clinical_data_selected_features_rf.csv")
    print(f"   - PCA-reduced features: {save_dir}{modality}_clinical_data_selected_features_pca.csv")

    print("🎉 Processing complete for", modality.upper())
    print("\n" + "=" * 50 + "\n")

"""### image data feature extraction"""

for modality in data.keys():
    print (modality)

    # Load Image Data with label
    images = data[modality]['image']
    labels = data[modality]['clinical_data']['label']

    # Reshape images to include channel dimension
    images = np.expand_dims(images, axis=-1)

    # Expand Grayscale Images to 3 Channels  # VGG16 expects 3-channel input, so replicate grayscale data across 3 channels
    images_rgb = np.repeat(images, 3, axis=-1)

    # Load the model without the top classification layer
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add a custom flattening layer for feature extraction
    flatten = Flatten()(vgg16.output)
    feature_extractor = Model(inputs=vgg16.input, outputs=flatten)

    # Extract Features from Images
    features = feature_extractor.predict(images_rgb, verbose=1)

    # Save features and labels for later use
    save_dir = os.path.join(generated_data_directory, 'selected_features_image_npy',modality)
    ensure_directory_exists(save_dir, create_if_missing=True)

    np.save(os.path.join(save_dir,f"{modality}_image_features.npy"), features)
    np.save(os.path.join(save_dir, f"{modality}_image_labels.npy"), labels)

    print(f"Features extracted and saved to '{save_dir}{modality}_image_features.npy'.")
    print(f"Labels saved to '{save_dir}{modality}_image_labels.npy'.")

"""### Data fusion"""

# Define prefixes and models
prefixes = ['CT', 'CXR']
models = ['clinical_data_selected_features_mi', 'clinical_data_selected_features_rf', 'clinical_data_selected_features_pca']

# Mapping for shorter model names
model_mapping = {
    'clinical_data_selected_features_mi': 'mi',
    'clinical_data_selected_features_rf': 'rf',
    'clinical_data_selected_features_pca': 'pca'
}

for prefix in prefixes:
    for model in models:
        # Load Extracted Features
        # Load clinical and imaging features
        clinical_features_file = f"{prefix}_{model}.csv"
        imaging_features_file = f"{prefix}_image_features.npy"
        labels_file = f"{prefix}_image_labels.npy"

        clinical_features = pd.read_csv(os.path.join(generated_data_directory,'selected_features_csv', prefix, clinical_features_file))
        imaging_features = np.load(os.path.join(generated_data_directory,'selected_features_image_npy', prefix, imaging_features_file))
        labels = np.load(os.path.join(generated_data_directory,'selected_features_image_npy', prefix, labels_file))  # Common labels for both datasets

        # Normalize the Features
        # Standardize clinical features
        scaler_clinical = StandardScaler()
        clinical_features_normalized = scaler_clinical.fit_transform(clinical_features)

        # Standardize imaging features
        scaler_imaging = StandardScaler()
        imaging_features_normalized = scaler_imaging.fit_transform(imaging_features)

        # Concatenate Features for Fusion
        # Combine clinical and imaging features
        fused_features = np.hstack((clinical_features_normalized, imaging_features_normalized))

        # Map model name to shorter name
        short_model = model_mapping[model]

        save_dir = os.path.join(generated_data_directory, 'fused_data',prefix,short_model)
        ensure_directory_exists(save_dir, create_if_missing=True)

        # Save Fused Features and Splits
        np.save(os.path.join(save_dir, f"{prefix}_{short_model}_fused_features.npy"), fused_features)
        np.save(os.path.join(save_dir, f"{prefix}_{short_model}_labels.npy"), labels)

        #  # Split the Dataset
        # # Step 1: Split into training (80%) and temporary set (20%)
        # X_train, X_temp, y_train, y_temp = train_test_split(
        #     fused_features, labels, test_size=0.2, random_state=42, stratify=labels
        # )

        # # Step 2: Split the temporary set into testing (10%) and validation (10%)
        # X_test, X_val, y_test, y_val = train_test_split(
        #     X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        # )

        # # Map model name to shorter name
        # short_model = model_mapping[model]

        # save_dir = os.path.join(generated_data_directory, 'fused_data',f'{prefix}')
        # ensure_directory_exists(save_dir, create_if_missing=True)

        # # Save Fused Features and Splits
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_fused_features_train.npy"), X_train)
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_fused_features_test.npy"), X_test)
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_fused_features_val.npy"), X_val)
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_labels_train.npy"), y_train)
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_labels_test.npy"), y_test)
        # np.save(os.path.join(save_dir, f"{prefix}_{short_model}_labels_val.npy"), y_val)

        print(f"Data fusion completed for {prefix}_{short_model}_model. Fused features and labels saved.")

print("Data fusion completed. Fused features and labels saved.")

