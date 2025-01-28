# import re
# import csv
# import os
# import sys


# def split_into_sections(content, separator="========================"):
#     """
#     Splits a text file into sections using the specified separator.
    
#     :param file_path: Path to the text file.
#     :param separator: The string used to separate sections in the file.
#     :return: A list of sections.
#     """
#     sections = []
#     # with open(file_path, "r") as file:
#     #     content = file.read()
    
#     # Split the content using the separator
#     sections = content.split(separator)
    
#     # Remove any empty sections or extra spaces
#     sections = [section.strip() for section in sections if section.strip()]
    
#     return sections
# def extract_data(file_content):
#     # Define the regex patterns to extract the required information
#     # [CT]  [image_data_only] [mi] Fully_Connected Validation Results (Sample size = 326_CT_557_CXR):
#     feature_extraction_model_pattern =  re.compile(r"\[.*?\]\s+\[.*?\]\s+\[(.*?)\]")
#     model_pattern = re.compile(r'\[(.*?)\] Validation parameters: (.*?)\n')
#     accuracy_pattern = re.compile(r'\[CT\] \[image_data_only\] Validation Accuracy: (.*?)\n')
#     inference_time_train_pattern = re.compile(r'\[CT\] \[image_data_only\] Validation inference_time_train: (.*?)\n')
#     inference_time_validation_pattern = re.compile(r'\[CT\] \[image_data_only\] Validation inference_time_Validation: (.*?)\n')
#     classification_report_pattern = re.compile(r'\[CT\] \[image_data_only\] Validation Classification Report:.*?accuracy.*?(\d+\.\d+).*?macro avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)', re.DOTALL)

#     # Initialize a list to store the extracted data
#     data = []
#     sections = []
#     # Split the file content into sections for each model
#     # sections = file_content.split('========================')
#     sections = file_content.split('========================')
#     sections = [section.strip() for section in sections if section.strip()]
#     # sections = split_into_sections(file_content)
#     # print (sections[9])

#     for section in sections:
#         model_match = model_pattern.search(section)
#         if model_match:
#             feature_extraction_name = feature_extraction_model_pattern.search(section).group(1)           
#             model_name = model_match.group(2)
#             accuracy_match = accuracy_pattern.search(section)
#             inference_time_train_match = inference_time_train_pattern.search(section)
#             inference_time_validation_match = inference_time_validation_pattern.search(section)
#             classification_report_match = classification_report_pattern.search(section)

#             if accuracy_match and inference_time_train_match and inference_time_validation_match and classification_report_match:
#                 accuracy = float(accuracy_match.group(1)) * 100
#                 precision = float(classification_report_match.group(2)) * 100
#                 recall = float(classification_report_match.group(3)) * 100
#                 f1_score = float(classification_report_match.group(4)) * 100
#                 inference_time_train = float(inference_time_train_match.group(1))
#                 inference_time_validation = float(inference_time_validation_match.group(1))

#                 # Append the extracted data to the list
#                 data.append([
#                     model_name,
#                     f"{accuracy:.2f}%",
#                     f"{precision:.2f}%",
#                     f"{recall:.2f}%",
#                     f"{f1_score:.2f}%",
#                     inference_time_train,
#                     inference_time_validation
#                 ])

#     return feature_extraction_name, data

# def export_to_csv(file_name,ext_name,data, output_file):
#     # Define the header
#     header = [ext_name.upper(), "Accuracy", "precision", "recall", "f1-score", "inference time train (s)", "inference time test (s)"]

#     # Write the data to a CSV file
#     with open(output_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([file_name])  # Write the file name
#         writer.writerow(header)  # Write the header
#         writer.writerows(data)   # Write the data rows
#         writer.writerows([[]] * 5) 

# def main():
#     check_path()

    
    
#     categories = [
#     category for category in os.listdir(result_data_directory)
#     if os.path.isdir(os.path.join(result_data_directory, category)) and category != 'logs']

#     total_categories = len(categories)  # Total number of categories

#     for index, category in enumerate(categories, start=1):  # Use enumerate to get the index
#         category_path = os.path.join(result_data_directory, category)
#         for result in ['evaluation_result']:
#             folder_path = os.path.join(category_path, result)
#             for modality in ['CT', 'CXR']:
#                 modality_path = os.path.join(folder_path, modality)

#                  # Check if the modality_path exists and is a directory
#                 if os.path.exists(modality_path) and os.path.isdir(modality_path):
#                     # Get the list of .txt files in the modality_path
#                     file_list = [os.path.join(modality_path, file) for file in os.listdir(modality_path) if file.endswith('.txt')]

#                     # Now you can use file_list as needed
#                     # print(f"Processing {modality_path}:")
#                     print(f"Found {len(file_list)} .txt files:")

#                     txt_dir = os.path.join(result_data_directory,'logs')
#                     ensure_directory_exists(txt_dir, create_if_missing=True)

#                     output_csv_file = os.path.join(txt_dir, f"{modality}_summary_evaluation_results.csv")
#                       # Clear the output file if it already exists
#                     if os.path.exists(output_csv_file):
#                         os.remove(output_csv_file)

#                     for file_name in file_list:
#                         # print(f"Processing file: {file_name}")
#                         with open(file_name, 'r') as file:
#                             file_content = file.read()

#                         # Extract the data
#                         ext_name, data = extract_data(file_content)

#                         # Export the data to a CSV file
#                         export_to_csv(file_name,ext_name, data, output_csv_file)

#                         # print(f"Data from {file_name} has been appended to {output_csv_file}")
#                 else:
#                     print(f"Directory not found: {modality_path}")
    

#     print(f"Data export done. Result stored in \n{txt_dir}")
    

# def ensure_directory_exists(directory_path, create_if_missing=False):
#     if not os.path.exists(directory_path):
#         if create_if_missing:  # Create the directory if flagged
#             os.makedirs(directory_path)
#             print(f"📁 Created directory: {directory_path}")
#         else:  # Show error and exit if not allowed to create
#             print(f"❌ ERROR: Directory does not exist: {directory_path}. Exiting...")
#             sys.exit(1)  # Abort execution
#     else:
#         # print(f"✅ Directory exists: {directory_path}")
#         return directory_path




# CT = 326
# CXR = 557



# def check_path():
#     # Check directories
#     ensure_directory_exists(result_data_directory, create_if_missing=False)  # Must exist

# if __name__ == "__main__":
#     main()

import re
import csv
import os
import sys


def split_into_sections(content, separator="========================"):
    """
    Splits a text file into sections using the specified separator.
    
    :param file_path: Path to the text file.
    :param separator: The string used to separate sections in the file.
    :return: A list of sections.
    """
    sections = []
    # with open(file_path, "r") as file:
    #     content = file.read()
    
    # Split the content using the separator
    sections = content.split(separator)
    
    # Remove any empty sections or extra spaces
    sections = [section.strip() for section in sections if section.strip()]
    
    return sections
def extract_data(file_content):
    # Define the regex patterns to extract the required information
    # [CT]  [image_data_only] [mi] Fully_Connected Validation Results (Sample size = 326_CT_557_CXR):
    feature_extraction_model_pattern =  re.compile(r"\[.*?\]\s+\[.*?\]\s+\[(.*?)\]")
    model_pattern = re.compile(r'\[(.*?)\] Validation parameters: (.*?)\n')
    accuracy_pattern = re.compile(r'\[.*?\] \[.*?\] Validation Accuracy: (.*?)\n')
    inference_time_train_pattern = re.compile(r'\[.*?\] \[.*?\] Validation inference_time_train: (.*?)\n')
    inference_time_validation_pattern = re.compile(r'\[.*?\] \[.*?\] Validation inference_time_Validation: (.*?)\n')
    classification_report_pattern = re.compile(r'\[.*?\] \[.*?\] Validation Classification Report:.*?accuracy.*?(\d+\.\d+).*?macro avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)', re.DOTALL)


    # Initialize a list to store the extracted data
    data = []
    sections = []
    # Split the file content into sections for each model
    # sections = file_content.split('========================')
    sections = file_content.split('========================')
    sections = [section.strip() for section in sections if section.strip()]
    sections = split_into_sections(file_content)
    # print (sections[0])
    for section in sections:
        model_match = model_pattern.search(section)
        # print(model_match)
        # print(model_match.group(2))
        if model_match:
            feature_extraction_name = feature_extraction_model_pattern.search(section).group(1)           
            model_name = model_match.group(2)
            accuracy_match = accuracy_pattern.search(section)
            # print(accuracy_match)
            inference_time_train_match = inference_time_train_pattern.search(section)
            inference_time_validation_match = inference_time_validation_pattern.search(section)
            classification_report_match = classification_report_pattern.search(section)

            if accuracy_match and inference_time_train_match and inference_time_validation_match and classification_report_match:
                accuracy = float(accuracy_match.group(1)) * 100
                precision = float(classification_report_match.group(2)) * 100
                recall = float(classification_report_match.group(3)) * 100
                f1_score = float(classification_report_match.group(4)) * 100
                inference_time_train = float(inference_time_train_match.group(1))
                inference_time_validation = float(inference_time_validation_match.group(1))


                # Append the extracted data to the list
                data.append([
                    model_name,
                    f"{accuracy:.2f}%",
                    f"{precision:.2f}%",
                    f"{recall:.2f}%",
                    f"{f1_score:.2f}%",
                    inference_time_train,
                    inference_time_validation
                ])

    # exit (0)
    return feature_extraction_name, data

def export_to_csv(file_name,ext_name,data, output_file):
    # Define the header
    header = [ext_name.upper(), "Accuracy", "precision", "recall", "f1-score", "inference time train (s)", "inference time test (s)"]

    # Write the data to a CSV file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_name])  # Write the file name
        writer.writerow(header)  # Write the header
        writer.writerows(data)   # Write the data rows
        writer.writerows([[]] * 5) 

def main():
    check_path()

    
    
    categories = [
    category for category in os.listdir(result_data_directory)
    if os.path.isdir(os.path.join(result_data_directory, category)) and category != 'logs']

    total_categories = len(categories)  # Total number of categories

    for index, category in enumerate(categories, start=1):  # Use enumerate to get the index
        category_path = os.path.join(result_data_directory, category)
        for result in ['evaluation_result']:
            folder_path = os.path.join(category_path, result)
            for modality in ['CT', 'CXR']:
                modality_path = os.path.join(folder_path, modality)

                 # Check if the modality_path exists and is a directory
                if os.path.exists(modality_path) and os.path.isdir(modality_path):
                    # Get the list of .txt files in the modality_path
                    file_list = [os.path.join(modality_path, file) for file in os.listdir(modality_path) if file.endswith('.txt')]

                    # Now you can use file_list as needed
                    # print(f"Processing {modality_path}:")
                    print(f"In {category} {modality} Found {len(file_list)} .txt files:")

                    txt_dir = os.path.join(result_data_directory,'logs')
                    ensure_directory_exists(txt_dir, create_if_missing=True)

                    output_csv_file = os.path.join(txt_dir, f"{modality}_summary_evaluation_results.csv")
                      # Clear the output file if it already exists
                    if os.path.exists(output_csv_file):
                        os.remove(output_csv_file)

                    for file_name in file_list:
                        # print(f"Processing file: {file_name}")
                        with open(file_name, 'r') as file:
                            file_content = file.read()

                        # Extract the data
                        ext_name, data = extract_data(file_content)

                        # Export the data to a CSV file
                        export_to_csv(file_name,ext_name, data, output_csv_file)

                        # print(f"Data from {file_name} has been appended to {output_csv_file}")
                else:
                    print(f"Directory not found: {modality_path}")
    exit (0)
    file_list = [
        'CT_mi_evaluation_results_326_CT_557_CXR.txt',
        'CT_None_evaluation_results_326_CT_557_CXR.txt',
        'CT_pca_evaluation_results_326_CT_557_CXR.txt',
        'CT_rf_evaluation_results_326_CT_557_CXR.txt'
    ]
    txt_dir = os.path.join(result_data_directory, training_type, 'evaluation_result', d_type)
    ensure_directory_exists(txt_dir, create_if_missing=True)
    output_csv_file = os.path.join(txt_dir, f"{d_type}_{feature_model}_summary_evaluation_results_{dataset}.csv")
    

    # Clear the output file if it already exists
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)

    for file_name in file_list:
        # Read the file content
        with open(file_name, 'r') as file:
            file_content = file.read()

        # Extract the data
        ext_name, data = extract_data(file_content)

        # Export the data to a CSV file
        export_to_csv(ext_name, data, output_csv_file)

        print(f"Data from {file_name} has been appended to {output_csv_file}")
        # break

def ensure_directory_exists(directory_path, create_if_missing=False):
    if not os.path.exists(directory_path):
        if create_if_missing:  # Create the directory if flagged
            os.makedirs(directory_path)
            print(f"📁 Created directory: {directory_path}")
        else:  # Show error and exit if not allowed to create
            print(f"❌ ERROR: Directory does not exist: {directory_path}. Exiting...")
            sys.exit(1)  # Abort execution
    else:
        # print(f"✅ Directory exists: {directory_path}")
        return directory_path




CT = 326
CXR = 557

result_data_directory = f"/home/masresha/dataset/resultfile/{CT}_CT_{CXR}_CXR_resultfile"

def check_path():
    # Check directories
    ensure_directory_exists(result_data_directory, create_if_missing=False)  # Must exist

if __name__ == "__main__":
    main()