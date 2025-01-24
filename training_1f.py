import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Define a variable for the dataset of sample (directory number)
''' full dataset size'''
CT = 326 
CXR = 557

''' test dataset size '''
# CT =  20 
# CXR = 20 


dataset = f"{CT}_CT_{CXR}_CXR"
dataset_base_directory = f"/home/masresha/dataset/all_dataset"
generated_data_directory = f"/home/masresha/dataset/generatedfile/{CT}_CT_{CXR}_CXR_generated"
result_data_directory = f"/home/masresha/dataset/resultfile/{CT}_CT_{CXR}_CXR_resultfile"

# Check if the base directory exists
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
        return directory_path

# Check directories
ensure_directory_exists(dataset_base_directory, create_if_missing=False)  # Must exist
ensure_directory_exists(generated_data_directory, create_if_missing=True)  # Create if missing
ensure_directory_exists(result_data_directory, create_if_missing=True)  # Create if missing

print("All directory checks and setups are complete!")

# Load fused features and labels
def load_data(d_type, fussion_model, training_type):
    if training_type == 'fused_data':   
        load_dir = os.path.join(generated_data_directory,training_type, d_type,fussion_model)
        ensure_directory_exists(load_dir)
        X = np.load(ensure_directory_exists(os.path.join(load_dir, f"{d_type}_{fussion_model}_fused_features.npy")))
        y = np.load(ensure_directory_exists(os.path.join(load_dir, f"{d_type}_{fussion_model}_labels.npy")))
        return X, y
    elif training_type == 'separate_data':
        data_dir = os.path.join(generated_data_directory,'selected_features_csv',d_type)
        ensure_directory_exists(data_dir)
        data = pd.read_csv(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_clinical_data_selected_features_{fussion_model}.csv")))
        X = data.drop(columns=['label'])
        y = data['label']
        return X, y
    elif training_type == 'image_data':
        # /home/masresha/dataset/generatedfile/20_CT_20_CXR_generated/preprocessed/selected_features_image_npy/CT/CT_image_features.npy
        # /home/masresha/dataset/generatedfile/20_CT_20_CXR_generated/selected_features_image_npy/CT/CT_image_features.npy
        data_dir = os.path.join(generated_data_directory,'selected_features_image_npy',d_type)
        ensure_directory_exists(data_dir)
        X = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_features.npy")))
        y = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_labels.npy")))
        return X, y

def save_model(d_type, model_name, model, fussion_model, training_type):
    # Create 'saved_models' subfolder if it doesn't exist
    model_dir = os.path.join(generated_data_directory, training_type, 'saved_models',d_type)
    ensure_directory_exists(model_dir, create_if_missing=True)

    # Save model in 'saved_models' subfolder
    joblib.dump(model, os.path.join(model_dir,f"{d_type}_{training_type}_{fussion_model}_{model_name}_model_sample_size_{d_type}.pkl"))
    print(f"Model training and evaluation complete. Model saved as '{d_type}_{training_type}_{fussion_model}_{model_name}_model_sample_size_{d_type}.pkl'")

def export_evaluation_result(d_type, y_true, y_pred, model_name, fussion_model, inference_time_train, inference_time, evaluation_type, params, training_type):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=6)
    cm = confusion_matrix(y_true, y_pred)

    print(f"[{d_type}] [{fussion_model}] [{training_type}] {model_name} {evaluation_type} Results (Sample size = {dataset}):")
    print(f"[{d_type}] [{fussion_model}] [{training_type}] {evaluation_type} Accuracy:", acc)
    print(f"[{d_type}] [{fussion_model}] [{training_type}] {evaluation_type} inference_time_train:", inference_time_train)
    print(f"[{d_type}] [{fussion_model}] [{training_type}] {evaluation_type} inference_time_{evaluation_type}:", inference_time)
    print(f"{model_name} parameters: \n {str(params)}")

    print(f"[{d_type}] [{fussion_model}] {evaluation_type} Classification Report:", report, '\n')

    # Write results to a text file
    txt_dir = os.path.join(result_data_directory, training_type,'evaluation_result',d_type)
    ensure_directory_exists(txt_dir, create_if_missing=True)
    result_file = os.path.join(txt_dir, f"{d_type}_{fussion_model}_evaluation_results_{dataset}.txt")
    with open(result_file, "a") as file:
        file.write(f"\n========================\n[{d_type}]  [{training_type}] [{fussion_model}] {model_name} {evaluation_type} Results (Sample size = {dataset}):\n")
        file.write(f"[{model_name}{d_type}] {evaluation_type} parameters: {str(params)}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Accuracy: {acc}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} inference_time_train: {inference_time_train}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} inference_time_{evaluation_type}: {inference_time}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Classification Report: \n{report}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Confusion Matrix:\n")
        cm_df = pd.DataFrame(cm)  # Create a pandas DataFrame
        file.write(cm_df.to_string())  # Convert the DataFrame to a string
        file.write("\n")
    print(f"[{d_type}]  [{training_type}] [{fussion_model}] {model_name} {evaluation_type} results saved to {result_file}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'[{d_type}] [{fussion_model}] [{training_type}] {model_name} {evaluation_type} Confusion Matrix (Sample size = {dataset})')

    # Save confusion matrix
    plt.tight_layout()
    confusion_matrix_dir = os.path.join(result_data_directory, training_type,'evaluation_result',d_type,f'{d_type}_{fussion_model}_confusion_matrix')
    ensure_directory_exists(confusion_matrix_dir, True)
    confusion_matrix_file = os.path.join(confusion_matrix_dir, f"{d_type}_{fussion_model}_{model_name}_{evaluation_type}_confusion_matrix_sample_size_{dataset}.png")
    plt.savefig(confusion_matrix_file)
    plt.close()
    print(f"[{d_type}] [{fussion_model}] {model_name} {training_type} {evaluation_type} Confusion matrix saved to {confusion_matrix_file}")

# Training loop with validation
for d_type in ['CT', 'CXR']:
    print("👉👉👉👉 ", d_type, " 👈👈👈👈")

    feature_extraction_models = ['mi', 'rf', 'pca']

    for fussion_model in feature_extraction_models:
        # Define parameters
        parameters = {
            'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            'RandomForest': {'n_estimators': 100, 'random_state': 42},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            'CatBoost': {'iterations': 100, 'learning_rate': 0.1, 'depth': 3, 'random_state': 42, 'verbose': 0},
            'MLP': {'hidden_layer_sizes': (100,), 'max_iter': 500, 'random_state': 42},
            'SVM': {'kernel': 'rbf', 'C': 1, 'gamma': 'scale', 'probability': True},
            'LogisticRegression': {'max_iter': 500, 'random_state': 42},  # Shorter key for consistency
            'KNN': {'n_neighbors': 5}
        }

        # Define classifiers
        classifiers = {}

        classifiers['GradientBoosting'] = GradientBoostingClassifier(**parameters['GradientBoosting'])
        classifiers['RandomForest'] = RandomForestClassifier(**parameters['RandomForest'])
        classifiers['XGBoost'] = XGBClassifier(**parameters['XGBoost'])
        classifiers['LightGBM'] = LGBMClassifier(**parameters['LightGBM'])
        classifiers['CatBoost'] = CatBoostClassifier(**parameters['CatBoost'])
        classifiers['MLP'] = MLPClassifier(**parameters['MLP'])
        classifiers['SVM'] = SVC(**parameters['SVM'])
        classifiers['LogisticRegression'] = LogisticRegression(**parameters['LogisticRegression'])
        classifiers['KNN'] = KNeighborsClassifier(**parameters['KNN'])

        # for training_type in ['separate_data','fused_data']:
        for training_type in ['image_data']:
            print(f"⏩ Working on: {training_type} {fussion_model} training")
   
            # Load data
            X, y = load_data(d_type, fussion_model, training_type)
            
            # Step 1: Split into training (80%) and temporary set (20%)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2,random_state=42, stratify=y
            )

            # Step 2: Split the temporary set into testing (10%) and validation (10%)
            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )

            # Train and evaluate classifiers
            for idx, (model_name, model) in enumerate(classifiers.items(), start=1):
                print(f"[{idx}/{len(classifiers)}] [{training_type}] [{d_type}] [{fussion_model}] Training {model_name} with dataset size ({dataset})...")
                # Measure inference time
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                inference_time_train = end_time - start_time

                # Evaluate on test set
                start_time = time.time()
                y_test_pred = model.predict(X_test)
                end_time = time.time()
                inference_time = end_time - start_time

                export_evaluation_result(d_type, y_test, y_test_pred, model_name, fussion_model, inference_time_train, inference_time,"Test", parameters[model_name], training_type)

                # Evaluate on validation set
                start_time = time.time()
                y_val_pred = model.predict(X_val)
                end_time = time.time()
                inference_time = end_time - start_time
                export_evaluation_result(d_type, y_val, y_val_pred, model_name, fussion_model,inference_time_train,inference_time, "Validation", parameters[model_name], training_type)

                # Save the model
                save_model(d_type, model_name, model, fussion_model, training_type)
                print(f"{training_type} training completed and result is saved")    

print("All models evaluated and results saved.")

