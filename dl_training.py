# %%
import os
import sys
import time
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.svm import SVC
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D


# %% [markdown]
# Define a variable for the dataset of sample (directory number)

# %%
# full dataset size

CT = 326 
CXR = 557

# test dataset size 

# CT =  20 
# CXR = 20 

# %%
dataset = f"{CT}_CT_{CXR}_CXR"
dataset_base_directory = f"/home/masresha/dataset/all_dataset"
generated_data_directory = f"/home/masresha/dataset/generatedfile/{CT}_CT_{CXR}_CXR_generated"
result_data_directory = f"/home/masresha/dataset/resultfile/{CT}_CT_{CXR}_CXR_resultfile"

# %% [markdown]
# Check if the base directory exists<br>
# Function to check if a directory exists

# %%
def ensure_directory_exists(directory_path, create_if_missing=False):
    if not os.path.exists(directory_path):
        if create_if_missing:  # Create the directory if flagged
            os.makedirs(directory_path)
            # print(f"📁 Created directory: {directory_path}")
        else:  # Show error and exit if not allowed to create
            print(f"❌ ERROR: Directory does not exist: {directory_path}. Exiting...")
            sys.exit(1)  # Abort execution
    else:
        # print(f"✅ Directory exists: {directory_path}")
        return directory_path

# %% [markdown]
# Check directories

# %%
ensure_directory_exists(dataset_base_directory, create_if_missing=False)  # Must exist
ensure_directory_exists(generated_data_directory, create_if_missing=True)  # Create if missing
ensure_directory_exists(result_data_directory, create_if_missing=True)  # Create if missing
print("All directory checks and setups are complete!")

# %%
# Define folder path and ensure it exists
log_dir = os.path.join(result_data_directory, 'logs')
ensure_directory_exists(log_dir, create_if_missing=True)

# Configure logging
log_filename = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()  # For console output
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# %% [markdown]
# Load fused features and labels

# %%
def load_data(d_type, feature_model, training_type):
    if training_type == 'fused_data':   
        load_dir = os.path.join(generated_data_directory,training_type, d_type,feature_model)
        ensure_directory_exists(load_dir)
        X = np.load(ensure_directory_exists(os.path.join(load_dir, f"{d_type}_{feature_model}_fused_features.npy")))
        y = np.load(ensure_directory_exists(os.path.join(load_dir, f"{d_type}_{feature_model}_labels.npy")))
        return X, y
    elif training_type == 'clinical_data_only':
        data_dir = os.path.join(generated_data_directory,'selected_features_csv',d_type)
        ensure_directory_exists(data_dir)
        data = pd.read_csv(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_clinical_data_selected_features_{feature_model}.csv")))
        X = data.drop(columns=['label'])
        y = data['label']
        return X, y
    elif training_type == 'image_data_only':
        data_dir = os.path.join(generated_data_directory,'selected_features_image_npy',d_type)
        ensure_directory_exists(data_dir)
        X = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_features.npy")))
        y = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_labels.npy")))
        return X, y
    elif training_type == 'CNN':
        data_dir = os.path.join(generated_data_directory, 'merged_preprocessed_image_npy',d_type)
        ensure_directory_exists(data_dir)
        X = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_data.npy")))
        y = np.load(ensure_directory_exists(os.path.join(data_dir, f"{d_type}_image_labels.npy")))
        return X, y

# %%
def save_model(d_type, model_name, model, feature_model, training_type, save_type='plk'):
    # Create 'saved_models' subfolder if it doesn't exist
    model_dir = os.path.join(result_data_directory, training_type, 'saved_models',d_type)
    ensure_directory_exists(model_dir, create_if_missing=True)

    # Save model in 'saved_models' subfolder
    if save_type == 'pkl':
        joblib.dump(model, os.path.join(model_dir,f"{d_type}_{training_type}_{feature_model}_{model_name}_model_sample_size_{d_type}.pkl"))
        print(f"Model training and evaluation complete. Model saved as '{d_type}_{training_type}_{feature_model}_{model_name}_model_sample_size_{d_type}.pkl'")
    elif save_type == 'keras':
        model.save(os.path.join(model_dir, f"{d_type}_{training_type}_{feature_model}_{model_name}_model_sample_size_{d_type}.keras"))   
        print(f"Model training and evaluation complete. Model saved as '{d_type}_{training_type}_{feature_model}_{model_name}_model_sample_size_{d_type}.keras'")

# %%
def build_and_train_model(learning_type,X_train, y_train, X_val, y_val, model,model_name, feature_model, training_type, batch_size=32, epochs=50):
    if learning_type =='ML': 
        start_time = time.time()
        model.fit(X_train, y_train)
        inference_time_train = time.time() - start_time
        return model, inference_time_train
    elif learning_type == 'DL':
        # model = build_deep_learning_model(input_dim=X_train.shape[1], output_dim=len(np.unique(y_train)))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=1)
        inference_time_train = time.time() - start_time
        return model, inference_time_train, history

# %%
def evaluate_model(learning_type, model, X_test, y_test):
    if learning_type == 'ML':
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        end_time = time.time()
        inference_time_test = end_time - start_time
        return inference_time_test
    elif learning_type =="DL":
        start_time = time.time()
        dl_eval = model.evaluate(X_test, y_test, verbose=0)
        inference_time_test = time.time() - start_time
        return dl_eval, inference_time_test

# %%
def print_evaluation_summary(d_type, feature_model, training_type, model_name, evaluation_type, acc, inference_time_train, inference_time, report, params):
    """
    Print evaluation summary to the console.
    """
    print(f"[{d_type}] [{feature_model}] [{training_type}] {model_name} {evaluation_type} Results (Sample size = {dataset}):")
    print(f"[{d_type}] [{feature_model}] [{training_type}] {evaluation_type} Accuracy:", acc)
    print(f"[{d_type}] [{feature_model}] [{training_type}] {evaluation_type} inference_time_train:", inference_time_train)
    print(f"[{d_type}] [{feature_model}] [{training_type}] {evaluation_type} inference_time_{evaluation_type}:", inference_time)
    print(f"{model_name} parameters: \n {str(params)}")
    print(f"[{d_type}] [{feature_model}] {evaluation_type} Classification Report:", report, '\n')


# %%
def save_results_to_file(d_type, feature_model, training_type, model_name, evaluation_type, acc, inference_time_train, inference_time, report, cm, params):
    """
    Save the evaluation results to a text file.
    """
    txt_dir = os.path.join(result_data_directory, training_type, 'evaluation_result', d_type)
    ensure_directory_exists(txt_dir, create_if_missing=True)
    result_file = os.path.join(txt_dir, f"{d_type}_{feature_model}_evaluation_results_{dataset}.txt")
    
    with open(result_file, "a") as file:
        file.write(f"\n========================\n[{d_type}]  [{training_type}] [{feature_model}] {model_name} {evaluation_type} Results (Sample size = {dataset}):\n")
        file.write(f"[{model_name}{d_type}] {evaluation_type} parameters: {str(params)}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Accuracy: {acc}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} inference_time_train: {inference_time_train}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} inference_time_{evaluation_type}: {inference_time}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Classification Report: \n{report}\n")
        file.write(f"[{d_type}] [{training_type}] {evaluation_type} Confusion Matrix:\n")
        cm_df = pd.DataFrame(cm)  # Create a pandas DataFrame
        file.write(cm_df.to_string())  # Convert the DataFrame to a string
        file.write("\n")
    
    print(f"[{d_type}]  [{training_type}] [{feature_model}] {model_name} {evaluation_type} results saved to {result_file}")


# %%
def plot_and_save_confusion_matrix(d_type, feature_model, model_name, evaluation_type, cm, dataset, training_type):
    """
    Plot and save the confusion matrix.
    """
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'[{d_type}] [{feature_model}] [{training_type}] {model_name} {evaluation_type} \nConfusion Matrix (Sample size = {dataset})')

    # Save confusion matrix
    plt.tight_layout()
    confusion_matrix_dir = os.path.join(result_data_directory, training_type, 'evaluation_result', d_type, f'{d_type}_{feature_model}_confusion_matrix')
    ensure_directory_exists(confusion_matrix_dir, True)
    confusion_matrix_file = os.path.join(confusion_matrix_dir, f"{d_type}_{feature_model}_{model_name}_{evaluation_type}_confusion_matrix_sample_size_{dataset}.png")
    plt.savefig(confusion_matrix_file)
    plt.close()
    
    print(f"[{d_type}] [{feature_model}] {model_name} {training_type} {evaluation_type} \nConfusion matrix saved to {confusion_matrix_file}")


# %%
def plot_graph(history, d_type, feature_model, model_name, training_type, dataset, graph_type):
    """
    Plots and saves training/validation accuracy and loss graphs.
    
    Parameters:
    - history: Training history containing 'accuracy', 'val_accuracy', 'loss', and 'val_loss'
    - d_type: Dataset type for the plot title and file name
    - feature_model: Feature extraction model used in training
    - model_name: Name of the model being evaluated
    - training_type: Type of training used (e.g., 'fine-tuning', 'transfer')
    - dataset: Dataset used for training
    - graph_type: Type of graph to plot ('accuracy' or 'loss')
    """
    if graph_type == 'accuracy':
        metric = 'accuracy'
        val_metric = 'val_accuracy'
        title = f'[{d_type}]Training and validation accuracy of model: {model_name}, \nFeature extraction: {feature_model}, dataset type: {training_type}'
    elif graph_type == 'loss':
        metric = 'loss'
        val_metric = 'val_loss'
        title = f'[{d_type}]Training and validation loss of model: {model_name}, \nFeature extraction: {feature_model}, dataset type: {training_type}'
    else:
        raise ValueError("graph_type should be either 'accuracy' or 'loss'")
    
    # Extract data from history
    metric_data = history.history[metric]
    val_metric_data = history.history[val_metric]
    
    epochs = range(1, len(metric_data) + 1)

    # Plot the graph
    plt.figure()
    plt.plot(epochs, metric_data, label=f'Training {graph_type}')
    plt.plot(epochs, val_metric_data, label=f'Validation {graph_type}')
    plt.title(title)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    training_plots_dir = os.path.join(result_data_directory, training_type, 'evaluation_result', d_type, f'{d_type}_{feature_model}_training_plots')
    ensure_directory_exists(training_plots_dir, True)
    plot_file = os.path.join(training_plots_dir, f"{d_type}_{feature_model}_{model_name}_{graph_type}_training_plots_sample_size_{dataset}.png")
    plt.savefig(plot_file)
    print(f"[{d_type}] [{feature_model or ''}] {model_name} {training_type} '{graph_type}' training plots saved to {plot_file}")
    plt.close()


# %%
def build_deep_learning_model(input_dim, output_dim):
    # a fully connected (dense) neural network
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet50_model(input_shape, output_dim):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_efficientnetb0_model(input_shape, output_dim):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# %%
def export_evaluation_result(d_type, y_true, y_pred, model_name, feature_model, inference_time_train, inference_time, evaluation_type, params, training_type):
    """
    Export evaluation results including accuracy, classification report, confusion matrix, and more.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=6)
    cm = confusion_matrix(y_true, y_pred)
    
    print_evaluation_summary(d_type, feature_model, training_type, model_name, evaluation_type, acc, inference_time_train, inference_time, report, params)

    save_results_to_file(d_type, feature_model, training_type, model_name, evaluation_type, acc, inference_time_train, inference_time, report, cm, params)

    plot_and_save_confusion_matrix(d_type, feature_model, model_name, evaluation_type, cm, dataset, training_type)
    
    # plot_graph(history, d_type, feature_model, model_name, training_type, dataset, graph_type)


# %%
def predict_and_evaluate(learning_type,model, X_val, y_val, evaluation_type, model_name, feature_model, training_type, d_type, dataset, inference_time_train):
    if learning_type == 'ML':
        start_time = time.time()
        y_val_pred = model.predict(X_val)
        end_time = time.time()
        inference_time = end_time - start_time
        export_evaluation_result(d_type, y_val, y_val_pred, model_name, feature_model, inference_time_train, inference_time, evaluation_type, model_name, training_type)

    elif learning_type == 'DL':
        start_time = time.time()
        y_val_pred = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred, axis=1)
        inference_time = time.time() - start_time
        export_evaluation_result(d_type, y_val, y_val_pred, model_name, feature_model, inference_time_train, inference_time, evaluation_type, model_name, training_type)


# %% [markdown]
# Training loop with validation

# %%
# Count the total number of loops for progress tracking
total_dtypes = len(['CT', 'CXR'])
total_feature_models = len(['mi', 'rf', 'pca'])
total_training_types = len(['fused_data', 'clinical_data_only', 'image_data_only'])
total_classifiers = len([
    'GradientBoosting', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 
    'MLP', 'SVM', 'LogisticRegression', 'KNN'
])
total_cnn_models = len(['ResNet50', 'EfficientNetB0'])



for dtype_idx, d_type in enumerate(['CT', 'CXR'], start=1):
    logging.info(f"Processing data type ({dtype_idx}/{total_dtypes}): {d_type}")

    feature_extraction_models = ['mi', 'rf', 'pca']
    for feature_idx, feature_model in enumerate(['mi', 'rf', 'pca'], start=1):
        logging.info(f"    Feature model ({feature_idx}/{total_feature_models}): {feature_model}")
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

        for train_idx, training_type in enumerate(['clinical_data_only', 'image_data_only','fused_data'], start=1):
            logging.info(f"        Training type ({train_idx}/{total_training_types}): {training_type} ")
                        #  f"(Loops remaining: {loops_remaining})")

   
            # Load data
            X, y = load_data(d_type, feature_model, training_type)
            
            # Step 1: Split into training (80%) and temporary set (20%)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2,random_state=42, stratify=y
            )

            # Step 2: Split the temporary set into testing (10%) and validation (10%)
            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # --> fused_data, clinical_data_only , image_data_only
            ## Train and evaluate classifiers Machine learning models
            for model_idx, (model_name, model) in enumerate(classifiers.items(), start=1):
                logging.info(f"            Training ML model ({model_idx}/{total_classifiers}): {model_name}")
                logging.info(f"                Remaining ML models: {total_classifiers - model_idx}")

                
                model_ml, inference_time_train = build_and_train_model('ML',X_train, y_train, X_val, y_val, model,model_name, feature_model, training_type)
                inference_time_test = evaluate_model('ML',model_ml, X_test, y_test)
                predict_and_evaluate('ML',model_ml, X_val, y_val, "Validation", model_name, feature_model, training_type, d_type, dataset, inference_time_train)
                save_model(d_type, model_name, model, feature_model, training_type)
                print(f"{training_type} training completed and result is saved")

            ## deep learning  using Fully connected modail working on feature extracted data
            if training_type == 'clinical_data_only':
                logging.info(f"        Skipping deep learning for training type: {training_type}")
                continue  ## --> fused_data and image_data_only
            
            logging.info(f"        Training deep learning model: Fully_Connected")
            model_name = 'Fully_Connected'
            model_dl = build_deep_learning_model(input_dim=X_train.shape[1], output_dim=len(np.unique(y)))

            model_dl, inference_time_train, history = build_and_train_model('DL',X_train, y_train, X_val, y_val, model_dl,model_name, feature_model, training_type)
            dl_eval, inference_time_test = evaluate_model('DL',model_dl, X_test, y_test)
            predict_and_evaluate('DL',model_dl, X_val, y_val, "Validation", model_name, feature_model, training_type, d_type, dataset, inference_time_train)
            save_model(d_type, model_name, model_dl, feature_model, training_type, save_type='keras')
            
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)
            # plot figures models
            plot_graph(history, d_type, feature_model, model_name, training_type, dataset, 'accuracy')
            plot_graph(history, d_type, feature_model, model_name, training_type, dataset, 'loss')
            
    ## --> image only  , withou fusing the data
    for training_type in ['image_data_only']:
        # Train CNN Models for Image Data
        # Load data
        X, y = load_data(d_type, None, 'CNN')
        X = np.expand_dims(X, axis=-1) # Add a single channel for grayscale
        X = np.repeat(X, 3, axis=-1)  # Duplicate the single channel to simulate RGB
        # Print the mapping
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)  # Encodes labels as integers
        label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print(f"Label Mapping: {label_mapping}")
        
        # Step 1: Split into training (80%) and temporary set (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2,random_state=42, stratify=y
        )

        # Step 2: Split the temporary set into testing (10%) and validation (10%)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        height, width, channels = 224, 224, 3
        
        cnn_models = {"ResNet50": build_resnet50_model, "EfficientNetB0": build_efficientnetb0_model}
        total_cnn_models = len(cnn_models)
        for cnn_idx, (cnn_model_name, cnn_model_builder) in enumerate(cnn_models.items(), start=1):
            logging.info(f"            Training CNN model ({cnn_idx}/{total_cnn_models}): {cnn_model_name}")
            logging.info(f"                Remaining CNN models: {total_cnn_models - cnn_idx}")
            
            cnn_model = cnn_model_builder(input_shape=(height, width, channels), output_dim=len(np.unique(y_train)))
            model_dl, inference_time_train, history = build_and_train_model('DL',X_train, y_train, X_val, y_val, cnn_model,cnn_model_name, None, training_type)

            dl_eval, inference_time_test = evaluate_model('DL',model_dl, X_test, y_test)
            predict_and_evaluate('DL',model_dl, X_val, y_val, "Validation", cnn_model_name, None, training_type, d_type, dataset, inference_time_train)
            save_model(d_type, cnn_model_name, model_dl, None, training_type, save_type='keras')
                        
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)
            # plot figures models
            plot_graph(history, d_type, None, cnn_model_name, training_type, dataset, 'accuracy')
            plot_graph(history, d_type, None, cnn_model_name, training_type, dataset, 'loss')
            

    logging.info(f"Completed processing for data type: {d_type}")

            
logging.info("[Done] All models evaluated and results saved.")


