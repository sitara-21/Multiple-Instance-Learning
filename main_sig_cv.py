import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import itertools
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# import gc
import os

import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyspark.sql.functions import col, regexp_replace, lit, regexp_extract, when, trim

from dataloader import MILDataset, MILDataset_new
from model import Attention, GatedAttention
from model_new import FocalLoss

from pyspark.sql import SparkSession

# os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
# os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ["PATH"]
# Load Java module using subprocess (this ensures persistence)
# subprocess.run("module load gcc java/jdk-1.8u112", shell=True, executable="/bin/bash")

# Get the JAVA_HOME path
# java_home = subprocess.getoutput("readlink -f $(which java) | sed 's:/bin/java::'").strip()

# If JAVA_HOME is empty, manually set it (replace "/opt/java" with your correct path)
# if not java_home:
java_home = "/n/app/java/jdk-1.8u112"

# Set JAVA_HOME and update PATH
os.environ["JAVA_HOME"] = java_home
os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]

# Verify if JAVA_HOME is set
if os.environ["JAVA_HOME"]:
    print(f"JAVA_HOME set to: {java_home}")
else:
    raise EnvironmentError("JAVA_HOME is still not set. Please check the module load command.")

run_no = "0409_1"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ABMIL") \
    .config("spark.driver.memory", "250g") \
    .config("spark.executor.memory", "250g") \
    .config("spark.driver.maxResultSize", "250g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "300s") \
    .getOrCreate()


# bed_file = './samples/MRL0008CHraw_filtered_bedcov_annotated_all_fragments_fix_shifted_chrlab.bed'
bed_files_path_train = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/bed_files_train.txt"
bed_files_path_test = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/bed_files_test.txt"
motif_file1 = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/ref_files/allreads_v7_0_out5p.csv"
motif_file2 = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/ref_files/allreads_v7_0_in5p.csv"
motif_file3 = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/ref_files/MSK_allreads_v7_0_length_LLH.csv"
metadata_file = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/ref_files/info_ctDNA.txt"
# output_folder_train = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/bags_output/train"
# output_folder_test = "/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/bags_output/test"

column_names = ['chromosome', 'frag_start', 'frag_end', 'length_bin', 'out5p', 'in5p', 'gc_content', 'gc_weight', 'is_shifted', 'cnv_weight']

with open(bed_files_path_train, "r") as f:
    bed_files_train = [line.strip() for line in f.readlines()]

with open(bed_files_path_test, "r") as f:
    bed_files_test = [line.strip() for line in f.readlines()]

metadata_df = pd.read_csv(metadata_file, sep="\t")
metadata_df = metadata_df[["V1", "Tissue"]]
metadata_df["Tissue"] = metadata_df["Tissue"].apply(lambda x: 0 if x == "Healthy" else 1)
sample_labels = dict(zip(metadata_df["V1"], metadata_df["Tissue"]))
# print(sample_labels)

def data_prep(motif_file1,
              motif_file2,
              motif_file3,
              column_names,
              bed_file,
             sample_labels):
    motif_df1 = spark.read.option("header", True).csv(motif_file1)
    motif_df1 = motif_df1.withColumn('ID', regexp_extract(col('Sample'), r'filtered_(.*?raw)', 1))
    for col_name in motif_df1.columns:
    #     if col_name != 'ID':
        motif_df1 = motif_df1.withColumnRenamed(col_name, f"{col_name}_out")
    motif_df1 = motif_df1.drop("Sample_out")
    
    motif_df2 = spark.read.option("header", True).csv(motif_file2)
    motif_df2 = motif_df2.withColumn('ID', regexp_extract(col('Sample'), r'filtered_(.*?raw)', 1))
    for col_name in motif_df2.columns:
        # if col_name != 'ID':
        motif_df2 = motif_df2.withColumnRenamed(col_name, f"{col_name}_in")
    motif_df2 = motif_df2.drop("Sample_in")
    
    motif_df3 = spark.read.option("header", True).csv(motif_file3)
    motif_df3 = motif_df3.withColumn('ID', regexp_extract(col('Sample'), r'filtered_(.*?raw)', 1))
    for col_name in motif_df3.columns:
        # if col_name != 'ID':
        motif_df3 = motif_df3.withColumnRenamed(col_name, f"{col_name}_len")
    motif_df3 = motif_df3.drop("Sample_len")
    
    df = spark.read.option("delimiter", "\t").csv(bed_file).sample(fraction=0.4, seed=42)
    for i, col_name in enumerate(column_names[:len(df.columns)]):
            df = df.withColumnRenamed(f"_c{i}", col_name)
    df = df.drop("is_shifted", "cnv_weight")
    df = df.withColumn("length_bin", regexp_replace(col("length_bin"), "-", "_"))
    
    # Extract sample_name from filename (everything before first '_')
    sample_name = bed_file.split("/")[-1].split("_")[0]  # Extract from filename
    df = df.withColumn("sample_name", lit(sample_name))
    sample_name_match = sample_name.replace("raw", "")  # Remove 'raw' suffix

    # Assign correct label from metadata
    sample_label = sample_labels.get(sample_name_match, 1)
    
    joined_df1 = df.join(motif_df1, (df["out5p"] == motif_df1["Motif_out"]) & (df["sample_name"] == motif_df1["ID_out"]), "left")
    joined_df2 = joined_df1.join(motif_df2, (joined_df1["in5p"] == motif_df2["Motif_in"]) & 
                                 (joined_df1["sample_name"] == motif_df2["ID_in"]), "left")
    joined_df3 = joined_df2.join(motif_df3, (joined_df2["length_bin"] == motif_df3["Motif_len"]) & 
                                 (joined_df2["sample_name"] == motif_df3["ID_len"]), "left")
    
    joined_df3 = joined_df3.drop("chromosome", "frag_start", "frag_end", "length_bin", "out5p", "in5p", "Motif_out", 
                                 "ID_out", "Motif_in", "ID_in", "Motif_len", "ID_len", "sample_name")
    
    # Iterate through all columns and clean them
    for col_name in joined_df3.columns:
        # Trim whitespace, replace empty strings, and fill null values
        joined_df3 = joined_df3.withColumn(col_name, 
                    when(trim(col(col_name)) == "", 0)  # Convert empty strings or spaces to 0
                    .when(col(col_name).isNull(), 0)    # Convert nulls to 0
                    .otherwise(col(col_name)))          # Keep existing values
    
    return joined_df3, sample_label

def create_dataloader(bed_files, batch_size=1, shuffle=True):
    all_bags = []
    
    for bed_file in bed_files:
        print(f"Processing: {bed_file}")
        
        # Preprocess data and get label
        processed_df, sample_label = data_prep(motif_file1, motif_file2, motif_file3, column_names, bed_file, sample_labels)
        # print(processed_df.columns)
        # processed_df.cache()

        # Convert Spark DF to Pandas then to NumPy directly
        processed_array = processed_df.toPandas().astype(np.float16).values

        # Create MIL dataset
        dataset = MILDataset_new(processed_array, sample_label)

        # Extract the full bag once and store it
        bag_features, bag_label = dataset.get_bag()
        # Extract sample name correctly
        # sample_name = os.path.basename(bed_file).split("_")[0]  # Get everything before the first '_'

        # Save bag and label
        # torch.save({"features": bag_features, "label": bag_label}, os.path.join(output_folder, f"{sample_name}.pth"))
        
        all_bags.append((bag_features, bag_label))

    # Create DataLoader
    dataloader = DataLoader(all_bags, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# def plot_training_metrics(train_losses, train_aucs, save_path="/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/training_metrics_focal_llh_new.png"):
#     """
#     Plots Training Loss vs. Epochs and AUC vs. Epochs in the same plot.
#     """
#     epochs = range(1, len(train_losses) + 1)

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot Loss vs. Epochs
#     ax.plot(epochs, train_losses, 'b-', label='Training Loss')
#     ax.plot(epochs, train_aucs, 'r-', label='Training AUC')
#     ax.set_xlabel('Epochs')
#     plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
#     ax.legend(loc='best')

#     plt.title('Training Loss and AUC vs. Epochs')

#     # Save instead of showing
#     plt.savefig(save_path, dpi=300)
#     plt.close()

def train_model(model, dataloader, epochs=20, lr=0.001, plot_results=True):
    """
    Train MIL model using Focal Loss and plot training loss, AUC, and ROC curve.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    train_losses = []
    train_aucs = []
    all_labels = []
    all_probs = []

    for epoch in range(epochs):
        total_loss = 0.0
        true_labels = []
        predicted_probs = []

        for bag_features, bag_label in dataloader:
            optimizer.zero_grad()
            bag_features = bag_features.squeeze(0)
            # bag_features = bag_features.to(dtype=torch.float16)
            if bag_features.shape[0] == 0:
                raise ValueError("All instances were lost after squeeze!")

            bag_label = bag_label.float().unsqueeze(0).view(-1)
            bag_prediction = model(bag_features).view(-1)

            loss = criterion(bag_prediction, bag_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            true_labels.append(bag_label.item())
            predicted_probs.append(bag_prediction.item())

        auc_score = roc_auc_score(true_labels, predicted_probs)
        train_losses.append(total_loss)
        train_aucs.append(auc_score)
        all_labels.extend(true_labels)
        all_probs.extend(predicted_probs)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, AUC: {auc_score:.4f}")

    print("Training completed.")

    # Plot training loss and AUC and ROC curve
    if plot_results:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, epochs + 1)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.plot(epochs_range, train_losses, 'b-', label='Loss')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('AUC', color='red')
        ax2.plot(epochs_range, train_aucs, 'r-', label='AUC')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Training Loss and AUC vs Epochs')
        fig.tight_layout()
        plt.savefig(f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/training_metrics_combined{run_no}.png", dpi=300)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {train_aucs[-1]:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Train ROC Curve')
        plt.legend()
        plt.savefig(f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/train_roc_curve_run{run_no}.png", dpi=300)
        
        plt.close()


    return model, train_losses, train_aucs

def render_confusion_matrix(cm, class_names, normalize=False, cm_save_path=f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/test_CM_run{run_no}.png"):
    """Render confusion matrix as a matplotlib figure."""
    title = "Confusion matrix"
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(8, 8))
    vmax = 1 if normalize else None
    plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=300)
    plt.close()

def evaluate_model_with_attention(model, dataloader, roc_save_path=f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/test_roc_curve_run{run_no}.png"):
    """
    Evaluate MIL model using AUC, Confusion Matrix, and extract Attention Scores.
    """
    model.eval()
    true_labels = []
    predicted_probs = []
    predicted_labels = []
    all_attention_scores = []

    with torch.no_grad():
        for bag_features, bag_label in dataloader:
            bag_features, bag_label = bag_features.squeeze(0), bag_label.float()
            
            # Forward pass with attention extraction
            bag_prediction, _, attention_scores = model(bag_features, return_attention=True)

            # Convert probability to binary label (threshold=0.5)
            predicted_label = (bag_prediction >= 0.5).float()

            # Store predictions and true labels
            predicted_probs.append(bag_prediction.item())
            predicted_labels.append(predicted_label.item())
            true_labels.append(bag_label.item())

            # Store attention scores for each fragment in the bag
            all_attention_scores.append(attention_scores.squeeze(-1).cpu().numpy())  # Shape: (num_instances,)

    # Compute AUC
    auc_score = roc_auc_score(true_labels, predicted_probs)

    # Compute Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    class_names = ['Healthy', 'Cancer']
    render_confusion_matrix(cm, class_names, normalize=False)

    print(f"Test AUC: {auc_score:.4f}")
    # print("Confusion Matrix:")
    # print(cm)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_save_path, dpi=300)
    plt.close()

    return auc_score, cm, all_attention_scores

# def plot_attention_scores(attention_scores, sample_idx=0, save_path="/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/attention_scores_sample.png"):
#     """
#     Plots attention scores for a single sample in the test set.

#     Args:
#         attention_scores (list of numpy arrays): Attention scores for all test samples.
#         sample_idx (int): Index of the sample to visualize.
#     """
#     if sample_idx >= len(attention_scores):
#         print("Invalid sample index!")
#         return
    
#     scores = attention_scores[sample_idx]  # Get scores for the chosen bag
    
#     # Ensure it's a 1D array
#     scores = np.array(scores).squeeze()  # Removes extra dimensions if any

#     if scores.ndim != 1:
#         print(f"Unexpected shape: {scores.shape}, flattening...")
#         scores = scores.flatten()

#     plt.figure(figsize=(10, 4))
#     plt.bar(range(len(scores)), scores, color="blue")
#     plt.xlabel("Fragment Index")
#     plt.ylabel("Attention Score")
#     plt.title(f"Attention Scores for Sample {sample_idx}")
#     # Save plot instead of showing
#     plt.savefig(save_path, dpi=300)
#     plt.close()

def plot_attention_distribution(attention_scores, true_labels, save_path=f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/attention_distribution_run{run_no}.png"):
    """
    Plots a histogram of attention scores across all test samples, colored by their bag labels.

    Args:
        attention_scores (list of numpy arrays): Attention scores for all test samples.
        true_labels (list): Corresponding labels (0=Healthy, 1=Tumor) for each test bag.
    """
    healthy_scores = []
    tumor_scores = []

    # Separate scores by label
    for scores, label in zip(attention_scores, true_labels):
        if label == 0:
            healthy_scores.extend(scores.flatten())  # Flatten to 1D and store
        else:
            tumor_scores.extend(scores.flatten())  # Flatten to 1D and store

    # Plot histogram with different colors for healthy vs. tumor
    plt.figure(figsize=(8, 5))
    plt.hist(healthy_scores, bins=50, color="blue", alpha=0.7, label="Healthy")
    plt.hist(tumor_scores, bins=50, color="red", alpha=0.7, label="Tumor")
    plt.xscale('log')
    plt.xlabel("Attention Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attention Scores (Colored by Labels)")
    plt.legend()
    # Save plot instead of showing
    plt.savefig(save_path, dpi=300)
    plt.close()

# Load training data
dataloader_train = create_dataloader(bed_files_train, batch_size=1, shuffle=True)
dataloader_test = create_dataloader(bed_files_test, batch_size=1, shuffle=False)

# Get correct input dimension
sample_bag, _ = next(iter(dataloader_train))
# print(f"Sample Bag Shape (Before Squeeze): {sample_bag.shape}")  # Debugging

input_dim = sample_bag.squeeze(0).shape[-1]  # Extract feature size
hidden_dim = 128  

# Initialize model
model = Attention(input_dim, hidden_dim).to(torch.device("cpu"))  

print("\nStarting Training...")

# Train model
trained_model, train_losses, train_aucs = train_model(model, dataloader_train, epochs=20, lr=0.001)

# Save the trained model
save_path = f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/models/trained_model_focal_llh_run{run_no}.pth"
torch.save(trained_model.state_dict(), save_path)
print(f"Model saved at {save_path}")

print("\nRunning 10-fold cross-validation on preloaded training bags...")

# Extract loaded data from dataloader_train
full_data = dataloader_train.dataset  # List of (bag_tensor, label_tensor)
cv_fold_aucs = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_data)):
    print(f"\n[CV] Fold {fold + 1}/10")

    fold_train_data = [full_data[i] for i in train_idx]
    fold_val_data = [full_data[i] for i in val_idx]

    fold_train_loader = DataLoader(fold_train_data, batch_size=1, shuffle=True)
    fold_val_loader = DataLoader(fold_val_data, batch_size=1, shuffle=False)

    # Reinitialize model
    fold_model = Attention(input_dim, hidden_dim).to(torch.device("cpu"))

    # Train on fold
    trained_model_cv, _, _ = train_model(fold_model, fold_train_loader, epochs=10, lr=0.001, plot_results=False)

    # Evaluate on validation fold
    val_auc, _, _ = evaluate_model_with_attention(
        trained_model_cv, fold_val_loader, roc_save_path=f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/cv_fold_{fold+1}_roc_run{run_no}.png"
    )
    cv_fold_aucs.append(val_auc)

# Summary
print("\n Cross-validation completed.")
print(f"AUCs across 10 folds: {cv_fold_aucs}")
print(f"Mean AUC: {np.mean(cv_fold_aucs):.4f}, Std: {np.std(cv_fold_aucs):.4f}")


print("\nEvaluating on Test Set...")
# Evaluate on test data
test_auc, test_cm, test_attention_scores = evaluate_model_with_attention(trained_model, dataloader_test)

# Visualize attention scores for the first test sample
# plot_attention_scores(test_attention_scores, sample_idx=0, save_path="/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/attention_scores_sample_0_focal_llh_new.png")

# Extract true labels from the test dataloader
true_test_labels = []

for _, bag_label in dataloader_test:
    true_test_labels.append(bag_label.item())  # Convert tensor to scalar

# Now plot the histogram
plot_attention_distribution(test_attention_scores, true_test_labels, save_path=f"/n/data1/hms/dbmi/gulhan/lab/ankit/scripts/MIL/plots/attention_distribution_focal_llh_run{run_no}.png")

spark.stop()
