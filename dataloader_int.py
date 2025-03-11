import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from pyspark.sql.functions import col, input_file_name, split, regexp_replace, lit, sum, regexp_extract, when, trim

class MILDataset(Dataset):
    def __init__(self, spark_df, sample_label):
        """
        Args:
            spark_df (PySpark DataFrame): The PySpark DataFrame containing instance features.
            sample_label (int): The label for the entire bag (0 = healthy, 1 = tumor).
        """
        self.spark_df = spark_df
        self.sample_label = sample_label
        # self.column_names = spark_df.columns

        # Convert all columns to float type (efficiently in PySpark)
        for col_name in self.spark_df.columns:
            self.spark_df = self.spark_df.withColumn(col_name, self.spark_df[col_name].cast("float"))

        # Cache the DataFrame to avoid recomputation
        self.spark_df.cache()

        # Use PySpark's lazy iterator instead of .rdd.map()
        self.instances_iter = self.spark_df.toLocalIterator()

    def __len__(self):
        """Returns the number of instances in the bag (sample)."""
        return self.spark_df.count()  # Get row count efficiently in Spark

    def __getitem__(self, idx):
        """Retrieves one instance (fragment) as a tensor."""
        row = next(self.instances_iter)  # Fetch next row using the iterator
        feature_vector = torch.tensor([float(row[col]) for col in self.spark_df.columns], dtype=torch.float32)
        return feature_vector

    def get_bag(self):
        """Returns the entire bag (set of feature vectors) along with its label."""
        feature_list = []

        # Stream all rows one by one (avoids `.collect()` OOM issue)
        for row in self.spark_df.toLocalIterator():
            feature_vector = torch.tensor(list(row), dtype=torch.float32)
            feature_list.append(feature_vector)

        # Stack into a single tensor
        bag_tensor = torch.stack(feature_list)
        label_tensor = torch.tensor(self.sample_label, dtype=torch.float32)  # Single label for the entire bag
        return bag_tensor, label_tensor