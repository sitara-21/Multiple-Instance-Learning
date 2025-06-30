# Multiple-Instance-Learning

## Overview
This project uses Multiple Instance Learning (MIL) concepts on structured (tabular) data containing information regarding fragments for a sample (healthy/tumor). We leverage weakly labelled fragments - each sample's file contains millions of fragments and each fragment contains key information for a sample that enables its classification into healthy vs tumor sample - to classify samples into healthy vs tumor categories. We used attention based aggregation as pooling mechanism for aggregating results from each fragment while classifying a sample. Fragment features such as fragment lengths and 4-mer motifs were used to generate signature likelihoods which were subsequently used as features for the MIL model.

## Assumptions
Negative bags are said to contain only negative instances, while positive bags contain at least one positive instance. 
Positive and negative instances are sampled independently from a positive and a negative distribution.
Hence a healthy sample (negative bag) will contain only healthy instances (fragemnts) while a tumor sample (positive bag) will contain atleast one tumor instance (tumor fragment).

## Training
```main.py``` is the main training script on which further developments have been built: a) main.py -> Training with full dataset; b) main_alt.py -> Train with downsampled (keep 60% of fragments) fragment files; c) main_int.py -> Testing script for interactive SLURM on 02 HPC; d) main_sig.py -> optimized for faster training and inference using signature likelihoods; e) main_sig_cv.py -> performing CV to determine optimal paramters and hyperparameters with optimized script.

## Execution
```sbatch main.sh```
