# python early_fusion_slide_level_risk_scores.py --mode "only_wsi" --use_model "gbst" --do_hpo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# from lifelines.plotting import plot_schoenfeld_residuals
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import random
import optuna
import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_regression
import torchtuples as tt
from pycox.models import CoxPH

# from pycox.evaluation import concordance_index
import argparse
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pdb import set_trace

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
use_system = "cluster"  # cluster or laptop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model training and evaluation parameters"
    )

    parser.add_argument(
        "--check_PH_assumptions",
        action="store_true",
        help="Check proportional hazards assumptions",
    )
    parser.add_argument("--plot_embs", action="store_true", help="Plot embeddings")
    parser.add_argument(
        "--calc_information_content",
        action="store_true",
        help="Calculate metrics related to information content in the different modalities",
    )
    parser.add_argument(
        "--plot_survival", action="store_true", help="Plot survival curves"
    )
    parser.add_argument(
        "--drop_outliers",
        action="store_true",
        help="Drop very high values of survival duration in the data",
    )
    parser.add_argument(
        "--do_bootstrap",
        action="store_true",
        default=True,
        help="Perform bootstrap resampling on test data",
    )
    parser.add_argument(
        "--do_hpo",
        action="store_true",
        help="Perform hyperparameter optimization (mention --do_hpo for HPO)",
    )
    parser.add_argument(
        "--apply_pca", action="store_true", help="Apply PCA dimensionality reduction"
    )
    parser.add_argument(
        "--use_model",
        type=str,
        default="gbst",
        choices=["snn", "gbst"],
        help="Model type: gradient boosted survival tree (gbst) or survival neural network (snn)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rnaseq_wsi",
        choices=["rnaseq_wsi", "only_rnaseq", "only_wsi"],
        help="Data modality to use",
    )

    return parser.parse_args()


args = parse_args()

# on laptop
if use_system == "laptop":
    input_dir = "/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/"

if use_system == "cluster":
    # on polaris
    input_dir = "/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/early_fusion_inputs/"
    id = "2025-02-27-01-41-45_fold_1_epoch_3000"  # rnaseq and wsi embeddings distributions more similar

seed_value = 422
random.seed(seed_value)
np.random.seed(seed_value)

# file paths for rnaseq embeddings (generated using generate_rnaseq_embeddings_kfoldCV.py)
train_file = os.path.join(input_dir, f"rnaseq_embeddings_train_checkpoint_{id}.json")
val_file = os.path.join(input_dir, f"rnaseq_embeddings_val_checkpoint_{id}.json")
test_file = os.path.join(input_dir, f"rnaseq_embeddings_test_checkpoint_{id}.json")

train_file = os.path.join(
    input_dir, f"rnaseq_embeddings_train_checkpoint_hp_subset_{id}.json"
)
val_file = os.path.join(
    input_dir, f"rnaseq_embeddings_val_checkpoint_hp_subset_{id}.json"
)
test_file = os.path.join(
    input_dir, f"rnaseq_embeddings_test_checkpoint_hp_subset_{id}.json"
)


# load the rnaseq embeddings
train_embs = pd.read_json(train_file).T
val_embs = pd.read_json(val_file).T
test_embs = pd.read_json(test_file).T

print("Train embeddings shape (RNASeq):", train_embs.shape)

# print("Test embeddings shape (RNASeq):", test_embs.shape)

# combine val and test sets into a single held out set (the validation dataset wasn't used in training; kfoldCV was used with the training dataset only)
print("concatenating val and test sets to create a single held out dataset")
test_validation_embs = pd.concat(
    [val_embs, test_embs], axis=0
)  # concatenate along rows (samples)
test_validation_tcga_ids = list(val_embs.index) + list(test_embs.index)
print("Test + Validation embeddings shape (RNASeq):", test_validation_embs.shape)

print("Loading WSI data")
# set_trace()
# wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_average_uni_29Jan.rounded.json')).T
# wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_average_uni_30Jan_1000tiles.rounded.json')).T

# load wsi embedding file once and save as parquet file. Load the parquet file for subsequent inference runs
# wsi_embs = pd.read_json(os.path.join(input_dir, 'WSI_embeddings_uni_31Jan_1000tiles.truncated.json'), lines=True).T
# wsi_embs.to_parquet(os.path.join(input_dir, 'WSI_embeddings_uni_31Jan_1000tiles.truncated.json.parquet'))
# set_trace()

wsi_embs = pd.read_parquet(
    os.path.join(input_dir, "WSI_embeddings_uni_31Jan_1000tiles.truncated.json.parquet")
)

# check shapes
# embedding_shapes = wsi_embs.iloc[:, 0].apply(lambda x: np.array(x).shape)
# unique_shapes = embedding_shapes.unique()
# get the slide level embeddings by averaging the tile level embeddings
print("obtaining slide level embeddings by averaging the tile level embeddings")


wsi_embs["slide_embedding"] = wsi_embs.iloc[:, 0].apply(lambda x: np.mean(x, axis=0))
wsi_embs = wsi_embs.drop(columns=[0])

# remove the problematic WSIs (with penmarks etc)
exclude_ids_wsi = [
    "TCGA-86-6851",
    "TCGA-86-7701",
    "TCGA-86-7711",
    "TCGA-86-7713",
    "TCGA-86-7714",
    "TCGA-86-7953",
    "TCGA-86-7954",
    "TCGA-86-7955",
    "TCGA-86-8055",
    "TCGA-86-8056",
    "TCGA-86-8073",
    "TCGA-86-8074",
    "TCGA-86-8075",
    "TCGA-86-8076",
    "TCGA-86-8278",
    "TCGA-86-8279",
    "TCGA-86-8280",
    "TCGA-86-A4P7",
    "TCGA-86-A4P8",
]
wsi_embs = wsi_embs.drop(index=exclude_ids_wsi, errors="ignore")

# find common indices between the rnaseq and WSI embeddings
common_train_indices = (
    train_embs.index.intersection(wsi_embs.index)
    .to_series()
    .sample(frac=1, random_state=seed_value)
    .index
)
common_test_validation_indices = (
    test_validation_embs.index.intersection(wsi_embs.index)
    .to_series()
    .sample(frac=1, random_state=seed_value)
    .index
)

# select only common indices for training and test/validation sets
train_embs = train_embs.loc[common_train_indices]
test_validation_embs = test_validation_embs.loc[common_test_validation_indices]

# using wsi_embs for both as there was no training done
train_wsi_embs = wsi_embs.loc[common_train_indices]
test_validation_wsi_embs = wsi_embs.loc[common_test_validation_indices]

print("Train embeddings shape (WSI):", train_wsi_embs.shape)
print("Test + Validation embeddings shape (WSI):", test_validation_wsi_embs.shape)
# print("Test embeddings shape (WSI):", test_embs.shape)


# a scaler that does nothing
class IdentityScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X


###### normalize rnaseq embeddings based on training set
# use min/max across all the embeddings
# for fit_transform() input needs to be of shape n_samples x n_features
print("scaling rnaseq embeddings")
# scaler_rna = MinMaxScaler(feature_range=(-1, 1))
# scaler = RobustScaler()
scaler_rna = StandardScaler()
# scaler_rna = IdentityScaler()
train_embs_normalized = pd.DataFrame(
    scaler_rna.fit_transform(train_embs), index=train_embs.index
)  # fit only using the training data
test_validation_embs_normalized = pd.DataFrame(
    scaler_rna.transform(test_validation_embs), index=test_validation_embs.index
)

# normalize WSI embeddings using training data
print("scaling WSI embeddings")
# scaler_wsi = MinMaxScaler(feature_range=(-1, 1))
# scaler = RobustScaler()
scaler_wsi = StandardScaler()
# scaler_wsi = IdentityScaler()

# ensure non-empty arrays before stacking
if not train_wsi_embs.empty:
    train_wsi_embs_array = np.stack(train_wsi_embs["slide_embedding"].values)
    train_wsi_embs_normalized = pd.DataFrame(
        scaler_wsi.fit_transform(train_wsi_embs_array), index=train_wsi_embs.index
    )
else:
    train_wsi_embs_normalized = pd.DataFrame()

if not test_validation_wsi_embs.empty:
    test_validation_wsi_embs_array = np.stack(
        test_validation_wsi_embs["slide_embedding"].values
    )
    test_validation_wsi_embs_normalized = pd.DataFrame(
        scaler_wsi.transform(test_validation_wsi_embs_array),
        index=test_validation_wsi_embs.index,
    )
else:
    test_validation_wsi_embs_normalized = pd.DataFrame()

# combine normalized unimodal embeddings to obtain multimodal embeddings
print("concatenating embeddings")
if args.mode == "rnaseq_wsi":
    # X_train = pd.concat([train_embs_normalized, train_wsi_embs_normalized], axis=1)
    # X_test_validation = pd.concat([test_validation_embs_normalized, test_validation_wsi_embs_normalized], axis=1)

    rna_weight = 0.8
    wsi_weight = 0.2
    # apply weighted addition
    X_train = (train_embs_normalized * rna_weight) + (
        train_wsi_embs_normalized * wsi_weight
    )
    X_test_validation = (test_validation_embs_normalized * rna_weight) + (
        test_validation_wsi_embs_normalized * wsi_weight
    )

# set_trace()
elif args.mode == "only_wsi":
    X_train = train_wsi_embs_normalized
    X_test_validation = test_validation_wsi_embs_normalized
if args.mode == "only_rnaseq":
    X_train = train_embs_normalized
    X_test_validation = test_validation_embs_normalized

if args.plot_embs:

    def plot_embedding_statistics(rna_embs, wsi_embs, split):
        rna_min = rna_embs.min(axis=1)
        rna_max = rna_embs.max(axis=1)
        rna_mean = rna_embs.mean(axis=1)
        rna_std = rna_embs.std(axis=1)

        wsi_min = wsi_embs.min(axis=1)
        wsi_max = wsi_embs.max(axis=1)
        wsi_mean = wsi_embs.mean(axis=1)
        wsi_std = wsi_embs.std(axis=1)

        plt.figure(figsize=(12, 6))

        plt.plot(rna_min, label="RNA-seq Min", color="blue", linestyle="-")
        plt.plot(rna_max, label="RNA-seq Max", color="blue", linestyle="--")
        plt.plot(rna_mean, label="RNA-seq Mean", color="blue", linestyle="-.")
        plt.plot(rna_std, label="RNA-seq Std", color="blue", linestyle=":")

        plt.plot(wsi_min, label="WSI Min", color="red", linestyle="-")
        plt.plot(wsi_max, label="WSI Max", color="red", linestyle="--")
        plt.plot(wsi_mean, label="WSI Mean", color="red", linestyle="-.")
        plt.plot(wsi_std, label="WSI Std", color="red", linestyle=":")

        plt.xlabel("TCGA Sample Index")
        plt.ylabel("Embedding Value")
        plt.title("Statistics of RNA-seq and WSI Embeddings Across TCGA Samples")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"embs_stats_{split}_{timestamp}.png")
        plt.show()

    def plot_embedding_distributions(rna_embs, wsi_embs, split):
        plt.figure(figsize=(12, 6))
        plt.hist(
            rna_embs.values.flatten(),
            bins=100,
            alpha=0.5,
            label="RNA-Seq Embeddings",
            color="blue",
            density=True,
        )
        plt.hist(
            wsi_embs.values.flatten(),
            bins=100,
            alpha=0.5,
            label="WSI Embeddings",
            color="red",
            density=True,
        )

        plt.xlabel("Embedding Value")
        plt.ylabel("Density")
        plt.title("Distribution of RNA-Seq and WSI Embeddings (Train Dataset)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"embs_dist_{split}_{timestamp}.png")
        plt.show()

    # plot_embedding_statistics(test_validation_embs_array, test_validation_wsi_embs_array)
    plot_embedding_statistics(
        train_embs,
        pd.DataFrame(train_wsi_embs_array, index=train_wsi_embs.index),
        "train",
    )
    # set_trace()
    plot_embedding_distributions(
        train_embs,
        pd.DataFrame(train_wsi_embs_array, index=train_wsi_embs.index),
        "train",
    )
    # plot_embedding_distributions(test_validation_embs, pd.DataFrame(test_validation_wsi_embs_array, index=test_validation_embs.index))
    # plot_embedding_distributions(train_embs_normalized, train_wsi_embs_normalized)

    set_trace()

# load survival data
print("loading survival data")
mapping_df = pd.read_json(os.path.join(input_dir, "mapping_df_31Jan_1000tiles.json")).T
# convert the 'event_occurred' values to 0 and 1 for compatibility with Sksurv
# mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': 1, 'Alive': 0}).astype(int)
mapping_df["event_occurred"] = (
    mapping_df["event_occurred"].map({"Dead": 1, "Alive": 0}).astype(bool)
)
# prepare train and test survival data
train_survival = mapping_df.loc[common_train_indices]
test_validation_survival = mapping_df.loc[common_test_validation_indices]
set_trace()

train_survival["event_occurred"] = train_survival["event_occurred"].astype(bool)
test_validation_survival["event_occurred"] = test_validation_survival[
    "event_occurred"
].astype(bool)

train_surv = Surv.from_arrays(
    train_survival["event_occurred"].values, train_survival["time"].values
)
test_validation_surv = Surv.from_arrays(
    test_validation_survival["event_occurred"].values,
    test_validation_survival["time"].values,
)

if args.plot_survival:
    plt.figure(figsize=(8, 6))

    # Kaplan-Meier Fitter for train data
    kmf_train = KaplanMeierFitter()
    kmf_train.fit(
        train_survival["time"], train_survival["event_occurred"], label="Train Survival"
    )
    kmf_train.plot_survival_function(color="blue")

    # Kaplan-Meier Fitter for test data
    kmf_test = KaplanMeierFitter()
    kmf_test.fit(
        test_validation_survival["time"],
        test_validation_survival["event_occurred"],
        label="Test Survival",
    )
    kmf_test.plot_survival_function(color="red")

    plt.title("Survival Analysis: Train vs. Test")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)

    plt.savefig("survival_analysis.png", dpi=300, bbox_inches="tight")

    plt.show()

if args.check_PH_assumptions:

    def check_ph_assumption(data, dataset_name):
        """Check proportional hazards assumption using median survival time for given dataset."""
        print(f"\nChecking Proportional Hazards Assumption for {dataset_name}...\n")
        # checking KM plots

        median_time = np.median(data["time"])
        high_risk = data["time"] <= median_time
        low_risk = data["time"] > median_time

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(
            data["time"][high_risk],
            data["event_occurred"][high_risk],
            label="High risk",
        )
        kmf_low.fit(
            data["time"][low_risk], data["event_occurred"][low_risk], label="Low risk"
        )

        plt.figure()
        # kmf_high.plot()
        # kmf_low.plot()
        kmf_high.plot_survival_function(color="blue")
        kmf_low.plot_survival_function(color="red")
        plt.title(f"{dataset_name} stratified by risk estimates")
        plt.xlim(left=0)
        plt.ylim(top=1)
        plt.xlabel("Time (days)")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        # plt.savefig(f"{mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

    check_ph_assumption(train_survival, "Training Data")
    check_ph_assumption(test_validation_survival, "Test Data")


# HPO with optuna
print(f" ********  Training for mode: {args.mode}  **********")


if args.use_model == "gbst":
    if args.do_hpo:

        def objective(trial):

            # # n_estimators = trial.suggest_int('n_estimators', 10, 100)
            n_estimators = trial.suggest_int("n_estimators", 10, 100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 3)  # 1,5

            model = GradientBoostingSurvivalAnalysis(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                #  max_features='sqrt', # remove it?
                max_features=0.5,
                #  max_features=max_features,
                random_state=seed_value,
            )
            print(
                f"X_train shape: {X_train.shape}, train_surv shape: {train_surv.shape}"
            )
            kf = KFold(
                n_splits=5, shuffle=True, random_state=seed_value
            )  # 5-fold CV : 45
            c_indices = []

            for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_index],
                    X_train.iloc[val_index],
                )
                survival_train_fold = train_surv[train_index]
                survival_val_fold = train_surv[val_index]
                # set_trace()
                X_train_fold_np = X_train_fold.to_numpy()
                X_val_fold_np = X_val_fold.to_numpy()

                model.fit(X_train_fold_np, survival_train_fold)
                risk_scores_val = model.predict(X_val_fold_np)

                # Compute concordance index on validation fold
                c_index_val = concordance_index_censored(
                    survival_val_fold["event"],  # name changed internally by sksurv
                    survival_val_fold["time"],
                    risk_scores_val,
                )[0]
                c_indices.append(c_index_val)

            return np.mean(c_indices)  # average CI across folds

        num_cpu_cores = multiprocessing.cpu_count()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, n_jobs=num_cpu_cores)
        best_params = study.best_params
        print("Best parameters found: ", study.best_params)
        model = GradientBoostingSurvivalAnalysis(**best_params)
        # train the model
        model.fit(X_train, train_surv)
    else:
        print("training using model configuration determined from HPO study")
        if args.mode == "rnaseq_wsi":
            # model = GradientBoostingSurvivalAnalysis(n_estimators=97, # 22
            #                                         learning_rate=0.0575,
            #                                         max_depth=3,
            #                                         max_features=0.5,
            #                                         random_state=seed_value)
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=98,  # 22
                learning_rate=0.08,
                max_depth=3,
                max_features=0.5,
                random_state=seed_value,
            )
        elif args.mode == "only_rnaseq":
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=97,
                learning_rate=0.083,
                max_depth=1,
                max_features=0.5,
                random_state=seed_value,
            )

        elif args.mode == "only_wsi":
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=71,
                learning_rate=0.03618,
                max_depth=3,
                max_features=0.5,
                random_state=seed_value,
            )

        model.fit(X_train, train_surv)

elif args.use_model == "snn":
    # define the survival neural network
    def create_snn(input_dim, num_nodes=[32, 32], dropout=0.1):
        model = tt.practical.MLPVanilla(
            in_features=input_dim,
            num_nodes=num_nodes,
            out_features=1,
            batch_norm=True,
            dropout=dropout,
        )

        model = model.float()
        net = CoxPH(model, tt.optim.Adam(lr=1e-3))
        return net

    if args.do_hpo:

        def objective(trial):
            # Hyperparameters to optimize
            num_nodes_1 = trial.suggest_categorical("num_nodes_1", [8, 16, 32])
            num_nodes_2 = trial.suggest_categorical("num_nodes_2", [8, 16, 32])
            dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
            batch_size = trial.suggest_categorical("batch_size", [64, 128])
            epochs = trial.suggest_categorical("epochs", [150])

            print(f"\n{'='*20} Trial {trial.number} {'='*20}")
            print(
                f"\nStarting trial {trial.number}: num_nodes=({num_nodes_1}, {num_nodes_2}), dropout={dropout}, batch_size={batch_size}, epochs={epochs}"
            )

            model = create_snn(
                input_dim=X_train.shape[1],
                num_nodes=[num_nodes_1, num_nodes_2],
                dropout=dropout,
            )

            print(
                f"X_train shape: {X_train.shape}, train_surv shape: {train_surv.shape}"
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)  # 5-fold CV
            c_indices = []

            for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
                print(f"\n--- Trial {trial.number}, Fold {fold}/5 ---")
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_index],
                    X_train.iloc[val_index],
                )
                survival_train_fold = train_surv[train_index]
                survival_val_fold = train_surv[val_index]

                # copy arrays to ensure contiguous memory layout and consistent data type
                X_train_fold_np = np.array(
                    X_train_fold.to_numpy(), copy=True, dtype=np.float32
                )
                X_val_fold_np = np.array(
                    X_val_fold.to_numpy(), copy=True, dtype=np.float32
                )

                # format data for torch training; ensure arrays are contiguous and of consistent data type
                y_train_fold = (
                    np.array(survival_train_fold["time"], copy=True, dtype=np.float32),
                    np.array(survival_train_fold["event"], copy=True, dtype=np.float32),
                )

                model.fit(
                    X_train_fold_np,
                    y_train_fold,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=False,
                )

                risk_scores_val = model.predict(X_val_fold_np).flatten()
                c_index_val = concordance_index_censored(
                    survival_val_fold["event"],
                    survival_val_fold["time"],
                    risk_scores_val,
                )[0]
                print(f"Fold {fold} completed. C-index: {c_index_val:.4f}")
                c_indices.append(c_index_val)

            avg_c_index = np.mean(c_indices)
            print(
                f"\n*** Trial {trial.number} completed. Average C-index: {avg_c_index:.4f} ***"
            )
            print(f"{'='*55}\n")

            return avg_c_index

            # return np.mean(c_indices)  # average CI across folds

        num_cpu_cores = multiprocessing.cpu_count()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, n_jobs=num_cpu_cores)
        best_params = study.best_params

        # train the model with best parameters on full training data
        model = create_snn(
            input_dim=X_train.shape[1],
            num_nodes=[best_params["num_nodes_1"], best_params["num_nodes_2"]],
            dropout=best_params["dropout"],
        )

        # format full training data; ensure arrays are contiguous and of consistent data type
        y_train = (
            np.array(train_surv["time"], copy=True, dtype=np.float32),
            np.array(train_surv["event"], copy=True, dtype=np.float32),
        )
        X_train_np = np.array(X_train.to_numpy(), copy=True, dtype=np.float32)

        model.fit(
            X_train_np,
            y_train,
            batch_size=best_params["batch_size"],
            epochs=best_params["epochs"],
            verbose=True,
        )
    else:
        print("training using model configuration determined from HPO")
        model = create_snn(input_dim=X_train.shape[1], num_nodes=[64, 32], dropout=0.2)
        batch_size = 32
        epochs = 150

        y_train = (
            np.array(train_surv["time"], copy=True),
            np.array(train_surv["event"], copy=True),
        )
        X_train_np = np.array(X_train.to_numpy(), copy=True)

        model.fit(
            X_train_np, y_train, batch_size=batch_size, epochs=epochs, verbose=True
        )


# set_trace()
# # train the model
# model.fit(X_train, train_surv)

# predict risk scores
if args.use_model == "snn":
    X_train = np.array(X_train.to_numpy(), copy=True, dtype=np.float32)
    X_test_validation = np.array(
        X_test_validation.to_numpy(), copy=True, dtype=np.float32
    )

risk_scores_train = model.predict(X_train)
risk_scores_test = model.predict(X_test_validation)

# compute concordance indices
c_index_train = concordance_index_censored(
    train_survival["event_occurred"],
    train_survival["time"],
    risk_scores_train.flatten(),
)[0]
c_index_test = concordance_index_censored(
    test_validation_survival["event_occurred"],
    test_validation_survival["time"],
    risk_scores_test.flatten(),
)[0]

print(f"Train Concordance Index: {c_index_train:.4f}")
print(f"Test Concordance Index: {c_index_test:.4f}")


# Kaplan-Meier Analysis
median_risk = np.median(risk_scores_test)
high_risk = risk_scores_test >= median_risk
low_risk = risk_scores_test < median_risk

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
# set_trace()
kmf_high.fit(
    test_validation_survival["time"][high_risk.flatten()],
    test_validation_survival["event_occurred"][high_risk.flatten()],
    label="High Risk",
)
kmf_low.fit(
    test_validation_survival["time"][low_risk.flatten()],
    test_validation_survival["event_occurred"][low_risk.flatten()],
    label="Low Risk",
)

# compute log-rank p-value
logrank_p_value = logrank_test(
    test_validation_survival["time"][high_risk.flatten()],
    test_validation_survival["time"][low_risk.flatten()],
    test_validation_survival["event_occurred"][high_risk.flatten()],
    test_validation_survival["event_occurred"][low_risk.flatten()],
).p_value

print(f"log-rank test p-value: {logrank_p_value:.5f}")
# # set_trace()
# plt.figure()
# kmf_high.plot_survival_function(color='blue', ci_show=False)
# kmf_low.plot_survival_function(color='red', ci_show=False)
# plt.title(f"CI: {c_index_test:.3f},\nlog-rank test p-value: {logrank_p_value:.5f}")
# plt.xlim(left=0)
# plt.ylim(top=1, bottom=0)
# plt.xlabel("Time (days)")
# plt.ylabel("Survival Probability")
# plt.grid(True)
# plt.savefig(f"{args.mode}_{timestamp}.png", dpi=300, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(8, 6))  # Set figure size for better resolution
font_size = 16
kmf_high.plot_survival_function(color="blue", ci_show=False)
kmf_low.plot_survival_function(color="red", ci_show=False)
plt.title(
    f"CI: {c_index_test:.3f},\nlog-rank test p-value: {logrank_p_value:.5f}",
    fontsize=font_size,
)
plt.xlabel("Time (days)", fontsize=font_size)
plt.ylabel("Survival Probability", fontsize=font_size)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(left=0)
plt.ylim(top=1, bottom=0)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(f"{args.mode}_{timestamp}.png", dpi=600, bbox_inches="tight", format="png")
plt.show()

times = np.sort(test_validation_survival["time"].astype(float).unique())[:-1]
auc_scores, mean_auc = cumulative_dynamic_auc(
    train_surv, test_validation_surv, risk_scores_test, times
)
# plot the time-dependent AUC curve
plt.figure()
plt.plot(times, auc_scores, "k-+")
plt.axhline(
    y=mean_auc, color="black", linestyle="--", label=f"Mean AUC: {mean_auc:.2f}"
)
plt.xlabel("Time")
plt.ylabel("AUC")
plt.title("Time-Dependent AUC")
plt.grid(True)
plt.savefig(f"AUC_{args.mode}_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.show()

# set_trace()

if args.do_bootstrap:
    n_bootstraps = 100
    rng = np.random.RandomState(seed=seed_value)
    c_indices_bootstrap = []

    # store vars for KM curves
    time_grid = np.linspace(0, np.max(test_validation_survival["time"]), 10000)
    km_curves_high_risk = []
    km_curves_low_risk = []

    test_validation_survival["time"] = test_validation_survival["time"].astype(float)
    test_validation_survival["event_occurred"] = test_validation_survival[
        "event_occurred"
    ].astype(bool)

    # bootstrapping
    for _ in range(n_bootstraps):
        # resample the test set with replacement
        X_resampled, y_resampled = resample(
            X_test_validation, test_validation_survival, random_state=rng
        )

        # predict risk scores for the resampled test set
        risk_scores = model.predict(X_resampled)

        # calculate CI
        c_index = concordance_index_censored(
            y_resampled["event_occurred"], y_resampled["time"], risk_scores
        )[0]
        c_indices_bootstrap.append(c_index)

        # KM fit for resampled dataset
        median_risk = np.median(risk_scores)
        high_risk = risk_scores >= median_risk
        low_risk = risk_scores < median_risk

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(
            y_resampled["time"][high_risk], y_resampled["event_occurred"][high_risk]
        )
        kmf_low.fit(
            y_resampled["time"][low_risk], y_resampled["event_occurred"][low_risk]
        )

        # store survival probabilities at time grid
        km_curves_high_risk.append(kmf_high.predict(time_grid))
        km_curves_low_risk.append(kmf_low.predict(time_grid))

    c_indices_bootstrap = np.array(c_indices_bootstrap)
    km_curves_high_risk = np.array(km_curves_high_risk)
    km_curves_low_risk = np.array(km_curves_low_risk)

    km_median_high = np.percentile(km_curves_high_risk, 50, axis=0)
    km_lower_high = np.percentile(km_curves_high_risk, 2.5, axis=0)
    km_upper_high = np.percentile(km_curves_high_risk, 97.5, axis=0)

    km_median_low = np.percentile(km_curves_low_risk, 50, axis=0)
    km_lower_low = np.percentile(km_curves_low_risk, 2.5, axis=0)
    km_upper_low = np.percentile(km_curves_low_risk, 97.5, axis=0)

    mean_c_index = np.mean(c_indices_bootstrap)
    std_c_index = np.std(c_indices_bootstrap)
    ci_lower = np.percentile(c_indices_bootstrap, 2.5)
    ci_upper = np.percentile(c_indices_bootstrap, 97.5)

    print(f"Mean C-index: {mean_c_index:.3f}")
    print(f"Standard Deviation of C-index: {std_c_index:.3f}")
    print(f"95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")

    # plot c indices across bootstrapped samples
    plt.figure(figsize=(10, 6))
    plt.boxplot(c_indices_bootstrap, vert=True)
    plt.title("Bootstrap Distribution of C-index")
    plt.xlabel("C-index")
    plt.ylim(0.4, 0.8)
    plt.savefig(
        f"bootstrapped_cindex_{args.mode}_{timestamp}.png", dpi=300, bbox_inches="tight"
    )
    # plt.show()

    # # Extract high-risk and low-risk groups from the bootstrapped KM plot
    # high_risk_times = test_validation_survival["time"][risk_scores >= np.median(risk_scores)]
    # high_risk_events = test_validation_survival["event_occurred"][risk_scores >= np.median(risk_scores)]
    # low_risk_times = test_validation_survival["time"][risk_scores < np.median(risk_scores)]
    # low_risk_events = test_validation_survival["event_occurred"][risk_scores < np.median(risk_scores)]

    # # Compute the log-rank test p-value
    # logrank_result = logrank_test(high_risk_times, low_risk_times, high_risk_events, low_risk_events)
    # logrank_p_value = logrank_result.p_value

    # # Plot bootstrapped KM curves with the final log-rank test p-value

    # # Plot bootstrapped KM curves with uncertainty regions
    # plt.figure(figsize=(8, 6))
    # plt.plot(time_grid, km_median_high, label="High Risk", color="red", linewidth=2)
    # plt.fill_between(time_grid, km_lower_high, km_upper_high, color="red", alpha=0.3) #, label="95% CI (High Risk)")

    # plt.plot(time_grid, km_median_low, label="Low Risk", color="blue", linewidth=2)
    # plt.fill_between(time_grid, km_lower_low, km_upper_low, color="blue", alpha=0.3) #, label="95% CI (Low Risk)")

    # # plt.title("Bootstrapped Kaplan-Meier Survival Curves")
    # plt.title(f"Bootstrapped Kaplan-Meier Survival Curves\nLog-rank test p-value: {logrank_p_value:.3f}")
    # plt.xlabel("Time (days)")
    # plt.ylabel("Survival Probability")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"bootstrapped_km_{args.mode}_{timestamp}.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # # Print the final log-rank test result
    # print(f"Final Log-rank test p-value: {logrank_p_value:.4f}")


set_trace()
