"""LightGBM-based trainer for directional trend prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .model_manager import ModelManager


@dataclass
class TrainingSummary:
    """Structured response describing a training run."""

    model_name: str
    metrics: Dict[str, float]
    artifact_path: str
    metadata: Dict[str, Any]


class LightGBMTrainer:
    """Encapsulates feature prep, LightGBM training, and persistence."""

    def __init__(
        self,
        model_manager: ModelManager,
        *,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_manager = model_manager
        self.params = default_params or {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": 0.05,
            "num_leaves": 32,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
        }

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        num_boost_round: int = 200,
        early_stopping_rounds: int = 20,
        validation_fraction: float = 0.2,
        model_name: Optional[str] = None,
        use_cv: bool = True,
        cv_folds: int = 5,
    ) -> TrainingSummary:
        if features.empty or labels.empty:
            raise ValueError("Cannot train LightGBM model with empty dataset.")
        aligned_labels = labels.loc[features.index]

        if use_cv and len(features) >= cv_folds * 2:
            # Use cross-validation for better evaluation
            cv_result = lgb.cv(
                self.params,
                lgb.Dataset(features, label=aligned_labels),
                num_boost_round=num_boost_round,
                nfold=cv_folds,
                stratified=False,
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(False)] if early_stopping_rounds else [lgb.log_evaluation(False)],
                return_cvbooster=True,
            )
            booster = cv_result['cvbooster']
            # Use the first fold's booster as representative
            booster = booster.boosters[0]
            best_iteration = cv_result.get('best_iteration', num_boost_round)
        else:
            # Fallback to simple train/valid split
            train_data, valid_data, valid_names = self._build_datasets(
                features, aligned_labels, validation_fraction
            )

            callbacks = []
            if early_stopping_rounds and early_stopping_rounds > 0 and len(valid_names) > 1:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            try:
                callbacks.append(lgb.log_evaluation(False))
            except Exception:
                pass

            # Try with callbacks first (newer LightGBM), fallback to direct args (older versions)
            try:
                booster = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, valid_data] if valid_data is not None else [train_data],
                    valid_names=valid_names,
                    callbacks=callbacks if callbacks else None,
                )
            except TypeError as e:
                if "early_stopping_rounds" in str(e):
                    # Fallback for older LightGBM versions
                    booster = lgb.train(
                        self.params,
                        train_data,
                        num_boost_round=num_boost_round,
                        valid_sets=[train_data, valid_data] if valid_data is not None else [train_data],
                        valid_names=valid_names,
                        early_stopping_rounds=early_stopping_rounds if valid_data else None,
                        verbose_eval=False,
                    )
                else:
                    raise

        preds = booster.predict(features)
        y_true = aligned_labels.to_numpy()
        metrics = {
            "accuracy": float(accuracy_score(y_true, (preds > 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0,
        }

        metadata = {
            "name": model_name or "lightgbm_trend",
            "params": self.params,
            "metrics": metrics,
            "num_features": features.shape[1],
            "feature_names": list(features.columns),
        }
        artifact_path = self.model_manager.save_model(booster, metadata)
        return TrainingSummary(
            model_name=metadata["name"],
            metrics=metrics,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )

    def _build_datasets(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        validation_fraction: float,
    ) -> Tuple[lgb.Dataset, Optional[lgb.Dataset], List[str]]:
        """Construct chronological LightGBM datasets (train + optional valid)."""
        total = len(features)
        valid_fraction = min(max(validation_fraction, 0.0), 0.5)
        if total < 2 or valid_fraction <= 0.0:
            logging.debug("Validation split disabled (%s rows, fraction=%s).", total, valid_fraction)
            return lgb.Dataset(features, label=labels), None, ["train"]

        split_idx = max(int(total * (1.0 - valid_fraction)), 1)
        if split_idx >= total:
            split_idx = total - 1
        train_feats = features.iloc[:split_idx]
        valid_feats = features.iloc[split_idx:]
        train_labels = labels.iloc[:split_idx]
        valid_labels = labels.iloc[split_idx:]
        if valid_feats.empty:
            logging.debug("Validation split produced 0 rows; using entire dataset for training.")
            return lgb.Dataset(features, label=labels), None, ["train"]

        train_dataset = lgb.Dataset(train_feats, label=train_labels)
        valid_dataset = lgb.Dataset(valid_feats, label=valid_labels, reference=train_dataset)
        return train_dataset, valid_dataset, ["train", "valid"]

    def walk_forward_train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        window_size: int = 1000,
        step_size: int = 200,
        min_train_size: int = 500,
        model_name: Optional[str] = None,
    ) -> TrainingSummary:
        """Walk-forward training for better out-of-sample validation."""
        if features.empty or labels.empty or len(features) < min_train_size:
            raise ValueError("Insufficient data for walk-forward training.")
        
        aligned_labels = labels.loc[features.index]
        total_len = len(features)
        models = []
        scores = []
        
        for start in range(0, total_len - window_size, step_size):
            end = min(start + window_size, total_len)
            train_end = max(start + min_train_size, end - 200)  # Ensure validation set
            
            train_features = features.iloc[start:train_end]
            train_labels = aligned_labels.iloc[start:train_end]
            valid_features = features.iloc[train_end:end]
            valid_labels = aligned_labels.iloc[train_end:end]
            
            if valid_features.empty:
                continue
            
            # Train on this window
            train_data = lgb.Dataset(train_features, label=train_labels)
            valid_data = lgb.Dataset(valid_features, label=valid_labels, reference=train_data)
            
            callbacks = [lgb.early_stopping(20), lgb.log_evaluation(False)]
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
            models.append(model)
            
            # Score on validation
            preds = model.predict(valid_features)
            auc = roc_auc_score(valid_labels, preds) if len(valid_labels.unique()) > 1 else 0.5
            scores.append(auc)
        
        if not models:
            raise ValueError("No models trained in walk-forward.")
        
        # Use the best model
        best_idx = scores.index(max(scores))
        best_model = models[best_idx]
        
        # Final metrics on entire dataset
        preds = best_model.predict(features)
        y_true = aligned_labels.to_numpy()
        metrics = {
            "accuracy": float(accuracy_score(y_true, (preds > 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0,
            "walk_forward_avg_auc": float(np.mean(scores)),
        }
        
        metadata = {
            "name": model_name or "walk_forward_lightgbm",
            "params": self.params,
            "metrics": metrics,
            "num_features": features.shape[1],
            "walk_forward_windows": len(models),
            "feature_names": list(features.columns),
        }
        artifact_path = self.model_manager.save_model(best_model, metadata)
        return TrainingSummary(
            model_name=metadata["name"],
            metrics=metrics,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )
