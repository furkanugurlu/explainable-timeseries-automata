import os
import logging
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from typing import List, Tuple, Generator
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from src.utils.config_loader import load_config, get_project_root

logger = logging.getLogger(__name__)

class SKABDataLoader:
    def __init__(self, config: dict = None):
        """
        Initialize the SKAB data loader pipeline.
        
        Args:
            config (dict, optional): Configuration dictionary. If None, loads default.
        """
        self.cfg = config if config else load_config()
        self.root_dir = get_project_root()
        
        # Fetch data path from config and resolve absolute path
        relative_path = self.cfg['data']['datasets']['skab']['path']
        self.data_dir = self.root_dir / relative_path
        
        self.subdirs = ['valve1', 'valve2']
        
    def load_all_data(self) -> pd.DataFrame:
        """
        Loads and combines all CSV files from valve1 and valve2 directories.
        Adds 'source_group' and 'source_file' to ensure traceability.
        
        Returns:
            pd.DataFrame: Combined dataframe.
        """
        all_dfs = []
        
        for subdir in self.subdirs:
            target_path = self.data_dir / subdir
            if not target_path.exists():
                logger.warning(f"Directory not found: {target_path}")
                continue
                
            # Find all CSV files
            csv_files = glob(str(target_path / "*.csv"))
            
            for file_path in csv_files:
                # SKAB files often use ';' delimiter, using 'sep=None, engine=python' is robust
                try:
                    # Reading with sep=';' as it's the default for SKAB, but handle alternatives
                    df = pd.read_csv(file_path, sep=';', index_col=None)
                except Exception:
                    df = pd.read_csv(file_path)
                
                # Add metadata columns
                df['source_group'] = subdir
                df['source_file'] = os.path.basename(file_path)
                
                all_dfs.append(df)
        
        if not all_dfs:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}/[valve1, valve2]. Please ensure the dataset is placed correctly.")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def prepare_xy_and_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Separates input features, target variable, and groups for GroupKFold.
        Drops excluded columns specified in business rules.
        """
        # Columns to strictly exclude from inputs
        exclude_cols = ['datetime', 'changepoint', 'source_group', 'source_file', 'anomaly']
        
        # Target variable
        if 'anomaly' not in df.columns:
            raise ValueError("The 'anomaly' column is missing from the dataset.")
            
        y = df['anomaly']
        
        # Feature set (Sensors only)
        # Select columns that are numeric and not in exclusion list
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Group identifier for GroupKFold
        groups = df['source_file']
        
        return X, y, groups, feature_cols

    def get_folds(self, n_splits: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields preprocessed Train/Test splits using GroupKFold.
        Strict enforcement of fit(Train) -> transform(Test) prevents leakage.
        """
        df = self.load_all_data()
        X, y, groups, _ = self.prepare_xy_and_groups(df)
        
        gkf = GroupKFold(n_splits=n_splits)
        
        # Extract source files to pass to GroupKFold indices
        # GroupKFold uses index mapping based on group unique values
        
        for fold_idx, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups)):
            # Split
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            
            # 1. Normalization: MinMaxScaler
            scaler = MinMaxScaler()
            # Fit ONLY on train
            X_train_scaled = scaler.fit_transform(X_train)
            # Transform test
            X_test_scaled = scaler.transform(X_test)
            
            # 2. Dimensionality Reduction: PCA (1 component / PC1)
            pca = PCA(n_components=1)
            # Fit ONLY on train scaled
            X_train_pca = pca.fit_transform(X_train_scaled)
            # Transform test scaled
            X_test_pca = pca.transform(X_test_scaled)
            
            logger.info(f"Fold {fold_idx + 1} processed. Train: {X_train_pca.shape}, Test: {X_test_pca.shape}")
            
            yield X_train_pca, X_test_pca, y_train, y_test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        loader = SKABDataLoader()
        logger.info("Initializing SKAB testing pipeline...")

        fold_gen = loader.get_folds(n_splits=5)
        for i, (X_tr, X_te, y_tr, y_te) in enumerate(fold_gen):
            logger.info(f"Fold {i+1} ready. Train size: {X_tr.shape}, Test size: {X_te.shape}")
            break

    except FileNotFoundError as e:
        logger.warning("[!] Pipeline code created successfully.")
        logger.warning(f"[!] Status: {e}")
        logger.warning("[!] Instructions: Place SKAB 'valve1' and 'valve2' folders under 'data/raw/skab/'.")
