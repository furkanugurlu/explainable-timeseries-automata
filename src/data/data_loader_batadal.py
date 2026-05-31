import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from src.utils.config_loader import load_config, get_project_root

logger = logging.getLogger(__name__)

class BATADALDataLoader:
    def __init__(self, config: dict = None):
        """
        Initialize the BATADAL data loader pipeline.
        
        Args:
            config (dict, optional): Configuration dictionary. If None, loads default.
        """
        self.cfg = config if config else load_config()
        self.root_dir = get_project_root()
        
        relative_path = self.cfg['data']['datasets']['batadal']['path']
        self.data_dir = self.root_dir / relative_path
        
    def find_training_dataset_2(self) -> Path:
        """
        Looks for the 'Training Dataset 2' file in the directory.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"BATADAL directory not found at: {self.data_dir}")
            
        # Scan files looking for indicators of "Dataset 2"
        for file in self.data_dir.glob("*.csv"):
            lower_name = file.name.lower()
            # Matches common naming conventions like "BATADAL_dataset02.csv" or "Training_Dataset_2.csv"
            if "dataset02" in lower_name.replace("_", "").replace(" ", "") or \
               "dataset2" in lower_name.replace("_", "").replace(" ", ""):
                return file
                
        # If direct search fails, list all CSVs and let the user pick or raise error
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
        logger.warning(f"Could not explicitly find 'Dataset 2'. Defaulting to: {csv_files[0].name}")
        return csv_files[0]

    def load_and_sort_data(self) -> pd.DataFrame:
        """
        Loads the selected dataset and sorts it chronologically.
        """
        data_file = self.find_training_dataset_2()
        logger.info(f"Loading file: {data_file.name}")
        
        # BATADAL sometimes uses comma, sometimes different separators. Strip whitespace.
        df = pd.read_csv(data_file, skipinitialspace=True)
        
        # Standardizing label column
        # Common aliases for BATADAL labels: 'ATT_FLAG', 'Label'
        if 'ATT_FLAG' in df.columns:
            pass
        elif 'Label' in df.columns:
            df.rename(columns={'Label': 'ATT_FLAG'}, inplace=True)
        
        # DateTime Column Standardizing & Sorting
        datetime_col = None
        for col in df.columns:
            if col.upper() in ['DATETIME', 'TIMESTAMP', 'DATE']:
                datetime_col = col
                break
        
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True)
            df = df.sort_values(by=datetime_col).reset_index(drop=True)
            self.datetime_col = datetime_col
        else:
            logger.warning("No chronological column detected. Assuming data is already ordered.")
            self.datetime_col = None
            
        return df

    def get_processed_splits(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data chronologically (60-20-20 default) and applies preprocessing.
        Prevents data leakage by fitting transformation only on Train data.
        
        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        df = self.load_and_sort_data()
        
        # Identify target column (ATT_FLAG is standard for BATADAL)
        target_col = None
        possible_targets = ['ATT_FLAG', 'anomaly', 'Label']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
                
        if not target_col:
            raise ValueError(f"Target column not found. Looked for: {possible_targets}")
            
        # Features: Drop label and chronological columns
        exclude_cols = [target_col]
        if hasattr(self, 'datetime_col') and self.datetime_col:
            exclude_cols.append(self.datetime_col)
            
        X = df.drop(columns=exclude_cols)
        
        # Ensure all input columns are converted to numeric, force error on artifacts
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        # Binarize: BATADAL uses ATT_FLAG=1 for attack, -1 or 0 for normal depending on version.
        # Treat any positive value as anomaly (1), everything else as normal (0).
        y = (df[target_col].fillna(0).values > 0).astype(int)
        
        # Chronological index calculation
        total_rows = len(df)
        train_end = int(total_rows * train_ratio)
        val_end = int(total_rows * (train_ratio + val_ratio))
        
        # Splitting (NO SHUFFLE)
        X_train = X.iloc[:train_end].values
        X_val = X.iloc[train_end:val_end].values
        X_test = X.iloc[val_end:].values
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        logger.info(f"Splitting complete -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 1. Normalization: MinMaxScaler
        scaler = MinMaxScaler()
        # Fit only on Train
        X_train_scaled = scaler.fit_transform(X_train)
        # Apply to Val and Test
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. PCA Dimensionality Reduction (1 component)
        pca = PCA(n_components=1)
        # Fit only on Train Scaled
        X_train_final = pca.fit_transform(X_train_scaled)
        # Apply to Val and Test
        X_val_final = pca.transform(X_val_scaled)
        X_test_final = pca.transform(X_test_scaled)
        
        logger.info(f"Final shape after PCA reduction -> Train: {X_train_final.shape}")
        
        return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        loader = BATADALDataLoader()
        logger.info("Initializing BATADAL Chronological Pipeline...")

        results = loader.get_processed_splits()
        X_tr, X_v, X_te, y_tr, y_v, y_te = results

        logger.info("Success!")
        logger.info(f"X Train Shape: {X_tr.shape} | Target Shape: {y_tr.shape}")
        logger.info(f"X Val Shape:   {X_v.shape} | Target Shape: {y_v.shape}")
        logger.info(f"X Test Shape:  {X_te.shape} | Target Shape: {y_te.shape}")

    except FileNotFoundError as e:
        logger.warning("[!] Pipeline logic built. Data not found.")
        logger.warning(f"[!] Status: {e}")
        logger.warning("[!] Instructions: Put BATADAL Training Dataset 2 inside 'data/raw/batadal/'.")
