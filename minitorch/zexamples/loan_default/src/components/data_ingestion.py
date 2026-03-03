
import os
import sys
import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Layer
from minitorch.losses.losses import Loss
from minitorch.optimizers.optim import Optimizer



@dataclass
class DataConfig:
    dataset_path: str = os.path.join(os.path.dirname(os.getcwd()), 'artifacts')
    
    
class DataIngestion:
    def __init__(self):
        self.config = DataConfig()
        
    def initiate_data_ingestion(self):
        try:
            print('🧪 Ingesting data...')
            #* here we can add code to read the dataset from the specified path
            file_url = "C:\\Users\\User\\Desktop\\Loan_Default.csv"
            
            print(f'📥 Loading dataset from {file_url}...')
            #* load the dataset using pandas
            df = pd.read_csv(file_url)
            print('✅ Loaded dataset successfully.')
            
            #* split the dataset into training and testing sets
            #* and save them to the specified paths
            print('🔀 Splitting dataset into training and testing sets...')
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            print('✅ Split dataset successfully.')
            
            print('💾 Saving dataset to artifacts directory...')
            artifacts_path = Path(self.config.dataset_path)
            
            print(f'📁 Checking if directory exists ...')
            if not artifacts_path.exists():
                print(f'📁 Creating directory at {artifacts_path}...')
                os.makedirs(artifacts_path, exist_ok=True)
            else:
                print(f'📁 Directory already exists at {artifacts_path}.')
                print('Overwriting existing files in the directory...')
                print('🧹 Cleaning up old artifacts directory...')
                
                for file in artifacts_path.iterdir():
                    file.unlink()
                    
                print('✅ Cleaned up old artifacts directory successfully, ready to save new files.')
                
            print(f'💾 Saving dataset to {artifacts_path}...')
            df.to_csv(os.path.join(artifacts_path, 'dataset.csv'), index=False)
            print('✅ Saved dataset successfully.')
            
            
            print(f'💾 Saving training and testing datasets to {artifacts_path}...')
            train_path = os.path.join(artifacts_path, 'train.csv')
            test_path = os.path.join(artifacts_path, 'test.csv')
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print('✅ Saved training and testing datasets successfully.')
            print('✅ Data ingestion completed successfully.')
            
            return (train_path, test_path)
            
        except Exception as e:
            print(f'❌ Error during data ingestion: {e}')
            raise e
