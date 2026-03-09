import pandas as pd
import numpy as np

from minitorch.tensor.tensor import Tensor
from minitorch.zexamples.loan_default.src.components.data_ingestion import DataIngestion
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self) -> None:
        data_ingestor = DataIngestion()
        train_url, test_url = data_ingestor.initiate_data_ingestion()
        
        #* store the data sets
        self.train_set = pd.read_csv(train_url)
        self.test_set = pd.read_csv(test_url)
        
        #* separate the features from target
        self.train_features = self.train_set.drop(columns=['ID', 'year', 'Status'], axis=1)
        self.test_features = self.test_set.drop(columns=['ID', 'year', 'Status'], axis=1)
        
        self.train_target = self.train_set['Status']
        self.test_target = self.test_set['Status']
        
    def initiate_data_transformation(self):
        try:
            #* preprocess the data
            print('🧪 Getting the preprocessing Objects ...' )
            train_preprocessor_obj, test_preprocesor_obj = self._get_preprocessor_object()
            
            print('✍️ Applying transformation to both the training and testing datasets ...')
            train_arr = train_preprocessor_obj.fit_transform(self.train_features)
            test_arr = test_preprocesor_obj.fit_transform(self.test_features)
            print('✅ Done with the transformatiom ')
            
            #* concatenate the features and the targets to a single array
            print('Concating thr features and the targets back to gether ...')
            print(self.train_target, self.test_target)
            train_array = np.c_[train_arr, np.array(self.train_target)]
            test_array = np.c_[test_arr, np.array(self.test_target)]
            print('✅✅ Done with the concatination and the data transformatiom.')
            
            return train_array, test_array

        except Exception as e:
            raise ValueError(e)
    
    def _get_preprocessor_object(self):
        #* extract the numerical and categorical columns
        train_num_cols = self.train_features.select_dtypes(include=['int64', 'float32']).columns
        test_num_cols = self.test_features.select_dtypes(include=['int64', 'float32']).columns
        
        train_cat_cols = self.train_features.select_dtypes(include=['object']).columns
        test_cat_cols = self.test_features.select_dtypes(include=['object']).columns
        
        #* create the transformers
        numerical_imputer = KNNImputer(n_neighbors=5)
        numerical_scaler = StandardScaler(with_mean=False)
        
        categorical_imputer = SimpleImputer(strategy= 'most_frequent')
        categorical_scaler = StandardScaler(with_mean=False)
        categorical_encoder = OneHotEncoder(drop='first')
        
        numerical_col_pipeline = Pipeline(
            steps= [
                ('imputer', numerical_imputer),
                ('scaler', numerical_scaler)
            ]
        )
        
        categorical_cols_pipeline = Pipeline(
            steps= [
                ('imputer', categorical_imputer),
                ('encoder', categorical_encoder),
                ('scaler', categorical_scaler)
                
            ]
        )
        
        train_preprocessor_obj = ColumnTransformer(transformers= [
            ('numerical', numerical_col_pipeline, train_num_cols),
            ('categorical', categorical_cols_pipeline, train_cat_cols)
        ])
        
        test_preprocessor_obj = ColumnTransformer(transformers=[
            ('numerical', numerical_col_pipeline, test_num_cols),
            ('categorical', categorical_cols_pipeline, test_cat_cols)
        ])
        
        return (
            train_preprocessor_obj,
            test_preprocessor_obj
        )
        
        
if __name__ == '__main__':
    print('🧪 Testing the Data Transformation class\n')
    data_transformer = DataTransformation()
    train_arr, test_arr = data_transformer.initiate_data_transformation()
    train_tensor, test_tensor = Tensor(train_arr), Tensor(test_arr)
    print('\n')
    print(f'Train Dataset Shape\t' ,{train_tensor.shape})
    print(f'Test Dataset Shape\t', {test_tensor.shape})
    print('\n')
    print('Train Dataset')
    print(train_tensor)
    print('\nTest Dataset')
    print(test_tensor)