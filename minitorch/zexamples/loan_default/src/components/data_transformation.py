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
        print(self.train_set.shape, self.test_set.shape)
        
        #* separate the features from target
        self.train_features = self.train_set.drop(columns=['ID', 'year', 'Status'], axis=1)
        self.test_features = self.test_set.drop(columns=['ID', 'year', 'Status'], axis=1)
        
        self.train_target = self.train_set['Status']
        self.test_target = self.test_set['Status']
        
    def initiate_data_transformation(self):
        try:
            #* preprocess the data
            print('🧪 Getting the preprocessing Object ...' )
            preprocessor_obj = self._get_preprocessor_object()
            
            print('✍️ Applying transformation to training datasets ...')
            train_arr = preprocessor_obj.fit_transform(self.train_features)
            
            print('✍️ Applying transformation to testing datasets ...')
            test_arr = preprocessor_obj.transform(self.test_features)
            print('✅ Done with the transformatiom ')
            
            #* concatenate the features and the targets to a single array
            print('Concating the features and the targets back together ...')
            y_train_array = np.array(self.train_target).reshape(-1, 1)
            y_test_array = np.array(self.test_target).reshape(-1, 1)
            
            train_array = np.c_[train_arr, y_train_array]
            test_array = np.c_[test_arr, y_test_array]
            print('✅✅ Done with the concatination and the data transformatiom.')
            
            print(train_array.shape, test_array.shape)
            return train_array, test_array

        except Exception as e:
            raise ValueError(e)
    
    def _get_preprocessor_object(self):
        #* extract the numerical and categorical columns
        num_col_identifiers = ['int32', 'int64', 'float64', 'float32']
        cat_col_identifiers = ['object']
        train_num_cols = self.train_features.select_dtypes(include= num_col_identifiers).columns
        train_cat_cols = self.train_features.select_dtypes(include= cat_col_identifiers).columns
    
        #* create the transformers
        numerical_col_pipeline = Pipeline(
            steps= [
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler(with_mean=True))
            ]
        )
        
        categorical_cols_pipeline = Pipeline(
            steps= [
                ('imputer', SimpleImputer(strategy= 'median')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]
        )
        
        preprocessor_obj = ColumnTransformer(transformers= [
            ('numerical', numerical_col_pipeline, train_num_cols),
            ('categorical', categorical_cols_pipeline, train_cat_cols)
        ])
        
        return preprocessor_obj
        
        
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