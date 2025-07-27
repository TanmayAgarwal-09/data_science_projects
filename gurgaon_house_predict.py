import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
 
Model_file="model.pkl"
Pipeline_file="pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())])
    
    cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)])
    
    return full_pipeline

if not os.path.exists(Model_file):
    #train
    housing=pd.read_csv("house_data.csv")
    housing['income_cat']=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, _ in split.split(housing, housing['income_cat']):
       housing = housing.loc[train_index].drop("income_cat", axis=1)   
    housing_labels = housing["median_house_value"].copy()   
    housing_features = housing.drop("median_house_value", axis=1)
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]
    pipeline=build_pipeline(num_attribs,cat_attribs)
    housing_prepared=pipeline.fit_transform(housing_features)
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    #save model and pipeline
    joblib.dump(model,Model_file)
    joblib.dump(pipeline,Pipeline_file)
    print("model train and saved")

else:
    #load model and pipeline
    model=joblib.load(Model_file)
    pipeline=joblib.load(Pipeline_file)
    print("model loaded")
    input_data=pd.read_csv("input_data.csv")
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["median_house_value"]=predictions
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")



                           


