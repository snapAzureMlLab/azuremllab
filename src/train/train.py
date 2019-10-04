import os
import argparse
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline,PipelineModel


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--AZUREML_RUN_TOKEN')
PARSER.add_argument('--AZUREML_RUN_ID')
PARSER.add_argument('--AZUREML_ARM_SUBSCRIPTION')
PARSER.add_argument('--AZUREML_ARM_RESOURCEGROUP')
PARSER.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
PARSER.add_argument('--AZUREML_ARM_PROJECT_NAME')
PARSER.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')
PARSER.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
PARSER.add_argument('--AZUREML_SERVICE_ENDPOINT')
PARSER.add_argument('--MODEL_PATH')
ARGS = PARSER.parse_args()

# Blob Configuration

# Blob Storage path --- if we want change the blob storage account then we will change storage path
 
storage_path = "fs.azure.account.key.nytaxidata.blob.core.windows.net"

# Store Access Token -- If we change storage account then we will change access token

storageaccess_token = "u0IA8jQWU5TwGwL61tb/onFAOHswohn8eJdy0UECLwXIXGP5FqivMS6yvUnzXigs489oj7q2z35deva+QFbjtw=="

# Set up the Access key 

spark.conf.set(storage_path,storageaccess_token)

# Set up the stoarge path of our account 

blob_path =  "wasbs://taxidata@nytaxidata.blob.core.windows.net/azure-ml/"

# trainData for training the model

#train_data =  "Azure_ml_train.parquet"

# (Uncomment for Retraining) Retraing Data for retraing the model 

retrain_data   = "Azure_ml_test.parquet"



# if we retrain the model then we will comment  the  trainpath of train_data and uncomment the  train path of retrain_data


#trainpath = blob_path + train_data 

trainpath = blob_path + retrain_data

training = spark.read.parquet(trainpath)

training.cache()

#Linear Regression Model
lr = LinearRegression(labelCol='actuals',maxIter=1000)
pipeline = Pipeline(stages=[lr])

model = pipeline.fit(training)
model_path = 'model/lr_retrain_model'

model.write().overwrite().save(model_path)
