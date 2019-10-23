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


# For the FIRST TIME train the model using "Azure_ml_train.parquet". So be sure to comment line "train_data   = "Azure_ml_test.parque"

# To RETRAIN the model, use "Azure_ml_test.parquet" So be sure to comment "train_data =  "Azure_ml_train.parquet"
      
train_data =  "Azure_ml_train.parquet" # Uncomment this line while training for the FIRST TIME

# train_data   = "Azure_ml_test.parquet" #Uncomment this line for RETRAINING the model


trainpath = blob_path + train_data 


training = spark.read.parquet(trainpath)

training.cache()

#Linear Regression Model
lr = LinearRegression(labelCol='actuals',maxIter=1000)
pipeline = Pipeline(stages=[lr])

model = pipeline.fit(training)

model_path = '/model/train_model'

model.write().overwrite().save(model_path)

