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
 
storagepath = "fs.azure.account.key.nytaxidata.blob.core.windows.net"

# Store Access Token -- If we change storage account then we will change access token

storageaccesstoken = "u0IA8jQWU5TwGwL61tb/onFAOHswohn8eJdy0UECLwXIXGP5FqivMS6yvUnzXigs489oj7q2z35deva+QFbjtw=="

# Set up the Access key 

spark.conf.set(storagepath,storageaccesstoken)

# Set up the stoarge path of our account 

storagePath =  "wasbs://taxidata@nytaxidata.blob.core.windows.net/azure-ml/"

# trainData for training the model

train_data =  "Azure_ml_train.parquet"

# Retraing Data for retraing the model 

retrain_data = "Azure_ml_test.parquet"

# fetch the training data from Blob

# fetch the retrain data from Blob

# if we retrain the model then we will comment  the  trainpath and uncomment the  retrainpath


trainpath = storagePath + train_data 

#trainpath = storagePath + retrain_data

training = spark.read.parquet(trainpath)

training.cache()

testpath = storagePath + retrain_data

test = spark.read.parquet(testpath)


#partition data for retrain - (train and test)

data_retrain, data_retrain_test = test.randomSplit([0.5,0.5])
data_retrain.cache()
data_retrain_test.cache()
retraining = training.union(data_retrain)

#Linear Regression Model
lr = LinearRegression(labelCol='actuals',maxIter=100)
pipeline = Pipeline(stages=[lr])

model = pipeline.fit(retraining)
model_path = 'model/lr_retrain_model'

model.write().overwrite().save(model_path)
