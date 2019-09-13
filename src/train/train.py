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


trainpath = "wasbs://taxidata@taxinydata.blob.core.windows.net/Azure_ml_train.parquet"

# trainpath = "wasbs://taxidata@taxinydata.blob.core.windows.net/Azure_ml_test.parquet"
training = spark.read.parquet(trainpath)
training.cache()
#load test data for re-train

testpath = "wasbs://taxidata@taxinydata.blob.core.windows.net/Azure_ml_test.parquet"
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

model_name = "model/lr_retrain_model" 

model_name_blob = os.path.join(model_name)
print("copy model from to local")

model_local = "file:" + os.getcwd() + model_name

dbutils.fs.cp(model_name, model_local, True)