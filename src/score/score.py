import json
from azureml.core.model import Model
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import *
from pyspark.ml.linalg import SparseVector, VectorUDT
import pyspark
  
def init():
  global spark
  global loaded_model
  spark = pyspark.sql.SparkSession.builder.appName("Azure ml services Retrained deployment").getOrCreate()
  model_path = Model.get_model_path(model_name="nyctaxi")
  loaded_model = PipelineModel.load(model_path)
 
def run(input_json):
  try:

    schema = StructType([StructField("features", VectorUDT(), True)])
    
    sc = spark.sparkContext
    input_list = json.loads(input_json)
    input_rdd = sc.parallelize(input_list)
    input_df = spark.read.json(input_rdd, schema = schema)
  
    pred_df = loaded_model.transform(input_df)
    pred_list = pred_df.collect()
    pred_array = [str(x["prediction"]) for x in pred_list]
    result = ",".join(pred_array)
    res = result.split(',')
    # you can return any data type as long as it is JSON-serializable
    return res
  except Exception as e:
    result = str(e)
    return "Internal Exception : " + result
