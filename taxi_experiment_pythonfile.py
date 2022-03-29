from pyspark.sql import SparkSession
import mlflow
from sklearn import preprocessing
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

spark.sql('show databases').show()


raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")

train_df, test_df, validate_df = raw_data.randomSplit([0.8, 0.1, 0.1], seed=12345)

lbl = preprocessing.LabelEncoder()
featuresCols = raw_data.columns[-2:]


vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)



with mlflow.start_run():
  
  # define model and record parameters
  dt = DecisionTreeRegressor().setLabelCol("fare_amount")
  mlflow.log_param("model", "dt")
  mlflow.log_param("experiment", "1")
  mlflow.log_param("depth", dt.getMaxDepth())

  # build and fit model pipeline
  pipeline = Pipeline().setStages([vectorAssembler, vectorIndexer, dt])
  pipelineModel = pipeline.fit(train_df)
  predictionsDF = pipelineModel.transform(test_df)
  
  # evaluate fit model and record metrics
  evaluator = RegressionEvaluator().setLabelCol("fare_amount")
  r2 = evaluator.evaluate(predictionsDF, {evaluator.metricName: "r2"})
  rmse = evaluator.evaluate(predictionsDF, {evaluator.metricName: "rmse"})
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("rmse", rmse)