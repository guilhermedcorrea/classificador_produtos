from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.recommendation import ALS

def create_spark_session():
   
    spark = SparkSession.builder.appName("RegressionExample").getOrCreate()
    return spark

def preprocess_data(spark, input_path, target_column, concorrente_prefix="Concorrente_", sample_size=None):
  
    data = spark.read.csv(input_path, header=True, inferSchema=True)

    concorrente_columns = [col for col in data.columns if col.startswith(concorrente_prefix)]

    if sample_size and 0 < sample_size <= 1:
        data = data.sample(False, sample_size, seed=42)

    feature_cols = [col for col in data.columns if col not in [target_column] + concorrente_columns]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(data)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(data)
    data = scaler_model.transform(data)

 
    training_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

    return training_df, test_df

def train_decision_tree_regressor(training_df):
   
    
    dt_regressor = DecisionTreeRegressor(featuresCol="scaled_features", labelCol="meupreco")


    param_grid = ParamGridBuilder() \
        .addGrid(dt_regressor.maxDepth, [5, 10, 15]) \
        .addGrid(dt_regressor.minInstancesPerNode, [1, 5, 10]) \
        .build()

    evaluator = RegressionEvaluator(labelCol="meupreco", predictionCol="prediction", metricName="mse")
    cross_validator = CrossValidator(estimator=dt_regressor, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    dt_model = cross_validator.fit(training_df)

    return dt_model.bestModel

def train_random_forest_regressor(training_df):
   
    rf_regressor = RandomForestRegressor(featuresCol="scaled_features", labelCol="meupreco", seed=42)

   
    param_grid = ParamGridBuilder() \
        .addGrid(rf_regressor.maxDepth, [5, 10, 15]) \
        .addGrid(rf_regressor.minInstancesPerNode, [1, 5, 10]) \
        .addGrid(rf_regressor.numTrees, [50, 100, 150]) \
        .build()


    evaluator = RegressionEvaluator(labelCol="meupreco", predictionCol="prediction", metricName="mse")
    cross_validator = CrossValidator(estimator=rf_regressor, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    rf_model = cross_validator.fit(training_df)

    return rf_model.bestModel

def train_als_recommender(training_df):

    als = ALS(userCol="quantidade", itemCol="meupreco", ratingCol="Margem", coldStartStrategy="drop")

 
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [5, 10, 15]) \
        .addGrid(als.maxIter, [10, 20, 30]) \
        .build()

   
    evaluator = RegressionEvaluator(labelCol="Margem", predictionCol="prediction", metricName="mse")
    cross_validator = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    als_model = cross_validator.fit(training_df)

    return als_model.bestModel

def evaluate_regressor(model, test_df):

    predictions = model.transform(test_df)


    evaluator = RegressionEvaluator(labelCol="meupreco", predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)

    evaluator = RegressionEvaluator(labelCol="meupreco", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    return mse, r2

if __name__ == "__main__":
   
    spark = create_spark_session()

    input_path = r'D:\estudo2907\urls_sellers.csv'
    target_column = "meupreco"
    concorrente_prefix = "Concorrente_"

    training_df, test_df = preprocess_data(spark, input_path, target_column, concorrente_prefix=concorrente_prefix, sample_size=0.2)

   
    dt_model = train_decision_tree_regressor(training_df)
    mse_dt, r2_dt = evaluate_regressor(dt_model, test_df)

    print("DecisionTreeRegressor")
    print(f"Mean Squared Error (MSE): {mse_dt}")
    print(f"R-squared (R2): {r2_dt}")


    rf_model = train_random_forest_regressor(training_df)
    mse_rf, r2_rf = evaluate_regressor(rf_model, test_df)

    print("\nRandomForestRegressor")
    print(f"Mean Squared Error (MSE): {mse_rf}")
    print(f"R-squared (R2): {r2_rf}")

 
    als_model = train_als_recommender(training_df)
    als_predictions = als_model.transform(test_df)
    als_mse, als_r2 = evaluate_regressor(als_predictions, test_df)

    print("\nALS Recommender")
    print(f"Mean Squared Error (MSE): {als_mse}")
    print(f"R-squared (R2): {als_r2}")