import os
import pyodbc
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.fpm import FPGrowth

os.environ["SPARK_HOME"] = r"D:\teste31\spark"
os.environ["JAVA_HOME"] = r"D:\teste31\jdk-20.0.2"
os.environ["HADOOP_HOME"] = r"D:\teste31\spark\hadoop"
os.environ["HIVE_HOME"] = r"D:\teste31\apache-hive-3.1.2"
os.environ["LIVY_CONF_DIR"] = r"apache-livy-0.7.1"

conf = SparkConf()
conf.set("spark.master", "local[*]")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.sql.adaptive.enabled", "true")
conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
conf.set("spark.dynamicAllocation.enabled", "false")
conf.set("spark.sql.adaptive.optimizeSkewsInRebalancePartitions.enabled", "true")
conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
conf.set("spark.sql.statistics.size.autoUpdate.enabled", "true")
conf.set("spark.sql.inMemoryColumnarStorage.compressed", "true")
conf.set("hive.exec.dynamic.partition", "true")
conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
conf.set("spark.sql.ansi.enabled", "true")
conf.set('spark.driver.extraClassPath', r"D:\teste31\spark\mssql-jdbc-12.2.0.jre11.jar")
conf.set('spark.executor.extraClassPath', r"D:\teste31\spark\mssql-jdbc-12.2.0.jre11.jar")

spark = SparkSession.builder \
    .appName("Exemplo PySpark com Hive") \
    .config(conf=conf) \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("spark.jars", ",".join([
        os.path.join(os.environ["HIVE_HOME"], "lib", "hive-exec-3.1.2.jar"),
        os.path.join(os.environ["HIVE_HOME"], "lib", "hive-metastore-3.1.2.jar"),
    ])) \
    .enableHiveSupport() \
    .getOrCreate()


def create_rdd_from_hive():

    server = 'DESKTOP-LRA2H5S'
    database = 'OLIST'
    username = 'sa'
    password = '123'
    conn_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(conn_string)
    cursor = conn.cursor()
    
  
    query = """
        WITH CTE_Pedidos AS (
            SELECT p.order_id,
                   STRING_AGG(i.product_id, ',') as products
              FROM pedidos p
             INNER JOIN pedidos_itens i ON p.order_id = i.order_id
             GROUP BY p.order_id
        )
        SELECT order_id, products
          FROM CTE_Pedidos
    """
    
    cursor.execute(query)

    data_rdd = cursor.fetchall()
    mapped_rdd = spark.sparkContext.parallelize(data_rdd).map(lambda row: (row[0], row[1]))
    
    
    cursor.close()
    conn.close()
    
    return mapped_rdd


data_rdd = create_rdd_from_hive()

schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("products", StringType(), True)
])
df = spark.createDataFrame(data_rdd, schema)


print("DataFrame com produtos agrupados (10 primeiras linhas):")
df.show(truncate=False)

fp_growth = FPGrowth(itemsCol="products", minSupport=0.1, minConfidence=0.5) 
model = fp_growth.fit(df)