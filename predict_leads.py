import re
import warnings
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col, udf, trim, when
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyhive import hive
from xgboost import XGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("LeadConversion").getOrCreate()
sc = spark.sparkContext

warnings.filterwarnings("ignore")

def load_data_from_hive():
    conn = hive.Connection(host="localhost", port=10000, username="your_username")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nome_do_banco.nome_da_tabela")
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    schema = StructType([StructField(col_name, StringType(), True) for col_name in columns])
    df = spark.createDataFrame(data, schema)
    return df


def preprocess_data(df):

    numeric_cols = ['Badges - Potencial presente (R$)', 'Badges - Potencial presente (m²)',
                    'Badges - Valor médio (R$)', 'Badges - Valor total obras (R$)',
                    'Badges - Quantidade de obras']
    for col in numeric_cols:
        df = df.withColumn(col, when(col(col).isNull(), 0).otherwise(col(col)))

 
    def clean_phone(phone):
        return re.sub(r'\D', '', phone) if phone else None

    phone_cols = ['Informações Gerais - CPF', 'Localização e Contato - telefonesInfluenciador - Telefone influenciador']
    for col in phone_cols:
        df = df.withColumn(col, udf(clean_phone)(col(col)))


    text_cols = ['Informações Gerais - Nome', 'Localização e Contato - Logradouro',
                 'Localização e Contato - Complemento', 'Localização e Contato - Bairro',
                 'Localização e Contato - Município', 'Localização e Contato - UF']
    for col in text_cols:
        df = df.withColumn(col, trim(col(col)))

 
    df = df.withColumn('Badges - Potencial presente (R$)', col('Badges - Potencial presente (R$)').cast(DoubleType()))
    df = df.withColumn('Badges - Potencial presente (m²)', col('Badges - Potencial presente (m²)').cast(DoubleType()))
    df = df.withColumn('Badges - Valor médio (R$)', col('Badges - Valor médio (R$)').cast(DoubleType()))
    df = df.withColumn('Badges - Valor total obras (R$)', col('Badges - Valor total obras (R$)').cast(DoubleType()))
    df = df.withColumn('Badges - Quantidade de obras', col('Badges - Quantidade de obras').cast(IntegerType()))


    df = df.withColumn('Potencial de vendas - Potencial presente (R$)', col('Potencial de vendas - Potencial presente (R$)').cast(DoubleType()))
    df = df.withColumn('Potencial de vendas - Potencial futuro (R$)', col('Potencial de vendas - Potencial futuro (R$)').cast(DoubleType()))
    df = df.withColumn('Potencial de vendas - Potencial presente (m²)', col('Potencial de vendas - Potencial presente (m²)').cast(DoubleType()))
    df = df.withColumn('Potencial de vendas - Potencial futuro (m²)', col('Potencial de vendas - Potencial futuro (m²)').cast(DoubleType()))


    df = df.withColumn('Caracteristicas das obras - Número de obras no ano', col('Caracteristicas das obras - Número de obras no ano').cast(IntegerType()))
    df = df.withColumn('Caracteristicas das obras - Quantidade de obras', col('Caracteristicas das obras - Quantidade de obras').cast(IntegerType()))

    df = df.dropDuplicates()

    return df


def apply_kmeans(df):
    feature_cols = ['Badges - Potencial presente (R$)', 'Badges - Potencial presente (m²)',
                    'Badges - Valor médio (R$)', 'Badges - Valor total obras (R$)',
                    'Badges - Quantidade de obras']

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    kmeans = KMeans(k=5, seed=1)
    model = kmeans.fit(df)

    df = model.transform(df)
    return df


def train_xgboost_model(df):
   
    feature_cols = ['Badges - Potencial presente (R$)', 'Badges - Potencial presente (m²)',
                    'Badges - Valor médio (R$)', 'Badges - Valor total obras (R$)',
                    'Badges - Quantidade de obras']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    indexer = StringIndexer(inputCol='Converted', outputCol='label')
    
 
    xgboost = XGBClassifier()

 
    pipeline = Pipeline(stages=[assembler, indexer, xgboost])

   
    model = pipeline.fit(df)

   
    predictions = model.transform(df)

  
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy:", accuracy)

    return model

data_df = load_data_from_hive()


data_df = preprocess_data(data_df)


data_df = apply_kmeans(data_df)


trained_model = train_xgboost_model(data_df)