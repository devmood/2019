from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import resource


resource.setrlimit(resource.RLIMIT_AS, (10*(1024**3), 10*(1024**3)))
spark  = SparkSession.builder.master("local").getOrCreate()
sc = spark.sparkContext

columns = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'platform', 'city', 'device', 'current_filters', 'impressions', 'prices']

traindata = (spark.
             read.
             csv("./data/train.csv", header=True).
             withColumn("step", F.col("step").cast(IntegerType())).repartition(8))

trainZippedWithArtId = (traindata.
                        select("session_id", "step").
                        orderBy("session_id", "step").
                        repartition(1).
                        withColumn("idx", F.monotonically_increasing_id()).
                        repartition(8))
trainZippedWithArtId.persist()

splitIdx = int(.8 * trainZippedWithArtId.count())
splitSession = (trainZippedWithArtId.
                filter(F.col("idx") == splitIdx).
                select("session_id").
                first()[0])

splitUDF = F.udf(lambda idx, sess: idx < splitIdx and sess != splitSession)

trainSplitFilter = (trainZippedWithArtId.
                    withColumn("split_filter", splitUDF(F.col("idx"), F.col("session_id"))).
                    select("session_id", "step", "split_filter"))

joined = traindata.join(trainSplitFilter, ["session_id", "step"], how="left")
mask = joined["split_filter"] == True

mytrain = joined[mask].drop(F.col("split_filter"))
myPartialTest = joined[~mask].drop(F.col("split_filter"))

mytrain.select(columns).repartition(1).write.csv("./mydata/mytrain.csv", header=True)
myPartialTest.select(columns).repartition(1).write.csv("./mydata/mygt.csv", header=True)

testTmp = (myPartialTest.
           groupBy("session_id").
           agg(F.collect_list(F.col("step")).alias("list_steps"), 
               F.collect_list(F.col("action_type")).alias("list_actions")))

def last_click(steps, actions):
  x = [(s,a) for s, a in zip(steps, actions) if a.startswith("click")]
  return x[-1] if len(x) > 0 else None

schema = (StructType([
    StructField("step", IntegerType(), nullable=True),
    StructField("action", StringType(), nullable=True)]))

clickudf = F.udf(last_click, schema)

lastClickInSession = (testTmp.
                      withColumn("last_click", clickudf(F.col("list_steps"), F.col("list_actions"))).
                      select("session_id", "last_click.step", "last_click.action").
                      toDF("session_id", "newstep", "newaction"))

statementudf = F.udf(lambda s, ns, a, na, r: "null" if s == ns and a == na else r)

mytest = (myPartialTest.
          join(lastClickInSession, ["session_id"], "left").
          withColumn("reference", statementudf(F.col("step"),
                                               F.col("newstep"),
                                               F.col("action_type"),
                                               F.col("newaction"),
                                               F.col("reference"))).
          drop("newstep"))

mytest.select(columns).repartition(1).write.csv("./mydata/mytest.csv", header=True)

