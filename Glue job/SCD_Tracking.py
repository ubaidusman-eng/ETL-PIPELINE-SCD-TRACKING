import sys
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_date, max as spark_max
from awsglue.utils import getResolvedOptions
import boto3
from awsglue.dynamicframe import DynamicFrame
from awsglue.context import GlueContext
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import count

# Create Glue and Spark contexts
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session


args = getResolvedOptions(sys.argv, ["S3_INPUT_PATH", "S3_OUTPUT_PATH", "DDB_TABLE"])

S3_INPUT_PATH = args["S3_INPUT_PATH"]
S3_OUTPUT_PATH = args["S3_OUTPUT_PATH"]
DDB_TABLE = args["DDB_TABLE"]

spark = SparkSession.builder.appName("SCD_Type2_Demo").getOrCreate()

# Step 1: Read new data

new_df = spark.read.option("header", True).csv(S3_INPUT_PATH)
new_df = new_df.withColumn("amount", col("amount").cast("double"))

# Step 2: Load existing data from DynamoDB (if any)


duplicate_rows_df = (
    new_df.groupBy(new_df.columns)
          .agg(count("*").alias("dup_count"))
          .filter(col("dup_count") > 1)
)

if duplicate_rows_df.count() > 0:
    print("Exact duplicate rows found in new_df — they will be removed:")
    duplicate_rows_df.show(truncate=False)
else:
    print("✅ No exact duplicate rows found in new_df.")

#Remove complete duplicates from new_df
new_df = new_df.dropDuplicates()
print("Rows in new_df after removing exact duplicates:", new_df.count())

dynamodb = boto3.client("dynamodb")
scan = dynamodb.scan(TableName=DDB_TABLE)
if scan["Items"]:
    old_df = spark.createDataFrame([
    {k: list(v.values())[0] for k, v in i.items()} for i in scan["Items"]
    ])
else:
    old_df = spark.createDataFrame([], new_df.schema)

# Step 3: Identify changed or new rows


join_cols = ["customer_id"]

# Rename duplicate columns in old_df
old_renamed_df = old_df.select(
    [col(c).alias(c + "_old") if c != "customer_id" else col(c) for c in old_df.columns]
)

# Perform the join
merged_df = new_df.join(old_renamed_df, on="customer_id", how="left")

# Detect changes

changed_df = merged_df.filter(
(col("amount") != col("amount_old")) |
(col("name") != col("name_old")) |
(col("city") != col("city_old"))
)
print("Rows in new_df:", new_df.count())
print("Rows in old_df:", old_df.count())

if old_df.count() == 0:
    print("First run detected — inserting all new records as version 1")
    changed_df = new_df

print("Rows in changed_df:", changed_df.count())
# Step 4: Assign version numbers

if old_df.count() > 0:
    max_version_df = old_df.groupBy("customer_id").agg(spark_max("version").alias("max_version"))
    new_version_df = changed_df.join(max_version_df, "customer_id", "left")
else:
    new_version_df = changed_df.withColumn("max_version", lit(0))

# 🔹 Assign unique version numbers within this batch
window_spec = Window.partitionBy("customer_id").orderBy(lit(None))
new_version_df = new_version_df.withColumn("row_num", row_number().over(window_spec))
new_version_df = new_version_df.withColumn("version", (col("max_version") + col("row_num")).cast("int"))
new_version_df = new_version_df.drop("row_num")


# Add metadata

final_df = (
    new_version_df
    .withColumn("is_current", lit(1))
    .withColumn("start_date", current_date())
    .withColumn("end_date", lit(None).cast("date"))
    .select("customer_id", "name", "email", "city", "amount", "version", "is_current", "start_date", "end_date")
)


# Step 5: Write back to DynamoDB and S3
print(final_df)

dyf_final = DynamicFrame.fromDF(final_df, glueContext, "dyf_final")

glueContext.write_dynamic_frame_from_options(
    frame=dyf_final,
    connection_type="dynamodb",
    connection_options={
        "dynamodb.output.tableName": "scd_customers2",
        "dynamodb.throughput.write.percent": "1.0"
    }
)

final_df.write.mode("overwrite").parquet(S3_OUTPUT_PATH)
