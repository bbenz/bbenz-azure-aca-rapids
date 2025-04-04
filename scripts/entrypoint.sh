#!/bin/bash
set -e

# Configure Spark environment for RAPIDS and XGBoost
export SPARK_CLASSPATH=/opt/rapids/rapids-4-spark_2.12-23.12.0.jar:/opt/rapids/cudf-23.12.0-cuda11.jar:/app/xgboost-rapids-aca.jar
export SPARK_SUBMIT_OPTS="--conf spark.driver.extraClassPath=/opt/rapids/rapids-4-spark_2.12-23.12.0.jar:/opt/rapids/cudf-23.12.0-cuda11.jar"

# Java 21 compatibility settings - Add JVM flags to open modules
export JAVA_OPTS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED"

# Accept command line args or use default values
DATA_SOURCE=${1:-"$DATA_SOURCE"}
if [ -z "$DATA_SOURCE" ]; then
  DATA_SOURCE="abfss://data-container@storageaccount.dfs.core.windows.net/agaricus_data.csv"
fi

# Determine if we should use GPU acceleration based on environment variable
USE_GPU=${USE_GPU:-"true"}
GPU_ENABLED_CONFIGS=""

if [ "$USE_GPU" = "true" ]; then
  echo "Starting XGBoost RAPIDS application with NVIDIA GPU acceleration"
  GPU_ENABLED_CONFIGS="--conf spark.plugins=com.nvidia.spark.SQLPlugin \
                       --conf spark.rapids.sql.enabled=true \
                       --conf spark.rapids.sql.explain=ALL \
                       --conf spark.executor.resource.gpu.amount=1 \
                       --conf spark.task.resource.gpu.amount=1 \
                       --conf spark.executor.extraJavaOptions=-Dai.rapids.cudf.prefer-pinned=true \
                       --jars /opt/rapids/rapids-4-spark_2.12-23.12.0.jar,/opt/rapids/cudf-23.12.0-cuda11.jar"
  START_TIME=$(date +%s%3N)
else
  echo "Starting XGBoost application in CPU-only mode (no RAPIDS acceleration)"
  START_TIME=$(date +%s%3N)
fi

echo "Data source: $DATA_SOURCE"
echo "Cosmos DB endpoint: $COSMOS_ENDPOINT"

# Run the Spark application with proper configurations
$SPARK_HOME/bin/spark-submit \
  --master local[*] \
  --class com.azure.rapids.xgboost.XGBoostRapidsAzure \
  $GPU_ENABLED_CONFIGS \
  --driver-java-options "$JAVA_OPTS" \
  /app/xgboost-rapids-aca.jar \
  --data-source "$DATA_SOURCE" \
  --cosmos-endpoint "$COSMOS_ENDPOINT" \
  --cosmos-key "$COSMOS_KEY" \
  --cosmos-db "VectorDB" \
  --cosmos-container "Vectors" \
  --use-gpu "$USE_GPU"

END_TIME=$(date +%s%3N)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo "XGBoost application finished"
echo "Total execution time: $EXECUTION_TIME milliseconds"