#!/bin/bash
# Run script for local execution with Java 21 compatibility

# Java 21 compatibility settings - Add JVM flags to open modules
JAVA_OPTS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED"

# Check if data source is specified
DATA_SOURCE=${1:-"./scripts/tmp_data/agaricus_data.csv"}
USE_GPU=${2:-"true"}

echo "Running XGBoost RAPIDS application with Java 21 compatibility flags"
echo "Data source: $DATA_SOURCE"
echo "GPU mode: $USE_GPU"

# Run the Spark application with Java 21 compatibility flags
java $JAVA_OPTS -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
    --data-source "$DATA_SOURCE" \
    --use-gpu "$USE_GPU" \
    "$@"