#!/bin/bash
# Run script for local execution with Java 21 compatibility

# Java 21 compatibility settings with comprehensive flags
JAVA_OPTS="--add-opens=java.base/java.nio=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.util=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.lang=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.util.concurrent=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.net=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.lang.reflect=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/sun.security.action=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/sun.misc=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.io=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.base/java.nio.channels=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=java.management/sun.management=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-opens=jdk.management/com.sun.management.internal=ALL-UNNAMED"

# Add necessary exports for accessing internal classes
JAVA_OPTS="$JAVA_OPTS --add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-exports=java.base/jdk.internal.ref=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-exports=java.base/jdk.internal.misc=ALL-UNNAMED"
JAVA_OPTS="$JAVA_OPTS --add-exports=java.base/sun.security.action=ALL-UNNAMED"

# Platform-specific workarounds
JAVA_OPTS="$JAVA_OPTS -Djava.security.manager=allow"
JAVA_OPTS="$JAVA_OPTS -Dsun.nio.ch.disableSystemWideOverlappingFileLockCheck=true"

# Spark-specific properties to use alternative memory allocation methods
JAVA_OPTS="$JAVA_OPTS -Dspark.memory.offHeap.enabled=false"
JAVA_OPTS="$JAVA_OPTS -Dspark.unsafe.exceptionOnMemoryLeak=false"
JAVA_OPTS="$JAVA_OPTS -Dspark.executor.allowSparkContext=true"
JAVA_OPTS="$JAVA_OPTS -Dspark.driver.extraJavaOptions=-Djdk.reflect.useDirectMethodHandle=false"
JAVA_OPTS="$JAVA_OPTS -Dspark.executor.extraJavaOptions=-Djdk.reflect.useDirectMethodHandle=false"

# Force Spark to use alternative implementation (not DirectByteBuffer constructor)
JAVA_OPTS="$JAVA_OPTS -Dspark.unsafe.platformFallback=true"
JAVA_OPTS="$JAVA_OPTS -Dspark.shuffle.compress=false"
JAVA_OPTS="$JAVA_OPTS -Dio.netty.noUnsafe=true"
JAVA_OPTS="$JAVA_OPTS -Dio.netty.tryReflectionSetAccessible=true"

# Memory settings
JAVA_OPTS="$JAVA_OPTS -Djdk.nio.maxCachedBufferSize=2000000"
JAVA_OPTS="$JAVA_OPTS -XX:+UseG1GC -XX:+DisableExplicitGC"

# Default values
DATA_SOURCE="./scripts/tmp_data/agaricus_data.csv"
USE_GPU="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-source)
      DATA_SOURCE="$2"
      shift 2
      ;;
    --use-gpu)
      USE_GPU="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

echo "Running XGBoost RAPIDS application with Java 21 compatibility flags"
echo "Data source: $DATA_SOURCE"
echo "GPU mode: $USE_GPU"

# Select the appropriate JAR file based on GPU mode
if [ "$USE_GPU" = "true" ]; then
    echo "Using GPU-optimized JAR file"
    # First check for the GPU-specific jar
    if [ -f "target/xgboost-rapids-aca-gpu-1.0-SNAPSHOT.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-gpu-1.0-SNAPSHOT.jar"
    # Then check for the shaded jar
    elif [ -f "target/xgboost-rapids-aca-1.0-SNAPSHOT-shaded.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-1.0-SNAPSHOT-shaded.jar"
    # Finally, check for the generic jar
    elif [ -f "target/xgboost-rapids-aca-1.0-SNAPSHOT.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-1.0-SNAPSHOT.jar"
    else
        echo "Error: No suitable JAR file found for GPU mode"
        exit 1
    fi
else
    echo "Using CPU-optimized JAR file"
    # First check for the CPU-specific jar
    if [ -f "target/xgboost-rapids-aca-cpu-1.0-SNAPSHOT.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-cpu-1.0-SNAPSHOT.jar"
    # Then check for the shaded jar
    elif [ -f "target/xgboost-rapids-aca-1.0-SNAPSHOT-shaded.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-1.0-SNAPSHOT-shaded.jar"
    # Finally, check for the generic jar
    elif [ -f "target/xgboost-rapids-aca-1.0-SNAPSHOT.jar" ]; then
        JAR_FILE="target/xgboost-rapids-aca-1.0-SNAPSHOT.jar"
    else
        echo "Error: No suitable JAR file found for CPU mode"
        exit 1
    fi
fi

echo "Using JAR file: $JAR_FILE"

# Run the Spark application with Java 21 compatibility flags
JAVA_EXEC=$(which java)
echo "Using Java executable: $JAVA_EXEC"

# Run with all necessary Java 17 module flags
$JAVA_EXEC \
    --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
    --add-opens=java.base/java.nio=ALL-UNNAMED \
    --add-opens=java.base/java.util=ALL-UNNAMED \
    --add-opens=java.base/java.lang=ALL-UNNAMED \
    --add-opens=java.base/java.util.concurrent=ALL-UNNAMED \
    --add-opens=java.base/java.net=ALL-UNNAMED \
    --add-opens=java.base/java.lang.invoke=ALL-UNNAMED \
    --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens=java.base/java.nio.DirectByteBuffer=ALL-UNNAMED \
    --add-opens=java.base/java.io=ALL-UNNAMED \
    --add-exports=java.base/sun.nio.ch=ALL-UNNAMED \
    --add-exports=java.base/jdk.internal.misc=ALL-UNNAMED \
    --add-exports=java.base/jdk.internal.ref=ALL-UNNAMED \
    -Dspark.serializer=org.apache.spark.serializer.JavaSerializer \
    -Dio.netty.tryReflectionSetAccessible=true \
    -Dsun.nio.ch.bugLevel="" \
    -Dfile.encoding=UTF-8 \
    -XX:+IgnoreUnrecognizedVMOptions \
    -jar $JAR_FILE \
    --data-source "$DATA_SOURCE" \
    --use-gpu "$USE_GPU"