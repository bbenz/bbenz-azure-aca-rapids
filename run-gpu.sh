#!/bin/bash
# Enhanced run script for GPU mode compatible with Java 21
# Addresses the DirectByteBuffer compatibility issue and other JVM module restrictions

# Extensive Java 21 compatibility flags for RAPIDS and Spark
JAVA_OPTS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  --add-opens=java.base/java.util=ALL-UNNAMED \
  --add-opens=java.base/java.lang=ALL-UNNAMED \
  --add-opens=java.base/java.util.concurrent=ALL-UNNAMED \
  --add-opens=java.base/java.net=ALL-UNNAMED \
  --add-opens=java.base/java.lang.invoke=ALL-UNNAMED \
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
  --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED \
  --add-opens=java.base/sun.security.action=ALL-UNNAMED \
  --add-exports=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-exports=java.base/jdk.internal.ref=ALL-UNNAMED \
  --add-exports=java.base/jdk.internal.misc=ALL-UNNAMED \
  --add-exports=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-exports=java.base/sun.security.action=ALL-UNNAMED"

# Memory and reflection settings to help with DirectByteBuffer access
JAVA_OPTS="$JAVA_OPTS -Djdk.nio.maxCachedBufferSize=2000000 -Dsun.reflect.inflationThreshold=2147483647"

# Spark specific flags to disable features that might cause JVM compatibility issues
JAVA_OPTS="$JAVA_OPTS -Dspark.memory.offHeap.enabled=false -Dspark.unsafe.exceptionOnMemoryLeak=false"

# Run the application with GPU mode explicitly enabled
java $JAVA_OPTS -jar $JAR_FILE \
    --data-source "$DATA_SOURCE" \
    --use-gpu "true" \
    $PARAMS
