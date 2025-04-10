# Use NVIDIA CUDA base image for GPU support with Java 21
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin:$SPARK_HOME/bin

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Apache Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.1-bin-hadoop3.tgz && \
    mv spark-3.5.1-bin-hadoop3 /opt/spark && \
    rm spark-3.5.1-bin-hadoop3.tgz

# Download RAPIDS Accelerator for Apache Spark
RUN mkdir -p /opt/rapids && \
    wget -O /opt/rapids/rapids-4-spark_2.12-23.12.0.jar https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/23.12.0/rapids-4-spark_2.12-23.12.0.jar && \
    wget -O /opt/rapids/cudf-23.12.0-cuda11.jar https://repo1.maven.org/maven2/ai/rapids/cudf/23.12.0/cudf-23.12.0-cuda11.jar

# Create app directory
WORKDIR /app

# Copy application JAR
COPY target/xgboost-rapids-aca-1.0-SNAPSHOT.jar /app/xgboost-rapids-aca.jar

# Copy any additional required files
COPY scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set environment variables for Cosmos DB (will be overridden by ACA deployment)
ENV COSMOS_ENDPOINT=""
ENV COSMOS_KEY=""

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]