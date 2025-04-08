# XGBoost RAPIDS Agaricus Mushroom Classification on Azure Container Apps

This project demonstrates how to run GPU-accelerated XGBoost machine learning models using NVIDIA RAPIDS on Azure Container Apps. The implementation uses Apache Spark with RAPIDS acceleration to process the Agaricus mushroom dataset, train an XGBoost classification model to identify edible vs. poisonous mushrooms, and store vector embeddings in Azure Cosmos DB.

Shortlink: https://aka.ms/sparkrapidsgpudemo


## Overview

The application:

1. Leverages NVIDIA RAPIDS for GPU-accelerated data processing
2. Trains an XGBoost model with GPU acceleration on the Agaricus mushroom dataset
3. Runs in Azure Container Apps with GPU support
4. Stores vector embeddings in Azure Cosmos DB for vector search capabilities
5. Reads and writes data from Azure Storage

## Dataset

This implementation uses the [Agaricus Mushroom dataset](https://archive.ics.uci.edu/dataset/73/mushroom) from the UCI Machine Learning Repository. The dataset includes descriptions of hypothetical samples of 23 species of gilled mushrooms in the Agaricus and Lepiota family. Each sample is classified as:

- Edible (e)
- Poisonous (p)

The dataset contains 8,124 instances with 22 categorical attributes like cap shape, odor, gill size, etc.

## Prerequisites

- [Azure Subscription](https://azure.microsoft.com/en-us/free/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Docker](https://www.docker.com/products/docker-desktop)
- [Maven](https://maven.apache.org/download.cgi)
- [Java 11 JDK](https://adoptopenjdk.net/)
- [NVIDIA CUDA drivers](https://developer.nvidia.com/cuda-downloads) (for local development)

## Local Development Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd bbenz-azure-aca-rapids
   ```

2. Install the required dependencies:
   ```bash
   mvn clean install
   ```

3. If you want to run the application locally with GPU support, ensure you have NVIDIA GPU with CUDA drivers installed.

## Configuration

The application uses the following configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-source` | Path to the input data file (ABFS path) | `abfss://data-container@storageaccount.dfs.core.windows.net/agaricus_data.csv` |
| `--cosmos-endpoint` | Azure Cosmos DB endpoint URL | From environment variable `COSMOS_ENDPOINT` |
| `--cosmos-key` | Azure Cosmos DB access key | From environment variable `COSMOS_KEY` |
| `--cosmos-db` | Cosmos DB database name | `VectorDB` |
| `--cosmos-container` | Cosmos DB container name | `Vectors` |

## Running the Application

### Local Execution

To run the application locally:

```bash
# Build the application
mvn clean package

# Run with sample data and Cosmos DB connection
java -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
  --data-source "/scripts/tmp_data/agaricus_data.csv" \
  --cosmos-endpoint "https://your-cosmos-account.documents.azure.com:443/" \
  --cosmos-key "your-cosmos-key"
```

You can retrieve your Cosmos DB connection details using Azure CLI:

```bash
# Get Cosmos DB connection details
COSMOS_ENDPOINT=$(az cosmosdb show --name your-cosmos-account --resource-group your-resource-group --query "documentEndpoint" -o tsv)
COSMOS_KEY=$(az cosmosdb keys list --name your-cosmos-account --resource-group your-resource-group --query "primaryMasterKey" -o tsv)
echo "COSMOS_ENDPOINT=$COSMOS_ENDPOINT" 
echo "COSMOS_KEY=$COSMOS_KEY"
```

### Comparing CPU vs GPU Performance

The application supports both CPU and GPU modes for performance comparison:

```bash
# Run with GPU acceleration (default)
java -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
  --data-source "/scripts/tmp_data/agaricus_data.csv" \
  --use-gpu true

# Run in CPU-only mode
java -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
  --data-source "/scripts/tmp_data/agaricus_data.csv" \
  --use-gpu false
```

For detailed performance comparison instructions, see [comparing-gpu-performance.md](comparing-gpu-performance.md)

### Docker Execution

To build and run the Docker container locally:

```bash
# Build the Docker image
docker build -t xgboost-rapids .

# Run the container with GPU support
docker run --gpus all \
  -e COSMOS_ENDPOINT="your-cosmos-endpoint" \
  -e COSMOS_KEY="your-cosmos-key" \
  xgboost-rapids
```

## Running with Java 21

This project supports running with Java 21, but due to changes in Java's module system, special JVM flags are needed.

### Local Execution with Java 21

Use the included `run.sh` script to run the application locally with Java 21:

```bash
# Make the script executable
chmod +x run.sh

# Run with default parameters (GPU mode)
./run.sh

# Run with specific data source
./run.sh "/path/to/data.csv"

# Run in CPU-only mode
./run.sh "./scripts/tmp_data/agaricus_data.csv" "false"

# Run with specific parameters including Cosmos DB
./run.sh "./scripts/tmp_data/agaricus_data.csv" "true" \
  --cosmos-endpoint "https://your-cosmos-account.documents.azure.com:443/" \
  --cosmos-key "your-cosmos-key"
```

Alternatively, you can run directly with the JVM flags:

```bash
java --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
     --add-opens=java.base/java.nio=ALL-UNNAMED \
     --add-opens=java.base/java.util=ALL-UNNAMED \
     --add-opens=java.base/java.lang=ALL-UNNAMED \
     --add-opens=java.base/java.util.concurrent=ALL-UNNAMED \
     --add-opens=java.base/java.net=ALL-UNNAMED \
     --add-opens=java.base/java.lang.invoke=ALL-UNNAMED \
     --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
     -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
     --data-source "./scripts/tmp_data/agaricus_data.csv" \
     --use-gpu "true"
```

### Building Different JAR Versions

This project comes with two POM files for different scenarios:

1. Default `pom.xml`: Includes all dependencies for standalone execution
2. `spark-cluster-pom.xml`: For Spark cluster environments where Spark is provided

To build with the Spark cluster POM:

```bash
mvn clean package -f spark-cluster-pom.xml
```

## Deploying to Azure

The project includes a deployment script that provisions all required Azure resources:

```bash
# Make the script executable
chmod +x scripts/deploy-to-azure.sh

# Run the deployment script
./scripts/deploy-to-azure.sh
```

This script will:

1. Create an Azure Resource Group
2. Provision an Azure Container Registry
3. Create an Azure Cosmos DB account with a database and container
4. Set up an Azure Storage account for data files
5. Build and push the Docker image to ACR
6. Create an Azure Container Apps environment with GPU support
7. Deploy the application to Azure Container Apps

### Azure Resources Created

- **Resource Group**: Contains all resources for the application
- **Azure Container Registry**: Stores the Docker image
- **Azure Cosmos DB**: Stores vector embeddings with vector search capability
- **Azure Storage**: Stores input data files
- **Azure Container Apps**: Hosts the application with GPU support

## Data Format

The application expects the mushroom dataset in a CSV format with header row. The dataset includes categorical features that are encoded for the XGBoost model. Key columns include:

- `class` - Target variable (edible='e' or poisonous='p')
- `cap-shape` - Bell=b, conical=c, convex=x, flat=f, etc.
- `cap-surface` - Fibrous=f, grooves=g, scaly=y, smooth=s
- `cap-color` - Brown=n, buff=b, cinnamon=c, gray=g, etc.
- `bruises` - Bruises=t, no=f
- `odor` - Almond=a, anise=l, creosote=c, fishy=y, etc.
- And 17 more categorical attributes

The code performs one-hot encoding on these categorical features before training.

## Project Structure

- `/src/main/java/com/azure/rapids/xgboost/`: Java source code
- `/scripts/`: Deployment and entrypoint scripts
- `/Dockerfile`: Container definition for building the application image
- `pom.xml`: Maven project configuration

## XGBoost Configuration

The XGBoost model is configured with the following parameters:

```java
params.put("eta", 0.1);
params.put("max_depth", 8);
params.put("objective", "binary:logistic");
params.put("num_round", 100);
params.put("tree_method", "gpu_hist");  // GPU-accelerated training
params.put("gpu_id", 0);
params.put("eval_metric", "auc");
```

These parameters can be adjusted in the `trainModel()` method to suit your specific use case.

## Cosmos DB Vector Search

The application stores feature vectors in Cosmos DB, which can be used with Cosmos DB's vector search capabilities. The vectors are stored along with prediction results and metadata.

## Performance Comparison

To make it easy to compare CPU and GPU performance, use the included comparison script:

```bash
# Make the script executable
chmod +x compare-cpu-gpu.sh

# Run the comparison with default dataset
./compare-cpu-gpu.sh

# Run with a specific dataset and additional parameters
./compare-cpu-gpu.sh "./path/to/your/data.csv" --cosmos-endpoint "your-endpoint" --cosmos-key "your-key"
```

The script will:
1. Run the application in CPU-only mode
2. Run the application in GPU-accelerated mode
3. Record execution times for both runs
4. Extract detailed timing metrics for each processing phase
5. Generate a comprehensive performance report in Markdown format

### Sample Performance Report

The script generates a detailed report in `./results/performance_report.md` with:
- Overall execution time comparison
- Phase-by-phase timing breakdown
- Speedup factors for each phase
- Accuracy comparison between CPU and GPU models

## Inspiration

This project was inspired by the [NVIDIA RAPIDS spark-rapids-examples](https://github.com/NVIDIA/spark-rapids-examples/tree/main/examples/XGBoost-Examples/agaricus) which demonstrates the use of XGBoost for mushroom classification. Our implementation is converted to Java running on Azure and extended to use:

- [NVIDIA RAPIDS](https://rapids.ai/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Azure Container Apps with GPU support](https://docs.microsoft.com/en-us/azure/container-apps/)
- [Azure Cosmos DB Vector Search](https://docs.microsoft.com/en-us/azure/cosmos-db/vector-search)

## Limitations and Next Steps

- The current implementation is optimized for binary classification problems
- For production use, consider implementing model evaluation metrics and cross-validation
- For very large datasets, consider implementing incremental training or distributed processing
- The vector search capabilities in Cosmos DB should be configured through the Azure Portal

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.