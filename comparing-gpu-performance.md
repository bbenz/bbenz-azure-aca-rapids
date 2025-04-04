# Comparing CPU vs GPU Performance with XGBoost Agaricus

This guide provides instructions on how to deploy and compare the performance of CPU vs GPU-accelerated XGBoost implementations for the Agaricus mushroom classification task.

## Local Development and Testing

1. **Build the application**:
   ```bash
   mvn clean package
   ```

2. **Run locally in CPU mode**:
   ```bash
   java -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar --use-gpu false
   ```

3. **Run locally in GPU mode** (requires NVIDIA GPU with CUDA drivers):
   ```bash
   java -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar --use-gpu true
   ```

## Deploying to Azure Container Apps

The `deploy-to-azure.sh` script supports deploying both CPU and GPU versions of the application simultaneously for performance comparison. The script:

1. Creates all necessary Azure resources (Resource Group, ACR, Cosmos DB, Storage)
2. Downloads and processes the Agaricus mushroom dataset from UCI
3. Uploads the dataset to Azure Storage with proper headers
4. Builds and pushes the Docker image
5. Deploys both CPU and GPU versions of the application (if requested)

### Deployment Options

1. **Deploy both versions for comparison**:
   ```bash
   chmod +x scripts/deploy-to-azure.sh
   ./scripts/deploy-to-azure.sh
   ```

2. **Deploy only CPU version**:
   ```bash
   ./scripts/deploy-to-azure.sh --cpu-only
   ```

3. **Deploy only GPU version**:
   ```bash
   ./scripts/deploy-to-azure.sh --gpu-only
   ```

## Enabling GPU in Azure Container Apps

To fully enable GPU support in Azure Container Apps:

1. Azure Container Apps requires special GPU-enabled compute, which is in preview as of April 2025
2. Update the Container App to use a GPU-enabled compute pool:

   ```bash
   az containerapp update \
     --name agaricus-gpu-app \
     --resource-group your-resource-group \
     --cpu 4 \
     --memory 16Gi \
     --scale-rule-name gpu-rule \
     --scale-rule-type azure-gpu \
     --min-replicas 1 \
     --max-replicas 1
   ```

## Comparing Performance

The implementation includes detailed timing information for each phase of the processing:
- Data loading and preparation
- Model training
- Prediction
- Cosmos DB storage
- Total execution time

You can compare performance between CPU and GPU versions with these commands:

1. **View CPU app logs**:
   ```bash
   az containerapp logs show --name agaricus-cpu-app --resource-group your-resource-group
   ```

2. **View GPU app logs**:
   ```bash
   az containerapp logs show --name agaricus-gpu-app --resource-group your-resource-group
   ```

3. **Extract timing metrics** (example using grep):
   ```bash
   az containerapp logs show --name agaricus-cpu-app --resource-group your-resource-group | grep "completed in"
   az containerapp logs show --name agaricus-gpu-app --resource-group your-resource-group | grep "completed in"
   ```

## Creating Performance Reports

Create a performance comparison report with:

```bash
echo "CPU vs GPU Performance Comparison" > performance_report.txt
echo "==================================" >> performance_report.txt
echo "CPU Timings:" >> performance_report.txt
az containerapp logs show --name agaricus-cpu-app --resource-group your-resource-group | grep "completed in\|Processing mode" >> performance_report.txt
echo "" >> performance_report.txt
echo "GPU Timings:" >> performance_report.txt
az containerapp logs show --name agaricus-gpu-app --resource-group your-resource-group | grep "completed in\|Processing mode" >> performance_report.txt
```

## Expected Performance Difference

With the Agaricus mushroom dataset:
- **Data loading phase**: Similar performance between CPU and GPU
- **Model training phase**: Expect 2-10x speedup with GPU depending on GPU type
- **Prediction phase**: Expect 1.5-3x speedup with GPU
- **Total time**: Expect significant improvement with GPU, especially for model training

## Sample Performance Results

Below is an example of expected performance differences between CPU and GPU modes:

| Processing Step | CPU Time (ms) | GPU Time (ms) | Speedup Factor |
|-----------------|--------------|--------------|----------------|
| Data preparation | 2,500 | 2,300 | 1.1x |
| Model training | 12,000 | 1,500 | 8.0x |
| Prediction | 3,000 | 1,200 | 2.5x |
| **Total processing** | **18,000** | **5,300** | **3.4x** |

*Note: Actual performance will vary based on hardware specifications, dataset size, and other factors.*

## Visualization of Results

You can visualize the performance difference using tools like:

1. Excel or Google Sheets for creating bar charts
2. Jupyter Notebooks with matplotlib
3. Azure Monitor dashboards for production monitoring

A comparison chart typically shows the stark difference in training time, which is where GPU acceleration provides the most significant benefits.

## Optimizing Performance Further

For even better GPU performance:
- Increase batch size for larger datasets
- Experiment with different XGBoost parameters for GPU
- Utilize multiple GPUs in a distributed setup
- Consider using Azure Machine Learning with dedicated GPU compute