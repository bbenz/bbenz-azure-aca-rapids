#!/bin/bash
# compare-cpu-gpu.sh - Script to run both CPU and GPU versions and compare performance

# Set Java 21 compatibility flags
JAVA_OPTS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED"

# Default data path - can be overridden with first argument
DATA_PATH=${1:-"./scripts/tmp_data/agaricus_data.csv"}
# Other args to pass to the application (e.g., Cosmos DB details)
shift 2>/dev/null || true
OTHER_ARGS="$@"

echo "===================================================="
echo "🔄 CPU vs GPU Performance Comparison - XGBoost Agaricus"
echo "===================================================="
echo "🔍 Data source: $DATA_PATH"
echo "🔍 Additional args: $OTHER_ARGS"
echo ""

# Create results directory if it doesn't exist
mkdir -p ./results

# Run CPU version
echo "🔄 Running CPU version..."
CPU_START_TIME=$(date +%s)

# Capture CPU logs
java $JAVA_OPTS -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
  --data-source "$DATA_PATH" \
  --use-gpu "false" \
  $OTHER_ARGS > ./results/cpu_output.log 2>&1

CPU_END_TIME=$(date +%s)
CPU_DURATION=$((CPU_END_TIME - CPU_START_TIME))

echo "✅ CPU run completed in $CPU_DURATION seconds"
echo ""

# Run GPU version
echo "🔄 Running GPU version..."
GPU_START_TIME=$(date +%s)

# Capture GPU logs
java $JAVA_OPTS -jar target/xgboost-rapids-aca-1.0-SNAPSHOT.jar \
  --data-source "$DATA_PATH" \
  --use-gpu "true" \
  $OTHER_ARGS > ./results/gpu_output.log 2>&1

GPU_END_TIME=$(date +%s)
GPU_DURATION=$((GPU_END_TIME - GPU_START_TIME))

echo "✅ GPU run completed in $GPU_DURATION seconds"
echo ""

# Calculate speedup
if [ $CPU_DURATION -gt 0 ]; then
  SPEEDUP=$(echo "scale=2; $CPU_DURATION / $GPU_DURATION" | bc)
else
  SPEEDUP="N/A"
fi

# Extract detailed timings from logs
echo "📊 Extracting performance metrics..."
CPU_DATA_LOAD=$(grep "Data loading and preparation completed in" ./results/cpu_output.log | awk '{print $NF}' | sed 's/ms//')
CPU_TRAINING=$(grep "Model training completed in" ./results/cpu_output.log | awk '{print $NF}' | sed 's/ms//')
CPU_PREDICTION=$(grep "Prediction completed in" ./results/cpu_output.log | awk '{print $NF}' | sed 's/ms//')
CPU_TOTAL=$(grep "XGBoost processing completed successfully in" ./results/cpu_output.log | awk '{print $NF}' | sed 's/ms//')

GPU_DATA_LOAD=$(grep "Data loading and preparation completed in" ./results/gpu_output.log | awk '{print $NF}' | sed 's/ms//')
GPU_TRAINING=$(grep "Model training completed in" ./results/gpu_output.log | awk '{print $NF}' | sed 's/ms//')
GPU_PREDICTION=$(grep "Prediction completed in" ./results/gpu_output.log | awk '{print $NF}' | sed 's/ms//')
GPU_TOTAL=$(grep "XGBoost processing completed successfully in" ./results/gpu_output.log | awk '{print $NF}' | sed 's/ms//')

# Create performance report
REPORT_FILE="./results/performance_report.md"
echo "# XGBoost Agaricus Performance Comparison: CPU vs GPU" > $REPORT_FILE
echo "" >> $REPORT_FILE
echo "## Test Information" >> $REPORT_FILE
echo "- **Date:** $(date)" >> $REPORT_FILE
echo "- **Data Source:** $DATA_PATH" >> $REPORT_FILE
echo "- **CPU Version:** Java 21, Spark 3.5.1" >> $REPORT_FILE
echo "- **GPU Version:** Java 21, Spark 3.5.1 with RAPIDS acceleration" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "## Overall Execution Time" >> $REPORT_FILE
echo "- **CPU Version:** $CPU_DURATION seconds" >> $REPORT_FILE
echo "- **GPU Version:** $GPU_DURATION seconds" >> $REPORT_FILE
echo "- **Speedup Factor:** $SPEEDUP x" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "## Detailed Performance Comparison" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "| Processing Stage | CPU Time (ms) | GPU Time (ms) | Speedup Factor |" >> $REPORT_FILE
echo "|------------------|---------------|---------------|----------------|" >> $REPORT_FILE

# Calculate speedups for each stage
if [ ! -z "$CPU_DATA_LOAD" ] && [ ! -z "$GPU_DATA_LOAD" ] && [ $GPU_DATA_LOAD -gt 0 ]; then
  DATA_LOAD_SPEEDUP=$(echo "scale=2; $CPU_DATA_LOAD / $GPU_DATA_LOAD" | bc)
else
  DATA_LOAD_SPEEDUP="N/A"
fi

if [ ! -z "$CPU_TRAINING" ] && [ ! -z "$GPU_TRAINING" ] && [ $GPU_TRAINING -gt 0 ]; then
  TRAINING_SPEEDUP=$(echo "scale=2; $CPU_TRAINING / $GPU_TRAINING" | bc)
else
  TRAINING_SPEEDUP="N/A"
fi

if [ ! -z "$CPU_PREDICTION" ] && [ ! -z "$GPU_PREDICTION" ] && [ $GPU_PREDICTION -gt 0 ]; then
  PREDICTION_SPEEDUP=$(echo "scale=2; $CPU_PREDICTION / $GPU_PREDICTION" | bc)
else
  PREDICTION_SPEEDUP="N/A"
fi

if [ ! -z "$CPU_TOTAL" ] && [ ! -z "$GPU_TOTAL" ] && [ $GPU_TOTAL -gt 0 ]; then
  TOTAL_SPEEDUP=$(echo "scale=2; $CPU_TOTAL / $GPU_TOTAL" | bc)
else
  TOTAL_SPEEDUP="N/A"
fi

echo "| Data Loading & Preparation | $CPU_DATA_LOAD | $GPU_DATA_LOAD | $DATA_LOAD_SPEEDUP |" >> $REPORT_FILE
echo "| Model Training | $CPU_TRAINING | $GPU_TRAINING | $TRAINING_SPEEDUP |" >> $REPORT_FILE
echo "| Prediction | $CPU_PREDICTION | $GPU_PREDICTION | $PREDICTION_SPEEDUP |" >> $REPORT_FILE
echo "| **Total Processing** | **$CPU_TOTAL** | **$GPU_TOTAL** | **$TOTAL_SPEEDUP** |" >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "## Accuracy Comparison" >> $REPORT_FILE
echo "" >> $REPORT_FILE
CPU_ACCURACY=$(grep "Model Accuracy:" ./results/cpu_output.log | awk '{print $NF}')
GPU_ACCURACY=$(grep "Model Accuracy:" ./results/gpu_output.log | awk '{print $NF}')

echo "- **CPU Model Accuracy:** $CPU_ACCURACY" >> $REPORT_FILE
echo "- **GPU Model Accuracy:** $GPU_ACCURACY" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "## Notes" >> $REPORT_FILE
echo "- The GPU version uses NVIDIA RAPIDS acceleration with the XGBoost 'gpu_hist' algorithm" >> $REPORT_FILE
echo "- The CPU version uses the standard XGBoost 'hist' algorithm" >> $REPORT_FILE
echo "- Both versions use identical parameters other than the acceleration mode" >> $REPORT_FILE

# Display summary
echo "📊 Performance Comparison Summary"
echo "=================================="
echo "CPU Total Time: $CPU_DURATION seconds"
echo "GPU Total Time: $GPU_DURATION seconds"
echo "Speedup Factor: $SPEEDUP x"
echo ""
echo "📝 Detailed performance report saved to: $REPORT_FILE"
echo "📝 Raw logs saved to: ./results/cpu_output.log and ./results/gpu_output.log"