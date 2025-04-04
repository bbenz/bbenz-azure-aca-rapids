package com.azure.rapids.xgboost;

import com.azure.cosmos.ConsistencyLevel;
import com.azure.cosmos.CosmosClient;
import com.azure.cosmos.CosmosClientBuilder;
import com.azure.cosmos.CosmosContainer;
import com.azure.cosmos.models.CosmosItemRequestOptions;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import scala.collection.JavaConverters;

/**
 * Java implementation of XGBoost with NVIDIA RAPIDS on Azure Container Apps
 * with Cosmos DB vector storage - For Agaricus Mushroom Classification
 * 
 * This implementation supports both CPU and GPU modes for performance comparison
 */
public class XGBoostRapidsAzure {
    private static final Logger logger = LoggerFactory.getLogger(XGBoostRapidsAzure.class);

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        logger.info("Starting XGBoostRapidsAzure application with Agaricus mushroom dataset");

        // Parse command line arguments
        String dataSourcePath = getArgValue(args, "--data-source", "abfss://data-container@storageaccount.dfs.core.windows.net/agaricus_data.csv");
        String cosmosEndpoint = getArgValue(args, "--cosmos-endpoint", System.getenv("COSMOS_ENDPOINT"));
        String cosmosKey = getArgValue(args, "--cosmos-key", System.getenv("COSMOS_KEY"));
        String cosmosDatabase = getArgValue(args, "--cosmos-db", "VectorDB");
        String cosmosContainer = getArgValue(args, "--cosmos-container", "Vectors");
        String useGpuStr = getArgValue(args, "--use-gpu", System.getenv("USE_GPU"));
        boolean useGpu = Boolean.parseBoolean(useGpuStr == null ? "true" : useGpuStr);
        
        logger.info("Configuration: Data Source: {}", dataSourcePath);
        logger.info("Configuration: Use GPU: {}", useGpu);
        
        // Configure Spark session with RAPIDS if GPU is enabled
        SparkSession.Builder sparkBuilder = SparkSession.builder()
                .appName("XGBoostRapidsAzure")
                .master("local[*]"); // Will use all available CPU cores, remove for cluster deployment
        
        if (useGpu) {
            logger.info("Configuring with RAPIDS GPU acceleration");
            sparkBuilder = sparkBuilder
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
                .config("spark.rapids.sql.enabled", "true")
                .config("spark.rapids.sql.explain", "ALL")
                .config("spark.executor.resource.gpu.amount", "1")
                .config("spark.task.resource.gpu.amount", "1")
                .config("spark.executor.extraJavaOptions", "-Dai.rapids.cudf.prefer-pinned=true");
        } else {
            logger.info("Configuring for CPU-only mode (no RAPIDS acceleration)");
        }
        
        SparkSession spark = sparkBuilder.getOrCreate();

        try {
            // Load and prepare data
            long dataLoadStartTime = System.currentTimeMillis();
            Dataset<Row> data = loadData(spark, dataSourcePath);
            Dataset<Row> preparedData = prepareFeatures(data);
            long dataLoadEndTime = System.currentTimeMillis();
            logger.info("Data loading and preparation completed in {} ms", (dataLoadEndTime - dataLoadStartTime));
            
            // Train XGBoost model
            String accelerationMode = useGpu ? "with GPU acceleration" : "with CPU only";
            logger.info("Training XGBoost model {}", accelerationMode);
            long trainingStartTime = System.currentTimeMillis();
            XGBoostClassificationModel model = trainModel(preparedData, useGpu);
            long trainingEndTime = System.currentTimeMillis();
            logger.info("Model training completed in {} ms", (trainingEndTime - trainingStartTime));
            
            // Apply model to make predictions
            long predictionStartTime = System.currentTimeMillis();
            Dataset<Row> predictions = model.transform(preparedData);
            predictions.show(10);
            long predictionEndTime = System.currentTimeMillis();
            logger.info("Prediction completed in {} ms", (predictionEndTime - predictionStartTime));
            
            // Display classification metrics
            logger.info("Evaluating model performance");
            evaluateModel(predictions);
            
            // Save vectors to Cosmos DB if credentials are provided
            if (cosmosEndpoint != null && !cosmosEndpoint.isEmpty() && 
                cosmosKey != null && !cosmosKey.isEmpty()) {
                logger.info("Saving vector data to Cosmos DB");
                long cosmosStartTime = System.currentTimeMillis();
                saveToCosmosDB(predictions, cosmosEndpoint, cosmosKey, cosmosDatabase, cosmosContainer);
                long cosmosEndTime = System.currentTimeMillis();
                logger.info("Cosmos DB vector storage completed in {} ms", (cosmosEndTime - cosmosStartTime));
            } else {
                logger.info("Skipping Cosmos DB vector storage (no credentials provided)");
            }
            
            long endTime = System.currentTimeMillis();
            logger.info("XGBoost processing completed successfully in {} ms", (endTime - startTime));
            logger.info("Processing mode: {}", useGpu ? "GPU-accelerated" : "CPU-only");
        } catch (Exception e) {
            logger.error("Error in XGBoost processing", e);
        } finally {
            spark.stop();
        }
    }
    
    /**
     * Load data from specified source
     */
    private static Dataset<Row> loadData(SparkSession spark, String dataPath) {
        logger.info("Loading Agaricus mushroom data from: {}", dataPath);
        
        // Load data - Agaricus dataset typically has a header row
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataPath);
        
        logger.info("Data loaded with {} rows", df.count());
        logger.info("Data schema:");
        df.printSchema();
        df.show(5);
        
        return df;
    }
    
    /**
     * Prepare features for XGBoost model - specific for Agaricus mushroom dataset
     * This dataset has categorical features that need to be one-hot encoded
     */
    private static Dataset<Row> prepareFeatures(Dataset<Row> data) {
        logger.info("Preparing feature vectors for Agaricus mushroom dataset");
        
        // Identify all categorical columns from the dataset
        // The Agaricus dataset consists of categorical features
        List<String> categoricalColumns = new ArrayList<>();
        Arrays.stream(data.schema().fieldNames())
              .filter(col -> !col.equals("class")) // Exclude target variable
              .forEach(categoricalColumns::add);
        
        logger.info("Categorical columns: {}", categoricalColumns);
        
        // Ensure we have a label column - rename 'class' to 'label' and convert
        // In Agaricus dataset: 'e' = edible (0), 'p' = poisonous (1)
        Dataset<Row> labeledData = data.withColumn("label",
                data.col("class").cast(DataTypes.StringType).equalTo("p").cast(DataTypes.DoubleType));
        
        // Create a pipeline for feature engineering
        List<PipelineStage> pipelineStages = new ArrayList<>();
        
        // For each categorical column, create a StringIndexer and OneHotEncoder
        List<String> encodedColumns = new ArrayList<>();
        for (String categoricalCol : categoricalColumns) {
            String indexedCol = categoricalCol + "_indexed";
            String encodedCol = categoricalCol + "_encoded";
            
            // Convert string category to numeric index
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(categoricalCol)
                    .setOutputCol(indexedCol)
                    .setHandleInvalid("keep");
            
            // Convert numeric index to one-hot vector
            OneHotEncoder encoder = new OneHotEncoder()
                    .setInputCol(indexedCol)
                    .setOutputCol(encodedCol);
            
            pipelineStages.add(indexer);
            pipelineStages.add(encoder);
            encodedColumns.add(encodedCol);
        }
        
        // Create a VectorAssembler to combine all feature columns
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(encodedColumns.toArray(new String[0]))
                .setOutputCol("features");
        pipelineStages.add(assembler);
        
        // Build and apply the pipeline
        Pipeline pipeline = new Pipeline().setStages(pipelineStages.toArray(new PipelineStage[0]));
        Dataset<Row> preparedData = pipeline.fit(labeledData).transform(labeledData);
        
        // Select only the columns needed for training
        preparedData = preparedData.select("label", "features");
        logger.info("Feature engineering completed");
        
        return preparedData;
    }
    
    /**
     * Train XGBoost model with GPU acceleration if enabled
     */
    private static XGBoostClassificationModel trainModel(Dataset<Row> data, boolean useGpu) {
        // Configure XGBoost parameters
        Map<String, Object> params = new HashMap<>();
        params.put("eta", 0.1);
        params.put("max_depth", 8);
        params.put("objective", "binary:logistic");
        params.put("num_round", 100);
        
        // Configure GPU acceleration if enabled
        if (useGpu) {
            params.put("tree_method", "gpu_hist");  // GPU-accelerated training
            params.put("gpu_id", 0);
        } else {
            params.put("tree_method", "hist");  // CPU-only training
        }
        
        params.put("eval_metric", "auc");
        
        try {
            // Convert Java Map to Scala immutable Map 
            scala.collection.immutable.Map<String, Object> scalaParams = 
                JavaConverters.mapAsScalaMapConverter(params).asScala().toMap(
                    scala.Predef$.MODULE$.<scala.Tuple2<String, Object>>conforms()
                );
            
            // Create the XGBoost classifier with parameters
            XGBoostClassifier xgbClassifier = new XGBoostClassifier(scalaParams)
                    .setFeaturesCol("features")
                    .setLabelCol("label");
            
            // Explicitly set some parameters directly (belt-and-suspenders approach)
            xgbClassifier.setMaxDepth(Integer.parseInt(params.get("max_depth").toString()));
            xgbClassifier.setNumRound(Integer.parseInt(params.get("num_round").toString()));
            
            logger.info("Training with parameters: {}", params);
            
            // Fit the model and return the classification model (not the classifier)
            return xgbClassifier.fit(data);
            
        } catch (Exception e) {
            logger.error("Error creating or training XGBoost model: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to train XGBoost model", e);
        }
    }
    
    /**
     * Evaluate the model performance
     */
    private static void evaluateModel(Dataset<Row> predictions) {
        // Calculate accuracy
        long totalCount = predictions.count();
        long correctCount = predictions.filter(
                predictions.col("label").equalTo(predictions.col("prediction"))).count();
        double accuracy = (double) correctCount / totalCount;
        
        logger.info("Model Accuracy: {}", accuracy);
        
        // Show the confusion matrix
        predictions.groupBy("label", "prediction").count().show();
    }
    
    /**
     * Save model vectors to Cosmos DB with vector search capability
     */
    private static void saveToCosmosDB(Dataset<Row> predictions, 
                                      String endpoint, 
                                      String key, 
                                      String database, 
                                      String container) {
        // If credentials are not provided, log error and return
        if (endpoint == null || key == null) {
            logger.error("Cosmos DB credentials not provided. Skipping vector storage.");
            return;
        }
        
        // Process each partition of the predictions DataFrame
        predictions.foreachPartition(partition -> {
            // Create Cosmos DB client
            CosmosClient cosmosClient = new CosmosClientBuilder()
                    .endpoint(endpoint)
                    .key(key)
                    .consistencyLevel(ConsistencyLevel.EVENTUAL)
                    .buildClient();
            
            CosmosContainer cosmosContainer = cosmosClient.getDatabase(database)
                    .getContainer(container);
            
            // Process each row in this partition
            partition.forEachRemaining(row -> {
                try {
                    // Create document with ID, vector features, and prediction
                    VectorDocument doc = new VectorDocument();
                    doc.setId(UUID.randomUUID().toString());
                    
                    // Extract feature vector and convert to array
                    org.apache.spark.ml.linalg.Vector features = row.getAs("features");
                    double[] featureArray = features.toArray();
                    doc.setFeatures(featureArray);
                    
                    // Get prediction - convert to human-readable form
                    double prediction = row.getAs("prediction");
                    doc.setPrediction(prediction);
                    doc.setMushroomClass(prediction < 0.5 ? "edible" : "poisonous");
                    
                    // Add metadata
                    doc.setCreatedAt(System.currentTimeMillis());
                    doc.setDatasetType("agaricus");
                    
                    // Save to Cosmos DB
                    cosmosContainer.createItem(doc, new CosmosItemRequestOptions());
                } catch (Exception e) {
                    logger.error("Error saving vector to Cosmos DB", e);
                }
            });
            
            // Close the client
            cosmosClient.close();
        });
    }
    
    /**
     * Helper method to get argument values
     */
    private static String getArgValue(String[] args, String argName, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(argName)) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
    
    /**
     * Document class for Cosmos DB vector storage
     */
    static class VectorDocument {
        private String id;
        private double[] features;
        private double prediction;
        private String mushroomClass;
        private long createdAt;
        private String datasetType;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public double[] getFeatures() {
            return features;
        }

        public void setFeatures(double[] features) {
            this.features = features;
        }

        public double getPrediction() {
            return prediction;
        }

        public void setPrediction(double prediction) {
            this.prediction = prediction;
        }
        
        public String getMushroomClass() {
            return mushroomClass;
        }
        
        public void setMushroomClass(String mushroomClass) {
            this.mushroomClass = mushroomClass;
        }

        public long getCreatedAt() {
            return createdAt;
        }

        public void setCreatedAt(long createdAt) {
            this.createdAt = createdAt;
        }
        
        public String getDatasetType() {
            return datasetType;
        }
        
        public void setDatasetType(String datasetType) {
            this.datasetType = datasetType;
        }
    }
}