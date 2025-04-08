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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

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
    
    // Special flag to help Spark with Java 17 compatibility
    static {
        logger.info("Setting up Java 17 compatibility settings");
        
        // Java 17 requires fewer workarounds than Java 21
        System.setProperty("io.netty.tryReflectionSetAccessible", "true");
        System.setProperty("spark.executor.allowSparkContext", "true");
        
        // XGBoost specific settings
        System.setProperty("xgboost.use.rmm", "false");
        System.setProperty("xgboost.rabit.tracker.disable", "true");
        System.setProperty("xgboost.force.local.training", "true");
    }
    
    /**
     * Initialize system properties to work around Java compatibility restrictions
     * Call this method at the very beginning of the application
     */
    private static void initializeJavaPlatformWorkaround() {
        // These system properties should be set before anything else
        System.setProperty("io.netty.tryReflectionSetAccessible", "true");
        
        // Needed for XGBoost
        System.setProperty("xgboost.use.rmm", "false");
        System.setProperty("xgboost.rabit.tracker.disable", "true");
        System.setProperty("xgboost.force.local.training", "true");
        
        logger.info("Java platform workarounds initialized");
    }
    
    public static void main(String[] args) {
        // Initialize Java platform workaround at the very beginning
        initializeJavaPlatformWorkaround();
        
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
        
        // Configure Spark session with Java 17 compatibility settings
        SparkSession.Builder sparkBuilder = SparkSession.builder()
                .appName("XGBoostRapidsAzure")
                .master("local[*]") // Will use all available CPU cores, remove for cluster deployment
                .config("spark.task.cpus", "1")
                .config("spark.executor.cores", "1")
                .config("spark.sql.execution.arrow.enabled", "true")
                .config("spark.databricks.xgboost.distributedMode", "false");

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
                logger.info("Cosmos DB endpoint: [{}]", cosmosEndpoint);
                logger.info("Cosmos DB key length: {} characters", cosmosKey != null ? cosmosKey.length() : 0);
                logger.info("Cosmos DB database: [{}]", cosmosDatabase);
                logger.info("Cosmos DB container: [{}]", cosmosContainer);
                
                long cosmosStartTime = System.currentTimeMillis();
                // Use CosmosDbSaver class instead of internal implementation
                CosmosDbSaver.savePredictionsToCosmosDb(predictions, cosmosEndpoint, cosmosKey, cosmosDatabase, cosmosContainer);
                long cosmosEndTime = System.currentTimeMillis();
                logger.info("Cosmos DB vector storage completed in {} ms", (cosmosEndTime - cosmosStartTime));
            } else {
                logger.error("Skipping Cosmos DB vector storage due to missing credentials");
                logger.error("Cosmos endpoint is {}null and {}empty", 
                          cosmosEndpoint == null ? "" : "NOT ", 
                          (cosmosEndpoint != null && cosmosEndpoint.isEmpty()) ? "" : "NOT ");
                logger.error("Cosmos key is {}null and {} characters long", 
                          cosmosKey == null ? "" : "NOT ",
                          cosmosKey != null ? cosmosKey.length() : 0);
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
        params.put("silent", 0);
        
        // Configure GPU acceleration if enabled
        if (useGpu) {
            params.put("tree_method", "gpu_hist");  // GPU-accelerated training
            params.put("gpu_id", 0);
        } else {
            params.put("tree_method", "hist");  // CPU-only training
        }
        
        // Disable distributed training completely
        params.put("use_external_memory", false);
        params.put("use_rmm", false);
        
        // Limit threads to avoid conflicts
        params.put("nthread", 1);
        
        // Disable distributed Rabit mode
        System.setProperty("xgboost.rabit.tracker.disable", "true");
        
        // Force training in single-instance mode
        params.put("force_local_training", "true");
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
                    .setLabelCol("label")
                    .setUseExternalMemory(false)
                    .setMissing(0.0f);
            
            // Explicitly set some parameters directly
            xgbClassifier.setMaxDepth(Integer.parseInt(params.get("max_depth").toString()));
            xgbClassifier.setNumRound(Integer.parseInt(params.get("num_round").toString()));
            
            logger.info("Training with parameters: {}", params);
            
            // Cache the data to avoid recomputation
            data.cache();
            
            // Fit the model and return the classification model
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
     * With robust error handling and local file fallback
     */
    private static void saveToCosmosDB(Dataset<Row> predictions, 
                                      String endpoint, 
                                      String key, 
                                      String database, 
                                      String container) {
        // If credentials are not provided, log error and use local file fallback
        if (endpoint == null || endpoint.isEmpty() || key == null || key.isEmpty()) {
            logger.error("Cosmos DB credentials not provided. Using local file fallback.");
            saveToLocalFile(predictions, "vector_data_backup.json");
            return;
        }
        
        // Clean up any potential whitespace in credentials
        endpoint = endpoint.trim();
        key = key.trim();
        
        logger.info("Cosmos DB saving started. Endpoint: {}, Database: {}, Container: {}", 
                endpoint, database, container);
        
        // Define max retry attempts for transient errors
        final int MAX_RETRIES = 3;
        boolean connectionSuccess = false;
        CosmosClient testClient = null;
        
        // First test the connection with retry logic before attempting batch save
        for (int retryCount = 0; retryCount < MAX_RETRIES && !connectionSuccess; retryCount++) {
            try {
                if (retryCount > 0) {
                    logger.info("Retrying Cosmos DB connection (attempt {} of {})", retryCount + 1, MAX_RETRIES);
                    // Exponential backoff between retries
                    Thread.sleep(1000 * (long)Math.pow(2, retryCount));
                }
                
                logger.info("Testing Cosmos DB connection...");
                testClient = new CosmosClientBuilder()
                        .endpoint(endpoint)
                        .key(key)
                        .consistencyLevel(ConsistencyLevel.EVENTUAL)
                        .buildClient();
                
                // Verify database exists
                testClient.getDatabase(database).read();
                logger.info("Successfully connected to Cosmos DB database: {}", database);
                
                // Verify container exists
                CosmosContainer testContainer = testClient.getDatabase(database).getContainer(container);
                testContainer.read();
                logger.info("Successfully connected to Cosmos DB container: {}", container);
                
                // Create a test document to verify write access
                VectorDocument testDoc = new VectorDocument();
                testDoc.setId("test-connection-" + UUID.randomUUID().toString());
                testDoc.setFeatures(new double[]{0.1, 0.2, 0.3});
                testDoc.setPrediction(0.0);
                testDoc.setMushroomClass("test");
                testDoc.setCreatedAt(System.currentTimeMillis());
                testDoc.setDatasetType("test-connection");
                
                logger.info("Creating test document with ID: {}", testDoc.getId());
                testContainer.createItem(testDoc);
                logger.info("Test document successfully created. Connection to Cosmos DB is working.");
                
                connectionSuccess = true;
            } catch (Exception e) {
                logger.error("Failed to connect to Cosmos DB on attempt {}: {}", retryCount + 1, e.getMessage());
                
                // Check for specific error conditions to provide better diagnostics
                if (e.getMessage().contains("Invalid API key")) {
                    logger.error("Authentication failure - check if your Cosmos DB key is correct.");
                } else if (e.getMessage().contains("Timed out")) {
                    logger.error("Connection timed out - check network and firewall settings.");
                } else if (e.getMessage().contains("Host not found")) {
                    logger.error("Endpoint not found - verify your Cosmos DB endpoint URL is correct.");
                }
                
                if (retryCount == MAX_RETRIES - 1) {
                    logger.error("Exhausted all retry attempts. Will attempt local file fallback.");
                    saveToLocalFile(predictions, "cosmos_fallback_" + System.currentTimeMillis() + ".json");
                }
            } finally {
                if (testClient != null && !connectionSuccess) {
                    testClient.close();
                    testClient = null;
                }
            }
        }
        
        // Skip batch operation if test connection failed after all retries
        if (!connectionSuccess) {
            logger.error("Unable to establish connection with Cosmos DB after {} attempts. Skipping batch processing.", MAX_RETRIES);
            return;
        }
        
        // Count how many items we're trying to save
        long itemCount = predictions.count();
        logger.info("Attempting to save {} items to Cosmos DB", itemCount);
        
        // Process each partition of the predictions DataFrame with batch mode for better efficiency
        try {
            // Make copies of these variables to make them effectively final for the lambda
            final String finalEndpoint = endpoint;
            final String finalKey = key;
            final String finalDatabase = database;
            final String finalContainer = container;
            final CosmosClient finalTestClient = testClient;
            
            predictions.foreachPartition(partition -> {
                // Count items in this partition for batching
                List<Row> partitionRows = new ArrayList<>();
                while (partition.hasNext()) {
                    partitionRows.add(partition.next());
                }
                
                logger.info("Processing partition with {} rows", partitionRows.size());
                
                // Initialize counters
                AtomicInteger successCount = new AtomicInteger(0);
                AtomicInteger errorCount = new AtomicInteger(0);
                AtomicInteger batchSize = new AtomicInteger(0);
                
                // Use client from test connection if possible to avoid reconnection
                CosmosClient cosmosClient = null;
                try {
                    // Reuse test client if available - we check if it's not null
                    // CosmosClient doesn't have an isClosed() method
                    if (finalTestClient != null) {
                        try {
                            // Test if the client is still valid by performing a simple operation
                            finalTestClient.readAllDatabases();
                            cosmosClient = finalTestClient;
                            logger.info("Reusing existing Cosmos DB connection");
                        } catch (Exception e) {
                            logger.warn("Previous client is no longer valid, creating a new one");
                            cosmosClient = null;
                        }
                    }
                    
                    if (cosmosClient == null) {
                        // Create new client if needed
                        cosmosClient = new CosmosClientBuilder()
                                .endpoint(finalEndpoint)
                                .key(finalKey)
                                .consistencyLevel(ConsistencyLevel.EVENTUAL)
                                .buildClient();
                        logger.info("Created new Cosmos DB connection");
                    }
                    
                    CosmosContainer cosmosContainer = cosmosClient.getDatabase(finalDatabase)
                            .getContainer(finalContainer);
                    
                    // Process in smaller batches for better reliability
                    final int BATCH_SIZE = 25;
                    List<List<Row>> batches = new ArrayList<>();
                    
                    // Create batches
                    for (int i = 0; i < partitionRows.size(); i += BATCH_SIZE) {
                        batches.add(partitionRows.subList(i, Math.min(i + BATCH_SIZE, partitionRows.size())));
                    }
                    
                    logger.info("Split partition into {} batches of approximately {} items each", 
                            batches.size(), BATCH_SIZE);
                    
                    for (List<Row> batch : batches) {
                        // Process each batch with retry logic
                        for (int retryAttempt = 0; retryAttempt < MAX_RETRIES; retryAttempt++) {
                            try {
                                if (retryAttempt > 0) {
                                    logger.info("Retrying batch (attempt {} of {})", retryAttempt + 1, MAX_RETRIES);
                                    Thread.sleep(1000 * (long)Math.pow(2, retryAttempt));
                                }
                                
                                int batchSuccessCount = 0;
                                List<VectorDocument> batchFailures = new ArrayList<>();
                                
                                // Process each row in this batch
                                for (Row row : batch) {
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
                                        
                                        // Save to Cosmos DB with timeout setting
                                        CosmosItemRequestOptions options = new CosmosItemRequestOptions();
                                        cosmosContainer.createItem(doc, options);
                                        batchSuccessCount++;
                                    } catch (Exception e) {
                                        // Add this row to the failures list for retry
                                        batchFailures.add(createDocFromRow(row));
                                    }
                                }
                                
                                // If all items in batch succeeded, we're done with this batch
                                if (batchFailures.isEmpty()) {
                                    successCount.addAndGet(batchSuccessCount);
                                    break; // Exit retry loop for this batch
                                } else if (retryAttempt == MAX_RETRIES - 1) {
                                    // Last retry attempt - add successful items to count
                                    successCount.addAndGet(batchSuccessCount);
                                    errorCount.addAndGet(batchFailures.size());
                                    
                                    // Log the failed items to a file for later processing
                                    saveFailedItemsToFile(batchFailures);
                                }
                            } catch (Exception e) {
                                logger.error("Error processing batch on attempt {}: {}", 
                                        retryAttempt + 1, e.getMessage());
                                
                                if (retryAttempt == MAX_RETRIES - 1) {
                                    // On final attempt, count whole batch as error
                                    errorCount.addAndGet(batch.size());
                                }
                            }
                        }
                        
                        // Log progress after each batch
                        batchSize.addAndGet(batch.size());
                        logger.info("Processed {} of {} documents ({}% complete, {} successes, {} errors)", 
                                batchSize.get(), partitionRows.size(),
                                String.format("%.1f", (batchSize.get() * 100.0 / partitionRows.size())),
                                successCount.get(), errorCount.get());
                    }
                    
                    logger.info("Partition processing complete. Saved {} documents, encountered {} errors", 
                            successCount.get(), errorCount.get());
                    
                } catch (Exception e) {
                    logger.error("Fatal error in partition processing for Cosmos DB: {}", e.getMessage());
                    logger.error("Details: ", e);
                } finally {
                    // Only close the client if we created it in this method
                    if (cosmosClient != null && cosmosClient != finalTestClient) {
                        cosmosClient.close();
                    }
                }
            });
            
            logger.info("All partitions processed for Cosmos DB storage");
        } catch (Exception e) {
            logger.error("Failed to execute Cosmos DB save operation: {}", e.getMessage(), e);
            logger.error("Falling back to local file storage");
            saveToLocalFile(predictions, "cosmos_error_fallback_" + System.currentTimeMillis() + ".json");
        } finally {
            // Close the test client if it's still open
            if (testClient != null) {
                testClient.close();
            }
        }
    }
    
    /**
     * Create a VectorDocument from a Spark Row
     */
    private static VectorDocument createDocFromRow(Row row) {
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
        
        return doc;
    }
    
    /**
     * Save failed items to a local file for later reprocessing
     */
    private static void saveFailedItemsToFile(List<VectorDocument> failedItems) {
        String filename = "cosmos_failed_items_" + System.currentTimeMillis() + ".json";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Using simple JSON serialization for the failed items
            for (VectorDocument doc : failedItems) {
                // Convert to simple JSON format
                String json = String.format(
                    "{\"id\":\"%s\",\"prediction\":%f,\"mushroomClass\":\"%s\",\"createdAt\":%d,\"datasetType\":\"%s\"}",
                    doc.getId(), doc.getPrediction(), doc.getMushroomClass(), doc.getCreatedAt(), doc.getDatasetType());
                writer.write(json);
                writer.newLine();
            }
            logger.info("Saved {} failed items to {}", failedItems.size(), filename);
        } catch (IOException e) {
            logger.error("Error saving failed items to file: {}", e.getMessage());
        }
    }
    
    /**
     * Fallback method to save predictions to a local file when Cosmos DB is unavailable
     */
    private static void saveToLocalFile(Dataset<Row> predictions, String filename) {
        logger.info("Saving predictions to local file: {}", filename);
        try {
            // Create a temporary directory if needed
            File backupDir = new File("cosmos_backups");
            if (!backupDir.exists()) {
                backupDir.mkdir();
            }
            
            // Full path for the backup file
            String fullPath = new File(backupDir, filename).getAbsolutePath();
            
            // Save as JSON file with timestamp in filename
            long saveCount = predictions.count();
            predictions.write().mode("overwrite").json(fullPath);
            
            logger.info("Successfully saved {} vector documents to {}", saveCount, fullPath);
            logger.info("To import this data to Cosmos DB later, use the Azure CLI or SDK.");
        } catch (Exception e) {
            logger.error("Error saving to local file: {}", e.getMessage());
        }
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