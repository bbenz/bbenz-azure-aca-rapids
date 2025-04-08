package com.azure.rapids.xgboost;

import com.azure.cosmos.ConsistencyLevel;
import com.azure.cosmos.CosmosClient;
import com.azure.cosmos.CosmosClientBuilder;
import com.azure.cosmos.CosmosContainer;
import com.azure.cosmos.models.CosmosItemRequestOptions;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Helper class for saving data to Cosmos DB without serialization issues
 */
public class CosmosDbSaver {
    private static final Logger logger = LoggerFactory.getLogger(CosmosDbSaver.class);
    private static final int MAX_RETRIES = 3;
    private static final int BATCH_SIZE = 25;    /**
     * Save data to Cosmos DB without using Spark's distributed execution
     */    public static void savePredictionsToCosmosDb(Dataset<Row> predictions,
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

        // Test connection first
        boolean connectionSuccess = false;
        CosmosClient cosmosClient = null;
        
        try {
            // Create test connection
            logger.info("Testing Cosmos DB connection...");
            cosmosClient = new CosmosClientBuilder()
                    .endpoint(endpoint)
                    .key(key)
                    .consistencyLevel(ConsistencyLevel.EVENTUAL)
                    .buildClient();
            
            // Verify database and container exist
            cosmosClient.getDatabase(database).read();
            CosmosContainer cosmosContainer = cosmosClient.getDatabase(database).getContainer(container);
            cosmosContainer.read();
            
            // Check if data already exists in the container
            logger.info("Checking if data already exists in Cosmos DB...");
            boolean dataExists = checkIfDataExists(cosmosContainer, "agaricus");
            
            if (dataExists) {
                logger.info("Data already exists in Cosmos DB container. Skipping import and using local backup instead.");
                saveToLocalFile(predictions, "vector_data_already_exists_" + System.currentTimeMillis() + ".json");
                return;
            }
            
            // Create a test document
            VectorDocument testDoc = new VectorDocument();
            testDoc.setId("test-connection-" + UUID.randomUUID().toString());
            testDoc.setFeatures(new double[]{0.1, 0.2, 0.3});
            testDoc.setPrediction(0.0);
            testDoc.setMushroomClass("test");
            testDoc.setCreatedAt(System.currentTimeMillis());
            testDoc.setDatasetType("test-connection");
            
            logger.info("Creating test document with ID: {}", testDoc.getId());
            cosmosContainer.createItem(testDoc);
            logger.info("Test document successfully created. Connection to Cosmos DB is working.");
            
            connectionSuccess = true;
            
            // If connection successful, proceed with non-distributed processing to avoid serialization issues
            if (connectionSuccess) {
                // Convert dataset to local collection to avoid serialization issues with Spark
                logger.info("Collecting data for processing (non-distributed mode)...");
                
                // Instead of using Spark's distributed processing, collect all data locally
                List<Row> allRows = predictions.collectAsList();
                logger.info("Collected {} rows for Cosmos DB processing", allRows.size());
                
                // Process in single thread with the existing connection
                processBatches(cosmosClient, database, container, allRows);
            }
        } catch (Exception e) {
            logger.error("Failed to connect to or write to Cosmos DB: {}", e.getMessage());
            logger.error("Details: ", e);
            saveToLocalFile(predictions, "cosmos_fallback_" + System.currentTimeMillis() + ".json");
        } finally {
            if (cosmosClient != null) {
                cosmosClient.close();
                logger.info("Cosmos DB client closed");
            }        }
    }
    
    /**
     * Check if data already exists in the Cosmos DB container
     * 
     * @param container The Cosmos DB container to check
     * @param datasetType The type of dataset to check for (e.g., "agaricus")
     * @return true if data exists, false otherwise
     */    private static boolean checkIfDataExists(CosmosContainer container, String datasetType) {
        try {
            logger.info("Checking for existing {} data in Cosmos DB...", datasetType);
            
            // Create a SQL query to count documents with the specified dataset type
            // This is more efficient than retrieving all documents
            String query = String.format("SELECT VALUE COUNT(1) FROM c WHERE c.datasetType = '%s'", datasetType);
            
            // Create query options
            com.azure.cosmos.models.CosmosQueryRequestOptions options = new com.azure.cosmos.models.CosmosQueryRequestOptions();
            
            // Execute the query with the correct API signature
            int count = container.queryItems(
                query,
                options,
                Integer.class)
                .stream()
                .findFirst()
                .orElse(0);
            
            // Consider data exists if there are more than just test documents
            // We use 10 as a threshold since there might be a few test documents
            boolean exists = count > 10;
            
            if (exists) {
                logger.info("Found {} existing documents with datasetType '{}' in Cosmos DB", count, datasetType);
            } else {
                logger.info("No significant amount of data found in Cosmos DB (count: {})", count);
            }
            
            return exists;
        } catch (Exception e) {
            logger.error("Error checking if data exists: {}", e.getMessage());
            logger.error("Details: ", e);
            // If we can't check, assume no data exists to be safe
            return false;
        }
    }
    
    /**
     * Process data in batches
     */
    private static void processBatches(CosmosClient cosmosClient, String database, String container, List<Row> allRows) {
        try {
            CosmosContainer cosmosContainer = cosmosClient.getDatabase(database).getContainer(container);
            int totalRows = allRows.size();
            int successCount = 0;
            int errorCount = 0;
            
            // Process in batches
            for (int i = 0; i < totalRows; i += BATCH_SIZE) {
                // Get current batch
                List<Row> batch = allRows.subList(i, Math.min(i + BATCH_SIZE, totalRows));
                logger.info("Processing batch {}/{} with {} items", 
                        (i/BATCH_SIZE) + 1, 
                        (int)Math.ceil(totalRows/(double)BATCH_SIZE), 
                        batch.size());
                
                // Process the batch with retry logic
                int batchSuccessCount = 0;
                List<VectorDocument> batchFailures = new ArrayList<>();
                
                // Process each row in the batch
                for (Row row : batch) {
                    try {
                        // Create document with ID, vector features, and prediction
                        VectorDocument doc = createDocFromRow(row);
                        
                        // Save to Cosmos DB
                        CosmosItemRequestOptions options = new CosmosItemRequestOptions();
                        cosmosContainer.createItem(doc, options);
                        batchSuccessCount++;
                    } catch (Exception e) {
                        logger.error("Error saving document: {}", e.getMessage());
                        // Add to failures list
                        batchFailures.add(createDocFromRow(row));
                    }
                }
                
                // Update counts
                successCount += batchSuccessCount;
                errorCount += batchFailures.size();
                
                // Log progress
                double percentComplete = successCount * 100.0 / totalRows;
                logger.info("Batch complete: {} successes, {} failures, total progress: {}/{} documents ({:.1f}%)", 
                        batchSuccessCount, 
                        batchFailures.size(),
                        successCount,
                        totalRows,
                        percentComplete);
                
                // If there were failures, save them for later processing
                if (!batchFailures.isEmpty()) {
                    saveFailedItemsToFile(batchFailures);
                }
                
                // Add a small delay between batches to avoid overwhelming the service
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            
            logger.info("Batch processing complete. Saved {} documents, encountered {} errors", 
                    successCount, errorCount);
        } catch (Exception e) {
            logger.error("Error during batch processing: {}", e.getMessage());
            logger.error("Details: ", e);
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
