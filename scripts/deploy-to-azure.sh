#!/bin/bash
# Azure Container Apps deployment script for XGBoost RAPIDS application
# This script can deploy both CPU and GPU versions for performance comparison
# usage: 
# chmod +x ./deploy-to-azure.sh
# ./scripts/deploy-to-azure.sh [--cpu-only] [--gpu-only]

# Process command line arguments - HARD CODING
DEPLOY_CPU=true
DEPLOY_GPU=true

# 
# Check if the user provided any arguments


if [ $# -eq 0 ]; then
    echo "No arguments provided. Deploying both CPU and GPU versions."
else
    echo "Arguments provided: $@"
fi


if [ "$1" == "--cpu-only" ]; then
    DEPLOY_GPU=false
    echo "Deploying CPU version only"
elif [ "$1" == "--gpu-only" ]; then
    DEPLOY_CPU=false
    echo "Deploying GPU version only"
fi

# Set variables (adjust these as needed)
RESOURCE_GROUP="bbenz-rapids-xgboost"
LOCATION="swedencentral"  # Ensure this region supports GPU in ACA
# REGISTRY_NAME="cpurapidsxgboostreg$(date +%s | cut -c 5-10)"
CPU_REGISTRY_NAME="cpurapidsxgboostreg"
GPU_REGISTRY_NAME="gpurapidsxgboostreg"
COSMOS_DB_ACCOUNT="rapids-xgboost-cosmos"
COSMOS_DB_NAME="VectorDB"
COSMOS_CONTAINER_NAME="Vectors"
# STORAGE_ACCOUNT_NAME="rapidsxgboostdata$(date +%s | cut -c 5-10)"
STORAGE_ACCOUNT_NAME="rapidsxgboostdata"
STORAGE_CONTAINER_NAME="data"

# App variables for CPU and GPU versions
CPU_APP_NAME="agaricus-cpu-app"
GPU_APP_NAME="agaricus-gpu-app"
CPU_CONTAINER_IMAGE="$CPU_REGISTRY_NAME.azurecr.io/xgboost-rapids:latest"
# GPU_CONTAINER_IMAGE="$CPU_REGISTRY_NAME.azurecr.io/xgboost-rapids:latest" 
GPU_CONTAINER_IMAGE="mcr.microsoft.com/k8se/gpu-quickstart:latest"
GPU_WORKLOAD_PROFILE_NAME="NC8as-T4"
GPU_WORKLOAD_PROFILE_TYPE="Consumption-GPU-NC8as-T4"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color



for arg in "$@"
do
    case $arg in
        --cpu-only)
        DEPLOY_GPU=false
        shift
        ;;
        --gpu-only)
        DEPLOY_CPU=false
        shift
        ;;
    esac
done

echo -e "${GREEN}Starting deployment of XGBoost Agaricus example to Azure Container Apps...${NC}"
if [ "$DEPLOY_CPU" = true ] && [ "$DEPLOY_GPU" = true ]; then
    echo -e "${GREEN}Will deploy both CPU and GPU versions for performance comparison${NC}"
elif [ "$DEPLOY_CPU" = true ]; then
    echo -e "${GREEN}Will deploy CPU version only${NC}"
else
    echo -e "${GREEN}Will deploy GPU version only${NC}"
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null
then
    echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if user is logged in to Azure
echo -e "${YELLOW}Checking Azure login status...${NC}"
az account show &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}You are not logged into Azure. Please login:${NC}"
    az login
fi

# Create Resource Group if it doesn't exist
echo -e "${YELLOW}Checking if Resource Group exists...${NC}"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Resource Group $RESOURCE_GROUP already exists${NC}"
else
    echo -e "${YELLOW}Creating Resource Group...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
fi

# Create Azure Container Registry if it doesn't exist - CPU
if [ "$DEPLOY_CPU" = true ]; then
    echo -e "${YELLOW}Checking if CPU Container Registry exists...${NC}"
    if az acr show --name $CPU_REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        echo -e "${GREEN}Container Registry $CPU_REGISTRY_NAME already exists${NC}"
    else
        echo -e "${YELLOW}Creating CPU Azure Container Registry...${NC}"
        az acr create --resource-group $RESOURCE_GROUP --name $CPU_REGISTRY_NAME --sku Basic --admin-enabled true
    fi

CPU_ACR_USERNAME=$(az acr credential show --name $CPU_REGISTRY_NAME --query "username" -o tsv)
CPU_ACR_PASSWORD=$(az acr credential show --name $CPU_REGISTRY_NAME --query "passwords[0].value" -o tsv)

fi



# Create Azure Container Registry if it doesn't exist - GPU
if [ "$DEPLOY_GPU" = true ]; then
    echo -e "${YELLOW}Checking if GPU Container Registry exists...${NC}"
    if az acr show --name $GPU_REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        echo -e "${GREEN}Container Registry $GPU_REGISTRY_NAME already exists${NC}"
    else
        echo -e "${YELLOW}Creating GPU Azure Container Registry...${NC}"
        az acr create --resource-group $RESOURCE_GROUP --name $GPU_REGISTRY_NAME --sku Basic --admin-enabled true
    fi

# Get ACR credentials
GPU_ACR_USERNAME=$(az acr credential show --name $GPU_REGISTRY_NAME --query "username" -o tsv)
GPU_ACR_PASSWORD=$(az acr credential show --name $GPU_REGISTRY_NAME --query "passwords[0].value" -o tsv)
fi


# Create Cosmos DB account if it doesn't exist
echo -e "${YELLOW}Checking if Cosmos DB account exists...${NC}"
if az cosmosdb show --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Cosmos DB account $COSMOS_DB_ACCOUNT already exists${NC}"
else
    echo -e "${YELLOW}Creating Cosmos DB account...${NC}"
    az cosmosdb create --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP --kind GlobalDocumentDB --default-consistency-level Eventual
fi

# Create Cosmos DB database if it doesn't exist
echo -e "${YELLOW}Checking if Cosmos DB database exists...${NC}"
if az cosmosdb sql database show --name $COSMOS_DB_NAME --account-name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Cosmos DB database $COSMOS_DB_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating Cosmos DB database...${NC}"
    az cosmosdb sql database create --account-name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP --name $COSMOS_DB_NAME
fi

# Create container with vector index if it doesn't exist
echo -e "${YELLOW}Checking if Cosmos DB container exists...${NC}"
if az cosmosdb sql container show --name $COSMOS_CONTAINER_NAME --database-name $COSMOS_DB_NAME --account-name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Cosmos DB container $COSMOS_CONTAINER_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating Cosmos DB container with vector search capability...${NC}"
    az cosmosdb sql container create \
        --account-name $COSMOS_DB_ACCOUNT \
        --resource-group $RESOURCE_GROUP \
        --database-name $COSMOS_DB_NAME \
        --name $COSMOS_CONTAINER_NAME \
        --partition-key-path "/id" \
        --throughput 400
fi

# Get Cosmos DB connection strings
COSMOS_ENDPOINT=$(az cosmosdb show --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP --query "documentEndpoint" -o tsv)
COSMOS_KEY=$(az cosmosdb keys list --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP --query "primaryMasterKey" -o tsv)

# Create storage account for data files if it doesn't exist
echo -e "${YELLOW}Checking if Azure Storage account exists...${NC}"
if az storage account show --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Storage account $STORAGE_ACCOUNT_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating Azure Storage account for data...${NC}"
    az storage account create --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2
fi

# Get storage account key
STORAGE_KEY=$(az storage account keys list --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)

# Create storage container if it doesn't exist
echo -e "${YELLOW}Checking if Azure Storage container exists...${NC}"
if az storage container exists --name $STORAGE_CONTAINER_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --query "exists" -o tsv | grep -q "true"; then
    echo -e "${GREEN}Storage container $STORAGE_CONTAINER_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating Azure Storage container...${NC}"
    az storage container create --name $STORAGE_CONTAINER_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY
fi

# Download and upload the Agaricus mushroom dataset
echo -e "${YELLOW}Downloading and processing Agaricus mushroom dataset...${NC}"
mkdir -p ./tmp_data
wget -O ./tmp_data/agaricus-lepiota.data https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data

# Add header line to the CSV file and upload to Azure Storage
echo "class,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat" > ./tmp_data/agaricus_data.csv
cat ./tmp_data/agaricus-lepiota.data >> ./tmp_data/agaricus_data.csv

# Upload the dataset to Azure Storage
az storage blob upload --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY \
    --container-name $STORAGE_CONTAINER_NAME --name "agaricus_data.csv" \
    --file ./tmp_data/agaricus_data.csv --overwrite

echo -e "${GREEN}Agaricus dataset uploaded to Azure Storage${NC}"

# Build Java application with Maven
echo -e "${YELLOW}Building Java application with Maven...${NC}"
mvn clean package

# Build and push Docker images
if [ "$DEPLOY_CPU" = true ]; then
    echo -e "${YELLOW}Building and pushing CPU Docker image to ACR...${NC}"
    az acr build --registry $CPU_REGISTRY_NAME --image xgboost-rapids:cpu-latest --file Dockerfile.cpu .
fi

if [ "$DEPLOY_GPU" = true ]; then
    echo -e "${YELLOW}Building and pushing GPU Docker image to ACR...${NC}"
    az acr build --registry $GPU_REGISTRY_NAME --image xgboost-rapids:gpu-latest --file Dockerfile.gpu .
fi

# Create Container Apps environment if it doesn't exist - CPU
if [ "$DEPLOY_CPU" = true ]; then
    echo -e "${YELLOW}Checking if CPU Container Apps environment exists...${NC}"
    if az containerapp env show --name "xgboost-env-cpu" --resource-group $RESOURCE_GROUP &> /dev/null; then
        echo -e "${GREEN}CPU Container Apps environment already exists${NC}"
    else
        echo -e "${YELLOW}Creating CPU Container Apps environment...${NC}"
        az containerapp env create \
            --name "xgboost-env-cpu" \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION
    fi
fi

# Get the data URL for the Agaricus dataset
DATA_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${STORAGE_CONTAINER_NAME}/agaricus_data.csv"
# Create SAS token for blob access
END_DATE=$(date -u -d "1 year" '+%Y-%m-%dT%H:%MZ')
SAS_TOKEN=$(az storage blob generate-sas --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY \
    --container-name $STORAGE_CONTAINER_NAME --name "agaricus_data.csv" \
    --permissions r --expiry $END_DATE -o tsv)
DATA_URL_WITH_SAS="${DATA_URL}?${SAS_TOKEN}"

# Deploy CPU version if requested
if [ "$DEPLOY_CPU" = true ]; then
    echo -e "${YELLOW}Checking if CPU Container App exists...${NC}"
    if az containerapp show --name $CPU_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        echo -e "${GREEN}CPU Container App $CPU_APP_NAME already exists${NC}"
        CPU_APP_URL=$(az containerapp show --name $CPU_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
        echo -e "${GREEN}CPU version deployed at: $CPU_APP_URL${NC}"
    else
        echo -e "${YELLOW}Deploying CPU version to Container Apps...${NC}"
        az containerapp create \
            --name $CPU_APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --environment "xgboost-env-cpu" \
            --registry-server "$CPU_REGISTRY_NAME.azurecr.io" \
            --registry-username $CPU_ACR_USERNAME \
            --registry-password $CPU_ACR_PASSWORD \
            --image "$CPU_REGISTRY_NAME.azurecr.io/xgboost-rapids:cpu-latest" \
            --target-port 8080 \
            --ingress "external" \
            --cpu 4 \
            --memory 8Gi \
            --min-replicas 1 \
            --max-replicas 1 \
            --env-vars "COSMOS_ENDPOINT=$COSMOS_ENDPOINT" "COSMOS_KEY=$COSMOS_KEY" \
            "DATA_SOURCE=$DATA_URL_WITH_SAS" "USE_GPU=false"
        
        CPU_APP_URL=$(az containerapp show --name $CPU_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
        echo -e "${GREEN}CPU version deployed at: $CPU_APP_URL${NC}"
    fi
fi

# Deploy GPU version if requested
if [ "$DEPLOY_GPU" = true ]; then

# Create Container Apps environment for GPU if it doesn't exist
echo -e "${YELLOW}Checking if GPU Container Apps environment exists...${NC}"
if az containerapp env show --name "xgboost-env-gpu" --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}GPU Container Apps environment already exists${NC}"
else
    echo -e "${YELLOW}Creating GPU Container Apps environment...${NC}"
    az containerapp env create \
        --name "xgboost-env-gpu" \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION
fi

# Check if workload profile exists for GPU
echo -e "${YELLOW}Checking if GPU workload profile exists...${NC}"
if az containerapp env workload-profile list --name "xgboost-env-gpu" --resource-group $RESOURCE_GROUP --query "[?name=='$GPU_WORKLOAD_PROFILE_NAME'].name" -o tsv | grep -q "$GPU_WORKLOAD_PROFILE_NAME"; then
    echo -e "${GREEN}GPU workload profile $GPU_WORKLOAD_PROFILE_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating workload profile for GPU...${NC}"
    az containerapp env workload-profile add \
        --name "xgboost-env-gpu" \
        --resource-group $RESOURCE_GROUP \
        --workload-profile-name $GPU_WORKLOAD_PROFILE_NAME \
        --workload-profile-type $GPU_WORKLOAD_PROFILE_TYPE
fi

    echo -e "${YELLOW}Deploying GPU version to Container Apps...${NC}"
    az containerapp create \
        --name $GPU_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --environment "xgboost-env-gpu" \
        --registry-server "$GPU_REGISTRY_NAME.azurecr.io" \
        --registry-username $GPU_ACR_USERNAME \
        --registry-password $GPU_ACR_PASSWORD \
        --image "$GPU_REGISTRY_NAME.azurecr.io/xgboost-rapids:gpu-latest" \
        --target-port 8080 \
        --ingress "external" \
        --cpu 8.0 \
        --memory 56.0Gi \
        --workload-profile-name $GPU_WORKLOAD_PROFILE_NAME \
        --env-vars "COSMOS_ENDPOINT=$COSMOS_ENDPOINT" "COSMOS_KEY=$COSMOS_KEY" \
        "DATA_SOURCE=$DATA_URL_WITH_SAS" "USE_GPU=true"
    
    # Enable GPU support on the container app (this is a simplified representation)
    echo -e "${YELLOW}Note: For actual GPU support, you need to use az containerapp update with the appropriate --scale-rule-name and --scale-rule-type parameters for GPU SKUs${NC}"
    
    GPU_APP_URL=$(az containerapp show --name $GPU_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
    echo -e "${GREEN}GPU version deployed at: $GPU_APP_URL${NC}"
fi

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${GREEN}----------------------------------------------------${NC}"
echo -e "${GREEN}Resource Group:${NC} $RESOURCE_GROUP"
if [ "$DEPLOY_CPU" = true ]; then
    echo -e "${GREEN}CPU Container App:${NC} $CPU_APP_NAME"
fi
if [ "$DEPLOY_GPU" = true ]; then
    echo -e "${GREEN}GPU Container App:${NC} $GPU_APP_NAME"
fi
echo -e "${GREEN}Cosmos DB Endpoint:${NC} $COSMOS_ENDPOINT"
echo -e "${GREEN}Storage Account:${NC} $STORAGE_ACCOUNT_NAME"
echo -e "${GREEN}Agaricus Data URL:${NC} $DATA_URL"
echo -e "${GREEN}----------------------------------------------------${NC}"
echo -e "${YELLOW}Note: You can compare CPU vs GPU performance by examining the logs of each Container App.${NC}"
echo -e "${YELLOW}To view logs for CPU app: az containerapp logs show --name $CPU_APP_NAME --resource-group $RESOURCE_GROUP${NC}"
echo -e "${YELLOW}To view logs for GPU app: az containerapp logs show --name $GPU_APP_NAME --resource-group $RESOURCE_GROUP${NC}"
echo -e "${YELLOW}For production GPU support, you may need to update the Container App configuration to use a GPU-enabled compute SKU.${NC}"