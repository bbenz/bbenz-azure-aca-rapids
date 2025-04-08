@echo off
REM Set environment variables for XGBoost application
set COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
set COSMOS_KEY=your-cosmos-primary-key

REM Default values
set DATA_SOURCE=./scripts/tmp_data/agaricus_data.csv
set USE_GPU=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse_args
if "%~1"=="--data-source" (
    set DATA_SOURCE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--use-gpu" (
    set USE_GPU=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse_args

REM Set Java options for module access
set JAVA_OPTS=--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED

echo Running XGBoost RAPIDS application with Java 21 compatibility flags
echo Data source: %DATA_SOURCE%
echo GPU mode: %USE_GPU%

REM Select the appropriate JAR file based on GPU mode
if "%USE_GPU%"=="true" (
    echo Using GPU-optimized JAR file
    set JAR_FILE=target/xgboost-rapids-aca-gpu-1.0-SNAPSHOT.jar
) else (
    echo Using CPU-optimized JAR file
    set JAR_FILE=target/xgboost-rapids-aca-cpu-1.0-SNAPSHOT.jar
)

REM Run the Java application with parsed parameters
java %JAVA_OPTS% -jar %JAR_FILE% --data-source "%DATA_SOURCE%" --use-gpu "%USE_GPU%"
