Вот пример файла **`troubleshooting.md`**, который поможет вашим пользователям решать распространённые проблемы с проектом:

---

# Troubleshooting Guide

This document aims to help users troubleshoot common issues they may encounter while using the **Blockchain Anomaly Detection** project.

## 1. Etherscan API Key Issues

**Problem**:  
The application is unable to fetch transaction data from the Etherscan API, and you receive an error about an invalid or missing API key.

**Solution**:  
Ensure that the Etherscan API key is correctly set in your environment variables. The API key is required to authenticate requests.

- On Linux/macOS:
  ```bash
  export ETHERSCAN_API_KEY=your_etherscan_api_key
  ```

- On Windows (Command Prompt):
  ```bash
  set ETHERSCAN_API_KEY=your_etherscan_api_key
  ```

- On Windows (PowerShell):
  ```bash
  $env:ETHERSCAN_API_KEY="your_etherscan_api_key"
  ```

Make sure the key is correctly added to the environment variables before running the application.

---

## 2. Etherscan Rate Limit Exceeded

**Problem**:  
The application returns a "rate limit exceeded" error, which stops it from fetching data.

**Solution**:  
The project has built-in rate limit handling. If this error persists:
- Check your usage limits on the [Etherscan API dashboard](https://etherscan.io/myapikey).
- If you are consistently hitting the limit, consider upgrading to a higher API tier or reduce the number of requests per second.
- Wait a few minutes, then re-run the application.

---

## 3. Missing Dependencies

**Problem**:  
You encounter errors related to missing dependencies when running the application.

**Solution**:  
Ensure that all required dependencies are installed by running the following command:

```bash
pip install -r requirements.txt
```

If the issue persists, make sure you're using the correct version of Python (3.9 or higher) and the correct virtual environment is activated.

---

## 4. Docker Container Fails to Start

**Problem**:  
The Docker container doesn't start, or it exits immediately after starting.

**Solution**:  
Check the container logs for more information:

```bash
docker logs <container_id>
```

Make sure the environment variables for the API key and address are correctly set in your `docker run` command:

```bash
docker run -e ETHERSCAN_API_KEY=your_api_key -e ETHERSCAN_ADDRESS=your_address -p 80:80 blockchain-anomaly-detection
```

Also, confirm that the `Dockerfile` is properly built by running:

```bash
docker build -t blockchain-anomaly-detection .
```

---

## 5. Incorrect Anomaly Detection Results

**Problem**:  
The anomaly detection results seem incorrect or not as expected.

**Solution**:  
- Verify the data input by checking if the raw transaction data is valid and clean.
- Review the contamination rate (percentage of anomalies) in the `IsolationForest` model. You can adjust the `contamination` parameter in `src/anomaly_detection/isolation_forest.py` to fine-tune the sensitivity of the model.

---

## 6. Test Failures

**Problem**:  
One or more tests fail during execution.

**Solution**:  
- Ensure that all dependencies are installed and the correct Python version is being used.
- Run the tests using the command below, ensuring you're in the project's root directory:
  
  ```bash
  pytest
  ```

- Check if the failing tests are due to missing or incorrect data inputs.

---

## 7. Visualization Errors

**Problem**:  
Visualizations (such as time-series plots or anomaly charts) are not displaying correctly.

**Solution**:  
- Ensure you have the required libraries installed, such as `matplotlib` and `seaborn`.
- If running the application on a headless server, use a non-interactive backend for `matplotlib`:
  
  ```python
  import matplotlib
  matplotlib.use('Agg')
  ```

This will allow the plots to be saved as files instead of being displayed interactively.

---

## Additional Help

If none of the solutions above resolves your issue, feel free to open an issue on the [GitHub repository](https://github.com/dkrizhanovskyi/blockchain-anomaly-detection/issues) with a detailed description of the problem.

---

Этот файл **`troubleshooting.md`** помогает пользователям решать наиболее распространённые проблемы, с которыми они могут столкнуться, от API проблем до неправильного обнаружения аномалий.