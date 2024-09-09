# API Documentation

## Etherscan API Integration

The Blockchain Anomaly Detection project uses the Etherscan API to fetch Ethereum blockchain transaction data.

### Endpoints Used

- **GET**: `/api?module=account&action=txlist`
    - Fetches a list of normal transactions for a specific Ethereum address.

### Required Parameters

1. **API Key**: Required to authenticate API requests (`apikey`).
2. **Ethereum Address**: The Ethereum address for which to fetch transactions (`address`).
3. **Start Block**: The starting block number to fetch transactions (`startblock`).
4. **End Block**: The ending block number to fetch transactions (`endblock`).
5. **Sort**: Sort order (`asc` or `desc`).

### Error Handling

- **Timeout**: If a request times out, the project automatically retries up to 3 times.
- **Rate Limiting**: If the rate limit is exceeded, the system will pause and retry the request after a waiting period.

