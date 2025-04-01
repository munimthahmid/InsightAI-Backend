# Common Issues and Troubleshooting

## API Connection Issues

### OpenAI API

- **Issue**: "Authentication Error" when making API calls to OpenAI
- **Solution**: Verify your OpenAI API key in the `.env` file and ensure it has sufficient credits

### Vector Database (Pinecone)

- **Issue**: "Failed to connect to Pinecone"
- **Solution**: Check your Pinecone API key and environment settings in the `.env` file

### News API

- **Issue**: "Exceeded request limit" error
- **Solution**: The free tier of News API has limited requests per day. Consider upgrading your plan or implementing caching

## Performance Issues

### Slow Research Response Times

- **Issue**: Research queries take a long time to complete
- **Solution**:
  - Reduce the `MAX_RESULTS_PER_SOURCE` setting in your environment variables
  - Check the LOG_LEVEL setting and set it to INFO or WARNING in production
  - Verify your network connection and API response times

### Memory Usage

- **Issue**: Application crashes due to memory constraints
- **Solution**:
  - Implement pagination for large research results
  - Optimize vector storage operations
  - Consider deploying to a higher-resource environment

## Deployment Issues

### Docker Container Fails to Start

- **Issue**: Docker container exits immediately after starting
- **Solution**:
  - Check logs with `docker logs <container_id>`
  - Verify all required environment variables are set
  - Ensure the database connection is properly configured

### CORS Errors

- **Issue**: Frontend cannot connect to backend due to CORS errors
- **Solution**:
  - Verify the `ALLOWED_HOSTS` setting in config.py
  - Ensure your frontend URL is included in the allowed origins

## Vector Database Issues

### Missing or Incorrect Embeddings

- **Issue**: Search results are irrelevant or missing
- **Solution**:
  - Check if the embeddings service is functioning correctly
  - Verify that documents are being properly indexed
  - Consider reindexing your vector database

### "Retrieved 0 results" Issue

- **Issue**: Logs show "Retrieved 0 results for query" despite documents being stored
- **Solution**:
  - This is typically caused by Pinecone's asynchronous nature where vectors aren't immediately queryable
  - The system now handles this automatically with:
    - Strategic delays between storage and querying operations
    - Backup storage in a shared research namespace
    - Multi-namespace querying that tries alternatives when the primary namespace fails
  - If the issue persists, you can:
    - Restart your research query after waiting 30+ seconds for Pinecone to fully update
    - Check Pinecone dashboard to verify index health
    - Review logs for any connection issues with Pinecone

### Index Creation Failures

- **Issue**: Cannot create or access the vector index
- **Solution**:
  - Verify Pinecone credentials and index name
  - Ensure the index dimensions match your embedding model (1536 for OpenAI embeddings)

## Research Quality Issues

### Poor Quality Research Results

- **Issue**: Research results are too generic or missing key information
- **Solution**:
  - Adjust research templates for your specific use case
  - Try using more specific queries
  - Consider implementing custom research workflows for specific domains

### Citation or Source Issues

- **Issue**: Missing or incorrect citations in research reports
- **Solution**:
  - Verify that all data sources are properly configured and accessible
  - Check for rate limiting on external APIs
  - Implement a citation verification step in the research process

### Missing URLs in References

- **Issue**: References showing "Source document" entries without URLs
- **Solution**:
  - Ensure the system is using the latest version with the URL extraction enhancements
  - Check that data sources are properly returning URL metadata
  - Verify that the `enhance_report_with_citations` method is correctly processing the URLs
  - Review logs to identify which source types may be missing URL information

### Citation Diversity

- **Issue**: Research reports over-relying on a single source for citations
- **Solution**:
  - Make sure you're using GPT-4o as the language model (set in `backend/app/services/research/report.py`)
  - Check that the prompt templates include instructions for diverse citation usage
  - Try increasing the number of results per source to provide more diverse material
  - Consider using research templates that encourage multi-perspective analysis
