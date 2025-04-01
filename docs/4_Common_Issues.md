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
  - The system now automatically generates embeddings if they're not found in the documents

### Missing Vector Embeddings Warning

- **Issue**: System logs show "No document vectors found, generating placeholder vectors for testing"
- **Solution**:
  - This is now handled automatically with the embedding generation enhancement
  - The AnalysisAgent will attempt to generate embeddings if they're not found in the documents
  - If embeddings still can't be generated, the system creates placeholders for testing purposes
  - No action required, but you can check the LOG_LEVEL to reduce warning messages

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

### JSON Parsing Errors in Analysis

- **Issue**: Analysis may fail with JSON parsing errors or invalid format specifiers in LLM responses
- **Solution**:
  - This is now fixed with enhanced JSON parsing and error handling in all analysis methods
  - The system properly escapes JSON format examples in prompts to prevent format string errors
  - LLM responses are now cleaned to handle markdown code blocks and other formatting issues
  - Each analysis component has fallback generation to ensure some result even if parsing fails
  - If the issue persists, check the logs for specific error messages and which component is failing

### Citation or Source Issues

- **Issue**: Missing or incorrect citations in research reports
- **Solution**:
  - Verify that all data sources are properly configured and accessible
  - Check for rate limiting on external APIs
  - Implement a citation verification step in the research process

### Missing URLs in References

- **Issue**: References showing "Source document" entries without URLs
- **Solution**:
  - This issue has been fixed with enhanced URL handling in the synthesis agent
  - The system now extracts URLs from multiple locations in document metadata
  - References now include proper markdown links for clickable URLs
  - "Document X" style references are automatically converted to numbered citations
  - The system adds additional metadata like authors and publication date when URLs aren't available
  - If URLs are still missing, verify that data source integrations are returning proper URL metadata

### Generic Document References

- **Issue**: Report mentions "Document 1", "Document 2" instead of using proper citation numbers
- **Solution**:
  - This issue has been fixed with improved citation handling
  - The synthesis agent now automatically converts "Document X" references to numbered citations
  - Prompts have been updated to guide the LLM to use proper citation format from the beginning
  - Various reference patterns are detected and converted, including "Documents X and Y" patterns
  - If the issue persists, try regenerating the report or check if the research templates need updating

### Citation Diversity

- **Issue**: Research reports over-relying on a single source for citations
- **Solution**:
  - Make sure you're using GPT-4o as the language model (set in `backend/app/services/research/report.py`)
  - Check that the prompt templates include instructions for diverse citation usage
  - Try increasing the number of results per source to provide more diverse material
  - Consider using research templates that encourage multi-perspective analysis
