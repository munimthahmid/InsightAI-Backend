# Setup Instructions

This guide provides detailed instructions for setting up the Autonomous AI Research Agent on your local machine or in a production environment.

## Prerequisites

Before starting, ensure you have the following:

- **Python 3.9+** installed on your system
- **Node.js 16+** and **npm** for frontend development
- **Docker** and **Docker Compose** (optional, for containerized deployment)
- API Keys for the following services:
  - OpenAI API
  - Pinecone (or other vector database)
  - News API
  - GitHub API (optional, but recommended)

## Backend Setup

### Local Development Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/ai-research-agent.git
   cd ai-research-agent/backend
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment variables:**

   Create a `.env` file in the backend directory with the following contents:

   ```
   # API Keys
   OPENAI_API_KEY=your_openai_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   NEWS_API_KEY=your_news_api_key_here
   GITHUB_TOKEN=your_github_token_here

   # Vector DB settings
   PINECONE_INDEX=research-agent

   # Application settings
   DEBUG=True
   LOG_LEVEL=INFO
   MAX_RESULTS_PER_SOURCE=5

   # Scalability settings
   WORKER_CONCURRENCY=2
   REQUEST_TIMEOUT_SECONDS=300
   USE_CACHE=True
   CACHE_TTL_SECONDS=3600
   ENABLE_RATE_LIMITING=True

   # Notification system (optional)
   ENABLE_NOTIFICATIONS=False
   NOTIFICATION_PROVIDER=console  # 'email' or 'console'
   NOTIFICATION_EMAIL_FROM=research@example.com
   NOTIFICATION_SMTP_SERVER=smtp.example.com
   NOTIFICATION_SMTP_PORT=587
   NOTIFICATION_SMTP_USERNAME=username
   NOTIFICATION_SMTP_PASSWORD=password
   ```

5. **Start the backend server:**

   ```bash
   python -m app.main
   ```

   The API will be available at `http://localhost:8000`.

## Frontend Setup

1. **Navigate to the frontend directory:**

   ```bash
   cd ai-research-agent/frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Create environment variables:**

   Create a `.env` file in the frontend directory:

   ```
   VITE_API_URL=http://localhost:8000/api/v1
   ```

4. **Start the development server:**

   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`.

## Docker Deployment

For a containerized deployment:

1. **Ensure Docker and Docker Compose are installed**

2. **Configure environment variables in docker-compose.yml or in .env files**

3. **Build and start the containers:**

   ```bash
   docker-compose up -d
   ```

## Pinecone Setup

To properly configure the vector database:

1. **Create a Pinecone account** at [pinecone.io](https://www.pinecone.io/)

2. **Create a new index** with the following settings:

   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
   - Name: research-agent (or match the name in your .env file)

3. **Copy API key and environment** to your .env file

## Notification System Setup (Optional)

The system supports sending notifications about research results:

1. **Enable notifications** by setting `ENABLE_NOTIFICATIONS=True` in your .env file

2. **Configure notification provider**:

   - For development, use `NOTIFICATION_PROVIDER=console` to print notifications to the console
   - For production, use `NOTIFICATION_PROVIDER=email` and configure SMTP settings

3. **Email Configuration**:
   - Set `NOTIFICATION_EMAIL_FROM` to the sender email address
   - Configure SMTP settings (`NOTIFICATION_SMTP_SERVER`, `NOTIFICATION_SMTP_PORT`, etc.)
   - Ensure your SMTP server allows the application to send emails

## Verifying Installation

To verify your installation is working correctly:

1. **Check backend API docs:** Navigate to `http://localhost:8000/docs`
2. **Test basic research functionality:** Perform a simple research query via the frontend
3. **Check logs:** Monitor logs for any errors during setup and initial usage
4. **Test notifications:** Send a test notification via the `/notify` endpoint

## Troubleshooting

Common issues and their solutions:

- **API key errors:** Ensure all API keys are correctly provided in the .env file
- **Vector DB connection issues:** Check Pinecone status and network connectivity
- **CORS errors:** Make sure the backend allows requests from your frontend origin
- **Long response times:** Check rate limits of the underlying APIs and adjust MAX_RESULTS_PER_SOURCE
- **Notification failures:** Verify SMTP settings if using email notifications

## Production Considerations

For production deployments:

- Configure proper TLS/SSL for secure connections
- Set up authentication for the API
- Consider setting up monitoring and alerts
- Adjust resource allocations based on expected usage
- Set DEBUG=False in production environments
- Configure proper email settings for notifications
- Consider using Redis for caching in high-load scenarios
