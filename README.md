# AI Research Agent Backend

## Overview

The backend system for the Autonomous AI Research Agent, a powerful AI-driven research assistant that collects, analyzes, and synthesizes information from multiple sources. This FastAPI-based service orchestrates the research process using modern NLP techniques and vector storage to deliver comprehensive research reports.

## Key Features

- **Multi-Source Research**: Collect data from ArXiv, News APIs, GitHub, Wikipedia, and Semantic Scholar
- **Advanced RAG Implementation**: Semantic search and contextual information retrieval
- **Research History Management**: Save and retrieve past research sessions
- **Template-Based Research**: Domain-specific templates for focused research
- **Literature Review Generation**: Create formal academic literature reviews
- **Citation Enhancement**: Properly formatted citations and references
- **Asynchronous Processing**: Non-blocking operations for efficient resource usage

## Technology Stack

- **Framework**: FastAPI
- **Language Model**: OpenAI GPT-4o
- **Vector Database**: Pinecone
- **Async Processing**: Python asyncio
- **Documentation**: Pydantic models with auto-generated OpenAPI docs

## Project Structure

```
backend/
├── app/                     # Main application package
│   ├── api/                 # API layer
│   │   ├── models/          # Pydantic data models
│   │   └── routes/          # API endpoints
│   ├── core/                # Core configurations
│   └── services/            # Business logic
│       ├── data_sources/    # External data source integrations
│       ├── research/        # Research orchestration module
│       ├── templates/       # Research template management
│       ├── vector_db/       # Vector database operations
│       └── notification/    # Notification delivery system
├── docs/                    # Documentation
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/)
- [Pinecone API key](https://www.pinecone.io/)
- [News API key](https://newsapi.org/)
- [GitHub API token](https://github.com/settings/tokens)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-research-agent.git
   cd ai-research-agent/backend
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   NEWS_API_KEY=your_news_api_key
   GITHUB_TOKEN=your_github_token
   ```

### Running the Server

Start the development server:

```bash
python run.py
```

The API will be available at `http://localhost:8000`.

API documentation is automatically generated and available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Research Endpoints

- `POST /api/v1/research` - Conduct research on a topic
- `POST /api/v1/research/template` - Research using a specific template
- `GET /api/v1/research/history` - Get research history
- `GET /api/v1/research/history/{id}` - Get specific research
- `POST /api/v1/research/compare` - Compare multiple research topics
- `POST /api/v1/research/literature-review` - Generate literature review

### Templates Endpoints

- `GET /api/v1/templates` - Get all templates
- `GET /api/v1/templates/{id}` - Get template by ID
- `POST /api/v1/templates` - Create new template
- `PUT /api/v1/templates/{id}` - Update template
- `DELETE /api/v1/templates/{id}` - Delete template

## Development

### Code Structure

The backend follows a modular architecture with clear separation of concerns:

1. **API Layer**: Handles HTTP requests, validation, and response formatting
2. **Services Layer**: Contains core business logic and external integrations
3. **Core Layer**: Application configuration and global settings

### Research Module

The research module is the heart of the system, consisting of:

- **ResearchAgent**: Orchestrates the research process
- **ResearchHistoryManager**: Manages storage and retrieval of research
- **ReportGenerator**: Creates well-structured reports with enhanced citations and URL references
- **Document Processor**: Ensures proper metadata preservation, including URLs for citations
- **Citation Enhancement**: Groups references by source type and ensures proper URL inclusion

### Adding New Data Sources

To add a new data source:

1. Create a new class in `app/services/data_sources/sources.py` that inherits from `BaseDataSource`
2. Implement the required methods (`fetch_data`, etc.)
3. Register the source in the `DataSourceManager`

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

1. [Project Overview](docs/1_Project_Overview.md)
2. [Setup Instructions](docs/2_Setup_Instructions.md)
3. [Code Structure](docs/3_Code_Structure.md)
4. [Common Issues](docs/4_Common_Issues.md)
5. [Changelog](docs/5_Changelog.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
