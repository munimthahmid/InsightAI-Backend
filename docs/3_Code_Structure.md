# Code Structure and Architecture

This document provides an overview of the Autonomous AI Research Agent's codebase architecture, explaining the organization, key components, and design decisions.

## Overall Architecture

The project follows a modern client-server architecture:

```
ai-research-agent/
├── backend/              # FastAPI backend service
│   ├── app/              # Main application package
│   │   ├── api/          # API layer
│   │   ├── core/         # Core configurations
│   │   ├── services/     # Business logic
│   │   └── utils/        # Utility functions
│   ├── docs/             # Documentation
│   └── requirements.txt  # Python dependencies
├── frontend/             # React frontend
└── docker-compose.yml    # Container orchestration
```

## Backend Structure

The backend follows a layered architecture pattern with clear separation of concerns:

### API Layer (`app/api/`)

```
api/
├── models/                 # Pydantic data models
│   ├── __init__.py
│   ├── research.py         # Research-related models
│   └── notification.py     # Notification models
├── routes/                 # API routes
│   ├── __init__.py
│   ├── advanced.py         # Advanced research features
│   ├── history.py          # Research history management
│   ├── notification.py     # Notification endpoints
│   ├── research.py         # Core research functionality
│   └── templates.py        # Template management
│   └── helpers/            # Route helper functions
└── endpoints/              # Legacy API endpoints (deprecated)
```

Key responsibilities:

- Request/response validation
- HTTP handling
- Route definitions
- Input processing

### Core Layer (`app/core/`)

```
core/
├── __init__.py
└── config.py        # Application configuration
```

Key responsibilities:

- Environment configuration
- Global settings
- Constants
- Scalability configuration

### Services Layer (`app/services/`)

```
services/
├── __init__.py
├── data_sources/          # Data fetching from external APIs
│   ├── __init__.py
│   ├── base.py            # Abstract base class for data sources
│   ├── manager.py         # Orchestrates data collection from all sources
│   └── sources.py         # Implementations for specific data sources
├── research/              # Research module
│   ├── __init__.py        # Module initialization
│   ├── agent.py           # Legacy research orchestration agent
│   ├── agents/            # Multi-agent system for research
│   │   ├── __init__.py
│   │   ├── base_agent.py  # Abstract base agent class
│   │   ├── controller_agent.py  # Orchestrates all other agents
│   │   ├── acquisition_agent.py # Collects data from sources
│   │   ├── analysis_agent.py    # Analyzes collected information
│   │   ├── synthesis_agent.py   # Generates research reports
│   │   └── critique_agent.py    # Validates research findings
│   ├── orchestration/     # Multi-agent orchestration tools
│   │   ├── __init__.py
│   │   ├── task_queue.py  # Asynchronous task management
│   │   ├── schemas.py     # JSON schemas for inter-agent communication
│   │   └── context_manager.py # Shared research state
│   ├── history.py         # Research history management
│   └── _get_source_title.py # Helper for extracting source titles
├── templates/             # Template management
│   ├── __init__.py
│   ├── models.py          # Template data models
│   └── manager.py         # Template management functionality
├── vector_db/             # Vector database operations
│   ├── __init__.py
│   ├── storage.py         # Vector storage functionality
│   ├── vector_operations.py # Vector operations for embedding/retrieval
│   ├── document_preparation.py # Prepares documents for storage
│   ├── processors.py      # Document processing for vector storage
│   └── clustering/        # Vector clustering algorithms
│       ├── __init__.py
│       ├── kmeans.py      # K-means clustering implementation
│       ├── hdbscan_cluster.py # HDBSCAN clustering implementation
│       └── mmr.py         # Maximum Marginal Relevance
└── notification/          # Notification delivery system
    ├── __init__.py
    └── service.py         # Notification service
```

Key responsibilities:

- Core business logic
- Integration with external services
- Data processing
- Complex operations
- Notification handling

### Utils Layer (`app/utils/`)

```
utils/
└── __init__.py              # Utility functions
```

Key responsibilities:

- Helper functions
- Shared utilities
- Common tools

## Key Components

### Research Module (`app/services/research/`)

The central components for managing the research process:

#### ResearchAgent (`app/services/research/agent.py`)

- Orchestrates the entire research workflow
- Coordinates data collection from multiple sources
- Processes and stores data in vector database
- Generates research reports using templates
- Handles focused report generation and literature reviews
- Manages research status tracking

#### ResearchHistoryManager (`app/services/research/history.py`)

- Saves research results to persistent storage
- Retrieves past research sessions
- Manages deletion and searching of research history
- Handles research metadata

#### ReportGenerator (`app/services/research/report.py`)

- Generates comprehensive research reports from retrieved documents
- Supports template-based report generation
- Enhances reports with properly formatted citations
- Generates formal literature reviews with academic formatting

### Multi-Agent System (`app/services/research/agents/`)

An advanced hierarchical system of specialized agents that work collaboratively to process research tasks:

#### BaseAgent (`app/services/research/agents/base_agent.py`)

- Abstract base class for all agent types
- Implements common functionality for agent operations
- Provides context management and activity logging
- Handles serialization/deserialization of complex data structures

#### ControllerAgent (`app/services/research/agents/controller_agent.py`)

- Orchestrates the entire multi-agent workflow
- Distributes tasks to specialized agents
- Manages the research context shared across agents
- Handles error recovery and agent coordination
- Monitors research progress and updates status

#### AcquisitionAgent (`app/services/research/agents/acquisition_agent.py`)

- Specializes in data gathering from multiple sources
- Interfaces with the DataSourceManager to collect information
- Processes and stores documents in the vector database
- Maintains document metadata including source URLs for citation
- Handles chunking and initial processing of collected information

#### AnalysisAgent (`app/services/research/agents/analysis_agent.py`)

- Specializes in analyzing collected information
- Performs document clustering and topic modeling
- Extracts key entities, claims, and relationships
- Identifies potential contradictions in research materials
- Generates a concept map showing relationships between entities
- Features robust JSON parsing and error handling:
  - Properly escapes JSON format examples in LLM prompts
  - Cleans and validates LLM responses to handle various formats
  - Implements fallback generation when parsing fails
  - Automatically generates vector embeddings when they're missing

#### SynthesisAgent (`app/services/research/agents/synthesis_agent.py`)

- Specializes in generating comprehensive research reports
- Creates well-structured reports based on analysis results
- Supports template-based report generation
- Enhances citations with source information and URLs
- Implements improved reference handling:
  - Converts "Document X" references to proper citation numbers
  - Creates clickable markdown links for all source URLs
  - Formats references with complete metadata
  - Detects and replaces various document reference patterns
  - Groups references by source type for better organization

#### CritiqueAgent (`app/services/research/agents/critique_agent.py`)

- Validates research findings and report quality
- Identifies potential inaccuracies or gaps
- Evaluates citation quality and source diversity
- Suggests improvements for report clarity and structure
- Provides confidence scoring for research claims

### Orchestration Tools (`app/services/research/orchestration/`)

Components that enable the multi-agent system to function efficiently:

#### TaskQueue (`app/services/research/orchestration/task_queue.py`)

- Manages asynchronous task execution across agents
- Handles task dependencies and scheduling
- Provides task status tracking and prioritization
- Implements error handling and retry logic for failed tasks

#### ContextManager (`app/services/research/orchestration/context_manager.py`)

- Maintains shared state across the research process
- Stores intermediate results from different agents
- Provides a centralized error and warning registry
- Tracks overall research status and progress
- Ensures data consistency across the multi-agent system

#### Schemas (`app/services/research/orchestration/schemas.py`)

- Defines JSON schemas for inter-agent communication
- Validates message formats between agents
- Provides structured task definitions with parameters
- Ensures compatibility between agent inputs and outputs

### DataSources (`app/services/data_sources/`)

Responsible for fetching data from external APIs:

- Abstract base class for consistent source implementations
- DataSourceManager to coordinate all data sources
- Implementations for:
  - ArXiv API for academic papers
  - News API for news articles
  - GitHub API for repositories
  - Wikipedia API for general information
  - Semantic Scholar for academic research
- Error handling for API interactions

### VectorStorage (`app/services/vector_db/`)

Manages the vector database operations:

- Converts text to vector embeddings
- Stores and retrieves data from Pinecone
- Performs semantic search
- Document processors for handling different source types

### TemplateManager (`app/services/templates/`)

Handles research templates:

- Pydantic models for template structure
- Loads and saves templates
- Provides domain-specific templates
- Manages template retrieval and application
- Default templates for common research scenarios

### NotificationService (`app/services/notification/`)

Handles notifications about research results:

- Abstract provider interface for extensibility
- Email notification support
- Console notification for development
- Pluggable architecture for new providers

## Data Flow

1. **API Request**: User submits a research query via the API
2. **Data Collection**: ResearchAgent uses DataSourceManager to fetch relevant information from multiple sources
3. **Document Processing**: Data is processed by DocumentProcessor for optimal chunking and metadata extraction
4. **Vector Storage**: Processed documents are stored in VectorStorage
5. **Semantic Search**: Vector database is queried for the most relevant documents
6. **Report Generation**: ReportGenerator uses the retrieved documents to create a comprehensive report
7. **Citations Enhancement**: Reports are enhanced with proper citations linking to source documents
8. **Research History**: Results are saved to history for future reference
9. **Response/Notification**: Formatted response returned to the client and/or sent as notification

## Design Patterns

The codebase utilizes several design patterns:

- **Repository Pattern**: Data access layer abstracts database operations
- **Service Layer Pattern**: Business logic encapsulated in service classes
- **Dependency Injection**: Services and configurations passed as dependencies
- **Factory Pattern**: Creation of complex objects like templates
- **Facade Pattern**: Simplified interfaces to complex subsystems
- **Strategy Pattern**: Interchangeable notification providers
- **Adapter Pattern**: Common interface for different notification methods

## Modular Design Principles

The codebase adheres to these design principles:

1. **Single Responsibility Principle**: Each module has one job
2. **Dependency Inversion**: High-level modules don't depend on low-level modules
3. **Open/Closed Principle**: Open for extension, closed for modification
4. **DRY (Don't Repeat Yourself)**: Code reuse through utilities and shared functions
5. **KISS (Keep It Simple, Stupid)**: Clear, straightforward implementations

## Async Processing

The backend utilizes FastAPI's asynchronous capabilities:

- Concurrent data fetching from multiple sources
- Non-blocking I/O operations
- Efficient handling of multiple requests
- Background task processing for notifications
- Async workflow for research process

## Error Handling

A robust error handling strategy includes:

- Detailed logging at various levels
- Graceful degradation when services are unavailable
- Specific error responses with actionable information
- Fallback mechanisms for service failures
- Status tracking of ongoing research
- Enhanced error handling in the multi-agent system:
  - Agent-specific error handling with fallback generation
  - Robust JSON parsing with cleaning and validation
  - Context-aware error tracking with agent attribution
  - Agent isolation to prevent cascading failures
  - Automatic recovery mechanisms for common errors:
    - Vector embedding generation when vectors are missing
    - Format string error prevention in LLM prompts
    - Response cleanup for invalid JSON or markdown formatting
    - Default entity/topic/concept generation when parsing fails
  - Enhanced reference section handling to ensure complete citations

## Scalability Features

The system is designed with scalability in mind:

- Worker concurrency configuration
- Request timeout management
- Caching mechanism for improved performance
- Rate limiting for API stability
- Configurable notification system
