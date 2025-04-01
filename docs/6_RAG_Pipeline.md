# RAG Pipeline Architecture

## Overview

The Autonomous AI Research Agent implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline that enables comprehensive research across multiple data sources. This pipeline integrates vector embeddings, semantic search, and large language models to deliver accurate, contextually relevant research reports. The system is enhanced with a multi-agent orchestration system and vector clustering capabilities for improved information retrieval and synthesis.

## Pipeline Flow Diagram

```
                       ┌─────────────────┐
                       │   User Query    │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │ ControllerAgent │
                       └────────┬────────┘
                                │
                     ┌──────────┼──────────┐
                     │          │          │
            ┌────────▼────────┐ │ ┌────────▼────────┐
            │ AcquisitionAgent│ │ │   AnalysisAgent │
            └────────┬────────┘ │ └────────┬────────┘
                     │          │          │
                     │          │          │
         ┌───────────┴──────────┼──────────┴───────────┐
         │                      │                      │
┌────────▼─────┐    ┌───────────▼──────────┐    ┌─────▼──────────┐
│ Data Sources  │   │   Vector Clustering  │    │ SynthesisAgent │
└──────┬────────┘   └───────────┬──────────┘    └─────┬──────────┘
        │                       │                     │
        │                       │                     │
        │                  ┌────▼────────┐      ┌─────▼──────┐
        │                  │  CritiqueAgent│     │  Research  │
        │                  └────┬────────┘      │   Report   │
        │                       │               └────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐      ┌─────────────────┐
│   Documents     │      │  Validation &   │
│  & Metadata     │      │   Refinement    │
└───────┬─────────┘      └─────────────────┘
        │
        │
┌───────▼─────────────┐
│  Vector Storage     │
│  with Clustering    │
└───────┬─────────────┘
        │
┌───────▼─────────────┐
│   Semantic Search   │
│   with MMR          │
└───────────────────┘
```

## Pipeline Components

### 1. Data Ingestion Layer

The data ingestion layer, managed by the AcquisitionAgent, collects information from multiple sources including:

- **Academic Papers**: ArXiv and Semantic Scholar
- **News Articles**: News API
- **Code Repositories**: GitHub
- **Encyclopedia Articles**: Wikipedia
- **Web Search Results**: Integrated search functionality

Each data source is implemented as a dedicated module in `app/services/data_sources/sources.py` that inherits from a common `BaseDataSource` class, allowing for consistent API access patterns and easy extension.

### 2. Document Processing

Located in `app/services/vector_db/processors.py`, the document processing subsystem:

- Transforms raw API responses into structured Document objects
- Preserves source-specific metadata (URLs, authors, publication dates, etc.)
- Enhances documents with additional context for better retrieval
- Optimizes chunking strategies based on content type:
  - Academic papers: 1500 tokens with 200 token overlap
  - News articles: 1000 tokens with 100 token overlap
  - GitHub repositories: 800 tokens with 100 token overlap
  - Wikipedia articles: 1200 tokens with 150 token overlap

### 3. Vector Embedding & Storage with Clustering

The enhanced embedding system (`app/services/vector_db/vector_operations.py` and `app/services/vector_db/storage.py`) now includes:

- Conversion of text to vector embeddings using OpenAI's text-embedding-ada-002 model
- Efficient storage and indexing in Pinecone vector database with:
  - Optimized batching (50 vectors per batch)
  - Strategic delays between operations to account for Pinecone's asynchronous nature
  - A dual-namespace approach that stores a subset in a shared namespace for backup retrieval
- **Enhanced clustering capabilities**:
  - K-means clustering implementation (`app/services/vector_db/clustering/kmeans.py`)
  - HDBSCAN clustering for density-based clustering (`app/services/vector_db/clustering/hdbscan_cluster.py`)
  - Automatic parameter tuning based on dataset characteristics
  - Cluster quality assessment metrics
- **Maximum Marginal Relevance**:
  - Diverse document selection within clusters (`app/services/vector_db/clustering/mmr.py`)
  - Configurable diversity parameter
  - Balance between relevance and diversity
- Namespace management to isolate research sessions
- Multi-stage querying that attempts multiple namespaces to ensure results
- Fallback to a mock vector storage for development without Pinecone
- Robust error handling with verification of stored vectors
- Detailed logging for operational transparency
- Metadata preservation for document retrieval context

### 4. Semantic Search & Retrieval

When conducting research, the system:

- Embeds the research query using the same model
- Performs vector similarity search against stored documents
- **Applies cluster-aware retrieval**:
  - Retrieves larger initial document sets for clustering
  - Groups semantically similar documents into clusters
  - Selects diverse representatives from each relevant cluster
- Applies metadata filters to target specific sources if needed
- Returns the most semantically relevant documents for the query
- Preserves crucial metadata including original URLs for citation

### 5. Multi-Agent Orchestration System

The system now features a hierarchical multi-agent architecture (`app/services/research/agents/`) that replaces the legacy ResearchAgent with specialized agents:

#### ControllerAgent

- Orchestrates the entire multi-agent workflow
- Distributes tasks to specialized agents
- Manages the research context shared across agents
- Handles error recovery and agent coordination
- Implements Thompson sampling for agent selection
- Builds task dependency graphs with DAG structure

#### AcquisitionAgent

- Specializes in data gathering from multiple sources
- Interfaces with the DataSourceManager to collect information
- Processes and stores documents in the vector database
- Maintains document metadata including source URLs for citation

#### AnalysisAgent

- Specializes in analyzing collected information
- Performs document clustering and topic modeling
- Extracts key entities, claims, and relationships
- Identifies potential contradictions in research materials
- Generates a concept map showing relationships between entities

#### SynthesisAgent

- Specializes in generating comprehensive research reports
- Creates well-structured reports based on analysis results
- Supports template-based report generation
- Enhances citations with source information and URLs

#### CritiqueAgent

- Validates research findings and report quality
- Identifies potential inaccuracies or gaps
- Evaluates citation quality and source diversity
- Suggests improvements for report clarity and structure

### 6. Orchestration Tools

The multi-agent system is supported by specialized orchestration components:

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

### 7. Report Generation

The report generation system (`app/services/research/report.py`):

- Synthesizes information from retrieved documents
- Uses GPT-4o for comprehensive analysis and insight generation
- Formats reports based on optional templates
- Enhances citations with proper academic formatting
- Ensures URL references are preserved for verification

## Advanced Features

### Cluster-Aware Prompt Engineering

The system now includes specialized prompt engineering based on cluster characteristics:

- Templates for different cluster types (high-coherence vs. diverse)
- Automatic prompt engineering with parameter tuning
- Adaptation of prompts based on identified topic clusters

### Templated Research

The agent supports template-based research through the `TemplateManager`, allowing for:

- Domain-specific research frameworks
- Consistent research methodologies
- Specialized report structures

### Literature Review Generation

For academic purposes, the system can generate formal literature reviews that:

- Follow standard academic formatting (APA, MLA, etc.)
- Group sources thematically or chronologically
- Include proper citations and references

### Citation Enhancement

The report generator implements citation enhancement to:

- Group references by source type
- Format citations according to academic standards
- Include URLs for digital verification
- Preserve DOIs and other academic identifiers

## Performance Considerations

- **Asynchronous Processing**: All data fetching operations run asynchronously for improved throughput
- **Chunking Optimization**: Document chunking strategies are tailored to content types
- **Efficient Vector Storage**: Namespace isolation prevents vector database bloat
- **Mock Storage Fallback**: Development without Pinecone is supported through a mock implementation
- **Distributed Agent Processing**: Multi-agent system allows for parallel processing of research tasks
- **Optimized Clustering**: Automatic parameter tuning for clustering algorithms based on dataset characteristics

## Extension Points

The RAG pipeline is designed for extensibility:

1. **New Data Sources**: Add a new class in `app/services/data_sources/sources.py`
2. **Custom Templates**: Create specialized research templates
3. **Alternative Embedding Models**: The embedding service can be configured to use different models
4. **Additional Vector Databases**: The storage layer can be extended to support alternatives to Pinecone
5. **New Agent Types**: The multi-agent system can be extended with additional specialized agents
6. **Custom Clustering Algorithms**: New clustering approaches can be added to the clustering directory

## Technical Implementation Details

### Key Technologies

- **LangChain**: Provides document handling, text splitting, and embedding integration
- **OpenAI API**: Powers the embedding and generation components
- **Pinecone**: Vector database for efficient similarity search
- **FastAPI**: Asynchronous API framework for the backend
- **Pydantic**: Data validation and settings management
- **FAISS/HDBSCAN**: Vector clustering and similarity search
- **NetworkX**: Task dependency graph management for the multi-agent system

### Configuration

Key configuration parameters for the RAG pipeline can be found in `app/core/config.py`:

- Vector dimensions: 1536 (matching OpenAI's embedding model)
- Index settings: Serverless Pinecone configuration with cloud and region specs
- Chunk size and overlap settings by content type
- API configuration for various data sources
- Clustering parameters and thresholds
- Agent-specific configuration settings

### Error Handling & Resilience

The pipeline implements several resilience strategies:

- Fallback to mock vector storage when Pinecone is unavailable
- Graceful handling of missing documents
- Source-specific error handling for API failures
- Timeout management for external API calls
- Detailed logging for diagnosing issues
- Task retry mechanisms in the multi-agent orchestration system
- Centralized error registry in the ContextManager
