# Changelog

All notable changes to the Autonomous AI Research Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Implemented Multi-agent Orchestration System**:

  - Created `BaseAgent` abstract class for all agent types
  - Implemented `ControllerAgent` to coordinate the research workflow
  - Developed specialized `AcquisitionAgent` for data gathering
  - Built `AnalysisAgent` for information processing and clustering
  - Created `SynthesisAgent` for report generation
  - Implemented `CritiqueAgent` for validation and quality control
  - Added orchestration tools for inter-agent communication:
    - Asynchronous `TaskQueue` for task management
    - `ContextManager` for shared research state
    - JSON schema validation for agent communication
    - Task dependency graphs using DAG structure
    - Thompson sampling for optimal agent selection

- **Enhanced RAG with Vector Clustering**:

  - Implemented K-means clustering for document organization
  - Added HDBSCAN clustering for density-based document grouping
  - Implemented Maximum Marginal Relevance for diverse document selection
  - Added automatic parameter tuning for clustering algorithms
  - Created cluster quality assessment metrics
  - Developed cluster-aware prompt engineering
  - Implemented templates for different cluster types
  - Enhanced retrieval with cluster-based document selection

- Implemented modular Research module with dedicated components:
  - ResearchAgent for orchestrating the entire research process
  - ResearchHistoryManager for storing and retrieving research history
  - ReportGenerator for creating comprehensive research reports
- Added support for enhanced citations with proper formatting and references section
- Implemented formal literature review generation with multiple citation styles
- Added focused report generation for deeper analysis of specific aspects
- Introduced research status tracking for monitoring ongoing research
- Created template-based report generation system with customizable structures
- Implemented robust vector database operations with timing controls:
  - Added strategic delays to account for Pinecone's asynchronous nature
  - Created a shared namespace backup system for reliable retrieval
  - Implemented multi-stage querying across multiple namespaces
  - Added enhanced verification and logging for vector operations
- Implemented automatic vector embedding generation when vectors are missing from documents
- Added robust JSON parsing and cleaning in all analysis methods with fallback generation

### Changed

- **Replaced single-agent architecture with multi-agent orchestration system**:

  - Refactored `ResearchAgent` class to use the new orchestration
  - Implemented specialized prompt templates for each agent type
  - Added logging and monitoring for agent performance
  - Created progress tracking for multi-agent operations

- **Enhanced vector operations with clustering capabilities**:

  - Modified `VectorOperations` class to support larger retrieval sets
  - Added clustering preparation methods
  - Integrated clustering into the main research process
  - Adapted report generation to utilize cluster information
  - Implemented cluster-based citation grouping

- Upgraded the language model from GPT-3.5-Turbo to GPT-4o for higher quality research reports
- Enhanced the URL extraction and preservation in references for better citation quality
- Improved context formatting to present document sources more clearly to the language model
- Enhanced prompt templates to encourage more diverse citation usage across all sources
- Modified the references section format to group by source type for better organization
- Added enhanced debugging logs to track source data and URL availability
- Updated all LLM prompts to explicitly request standardized numbered citation format
- Enhanced JSON prompt templates with proper format string escaping to prevent errors

### Fixed

- Fixed issue with URLs not appearing in the References section even when they exist in source data
- Fixed problem with report generation over-relying on a single source for citations
- Improved document processor to better preserve URL information when chunking documents
- Fixed JSON parsing errors in analysis agent by properly escaping JSON examples in prompts
- Fixed format string interpretation issues in LLM prompts
- Fixed "Document X" style references by automatically converting them to numbered citations
- Added pattern matching to properly replace various document reference patterns with citation numbers
- Improved error handling across all LLM interactions with proper JSON response cleaning
- Fixed reference section formatting to include clickable markdown links for all URLs
- Improved URL extraction by checking multiple locations in document metadata

## [2.0.0] - 2023-TBD

### Added

- Enhanced scalability architecture with improved service isolation
- New modular data source plugin system
- Configurable research workflows
- Advanced caching mechanism for improved performance
- Comprehensive error handling and recovery
- Detailed logging for all operations
- Support for multiple vector database providers

### Changed

- Restructured backend for better scalability and maintainability
- Improved research agent logic with better context handling
- Enhanced vector storage service with connection pooling
- Optimized embedding generation process
- Updated API endpoints for better resource management

### Removed

- Slack integration (replaced with more generic notification system)
- Legacy vector storage implementation
- Deprecated research templates

## [1.0.0] - 2023-03-28

### Added

- Initial release of the Autonomous AI Research Agent
- Multi-source data integration (Academic papers, News, GitHub, Wikipedia)
- Advanced RAG implementation with vector-based knowledge management
- Research specialization with domain-specific templates
- FastAPI backend with asynchronous processing
- Vector database integration with Pinecone
- React-based frontend with Chakra UI
- Slack integration for research delivery
