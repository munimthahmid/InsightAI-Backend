"""
Critique agent for evaluating research reports.

This agent validates research reports for accuracy, completeness,
bias, and other quality factors. It provides feedback and suggestions
for improvement.
"""

from typing import Dict, List, Any, Optional
import json
import time
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import asyncio

from app.core.config import settings
from app.services.research.agents.base_agent import BaseAgent
from app.services.research.orchestration.schemas import TaskSchema


class CritiqueAgent(BaseAgent):
    """
    Evaluates research reports for quality, accuracy and completeness.
    Provides constructive feedback and suggestions for improvement.
    """

    def __init__(
        self,
        context_manager=None,
        task_queue=None,
        agent_id=None,
    ):
        """
        Initialize the critique agent.

        Args:
            context_manager: Shared research context manager
            task_queue: Task queue for asynchronous operations
            agent_id: Optional unique identifier
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="critique",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        # Register task handler
        if task_queue:
            task_queue.register_handler("critique_task", self.execute_task)

        # Initialize language model for evaluation
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o",  # Use GPT-4o for nuanced evaluation
            temperature=0.1,  # Low temperature for critical analysis
        )

        logger.info(f"CritiqueAgent initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """
        Execute a critique task to evaluate a research report.

        Args:
            task: The task containing the query and parameters

        Returns:
            Dictionary with critique, suggestions and quality metrics
        """
        params = task.params
        query = params.get("query", "")

        await self.log_activity(
            "start_critique",
            {"task_id": task.task_id, "query": query},
        )

        try:
            # Get research context
            research_id = await self.get_context("research_id")
            report = await self.get_context("report")
            raw_data = await self.get_context("raw_data") or {}
            analysis_results = await self.get_context("analysis_results") or {}

            if not report:
                logger.warning("No report found to critique")
                return {
                    "success": False,
                    "error": "No report found to critique",
                    "query": query,
                }

            # Update status
            await self.context_manager.update_status("critiquing_report")

            # 1. Check for factual accuracy by comparing with source data
            factual_assessment = await self._assess_factual_accuracy(
                report=report, raw_data=raw_data, query=query
            )

            # 2. Evaluate completeness and coverage
            completeness_assessment = await self._assess_completeness(
                report=report,
                query=query,
                analysis_results=analysis_results,
            )

            # 3. Check for bias and balance
            bias_assessment = await self._assess_bias(
                report=report,
                query=query,
            )

            # 4. Evaluate structure and readability
            readability_assessment = await self._assess_readability(
                report=report,
                query=query,
            )

            # 5. Generate improvement suggestions
            improvement_suggestions = await self._generate_improvements(
                report=report,
                query=query,
                factual_assessment=factual_assessment,
                completeness_assessment=completeness_assessment,
                bias_assessment=bias_assessment,
                readability_assessment=readability_assessment,
            )

            # 6. Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                factual_assessment=factual_assessment,
                completeness_assessment=completeness_assessment,
                bias_assessment=bias_assessment,
                readability_assessment=readability_assessment,
            )

            # 7. Generate final critique summary
            critique_summary = await self._generate_critique_summary(
                query=query,
                quality_metrics=quality_metrics,
                factual_assessment=factual_assessment,
                completeness_assessment=completeness_assessment,
                bias_assessment=bias_assessment,
                readability_assessment=readability_assessment,
                improvement_suggestions=improvement_suggestions,
            )

            # Store critique in context
            await self.set_context("critique", critique_summary)
            await self.set_context("quality_metrics", quality_metrics)
            await self.set_context("improvement_suggestions", improvement_suggestions)

            # Log completion
            await self.log_activity(
                "critique_complete",
                {
                    "task_id": task.task_id,
                    "query": query,
                    "overall_score": quality_metrics.get("overall_score", 0),
                },
            )

            # Return results
            return {
                "success": True,
                "critique": critique_summary,
                "quality_metrics": quality_metrics,
                "improvements": improvement_suggestions,
                "assessments": {
                    "factual": factual_assessment,
                    "completeness": completeness_assessment,
                    "bias": bias_assessment,
                    "readability": readability_assessment,
                },
                "query": query,
                "timestamp": time.time(),
            }

        except Exception as e:
            # Log failure
            logger.error(f"Error in critique task: {str(e)}")
            await self.log_activity(
                "critique_failed",
                {"task_id": task.task_id, "error": str(e)},
            )

            # Add error to context
            await self.context_manager.add_error(str(e), "critique_agent")

            # Re-raise for task queue error handling
            raise

    async def _assess_factual_accuracy(
        self,
        report: str,
        raw_data: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """
        Assess the factual accuracy of the report by comparing with source data.

        Args:
            report: The research report
            raw_data: Raw data from sources
            query: The research query

        Returns:
            Assessment results with scores and identified issues
        """
        # Format a sample of raw data for comparison
        source_samples = []

        # Take up to 2 documents from each source type
        for source_type, docs in raw_data.items():
            for i, doc in enumerate(docs[:2]):
                # Extract content
                content = ""
                metadata = doc.get("metadata", {})

                if "page_content" in doc:
                    content = doc["page_content"]
                elif "content" in doc:
                    content = doc["content"]
                elif "text" in doc:
                    content = doc["text"]
                elif "text" in metadata:
                    content = metadata["text"]

                # Format source
                source_sample = f"SOURCE {i+1} ({source_type.upper()}):\n"
                source_sample += f"{content[:500]}..."  # Truncate long content

                source_samples.append(source_sample)

        # Join source samples
        sources_text = "\n\n---\n\n".join(source_samples[:5])  # Limit to 5 samples

        # Create prompt for factual assessment with explicit JSON formatting
        prompt_template = """You are an expert research evaluator tasked with assessing the factual accuracy of a research report.

RESEARCH QUERY: {query}

REPORT TO EVALUATE:
{report}

SOURCE SAMPLES (for fact verification):
{sources_text}

Analyze the report for factual accuracy by checking if claims and statements are supported by the source materials.

Your evaluation should:
1. Identify any factual errors or unsupported claims
2. Note cases where the report contradicts source materials
3. Assess the overall factual reliability of the report
4. Provide specific examples of accurate and inaccurate information

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
    "factual_score": 7,
    "identified_errors": [
        {{
            "claim": "The problematic claim from the report",
            "issue": "Description of the factual problem",
            "severity": "high|medium|low"
        }}
    ],
    "strengths": [
        "Description of factual strengths"
    ],
    "overall_assessment": "Brief overall assessment of factual accuracy"
}}

IMPORTANT: Only include the JSON object in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON object that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling
        try:
            response = await chain.arun(
                query=query,
                report=report[:7000],  # Limit report size for token constraints
                sources_text=sources_text,
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            assessment = json.loads(cleaned_response)
            logger.info(
                f"Completed factual accuracy assessment with score: {assessment.get('factual_score', 0)}/10"
            )
            return assessment
        except Exception as e:
            logger.error(f"Error in factual assessment: {str(e)}")
            # Return minimal fallback assessment
            return {
                "factual_score": 5,
                "identified_errors": [],
                "strengths": ["Unable to complete detailed factual assessment"],
                "overall_assessment": "Factual assessment could not be completed fully",
            }

    async def _assess_completeness(
        self,
        report: str,
        query: str,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess the completeness and coverage of the report.

        Args:
            report: The research report
            query: The research query
            analysis_results: Results from analysis

        Returns:
            Assessment results with scores and identified gaps
        """
        # Extract key information that should be covered
        entities = analysis_results.get("entities", [])
        topics = analysis_results.get("topics", [])
        claims = analysis_results.get("claims", [])

        # Format expected content
        expected_content = {}

        if topics:
            expected_content["topics"] = [
                topic.get("topic_name", "") for topic in topics[:5]
            ]

        if entities:
            expected_content["key_entities"] = [
                entity.get("name", "") for entity in entities[:8]
            ]

        if claims:
            expected_content["key_claims"] = [
                claim.get("claim", "") for claim in claims[:5]
            ]

        # Create prompt for completeness assessment with explicit JSON formatting
        prompt_template = """You are an expert research evaluator tasked with assessing the completeness of a research report.

RESEARCH QUERY: {query}

REPORT TO EVALUATE:
{report}

EXPECTED CONTENT (key elements that should be addressed):
{expected_content}

Analyze the report for completeness and coverage by checking if it addresses all key aspects of the research query.

Your evaluation should:
1. Identify any missing topics or aspects that should be covered
2. Assess if the report has sufficient depth across all relevant areas
3. Evaluate if the report addresses different perspectives
4. Check if the conclusion adequately synthesizes findings

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
    "completeness_score": 6,
    "coverage_gaps": [
        {{
            "missing_element": "Description of what's missing",
            "importance": "high|medium|low"
        }}
    ],
    "adequately_covered": [
        "Description of well-covered aspects"
    ],
    "overall_assessment": "Brief overall assessment of completeness"
}}

IMPORTANT: Only include the JSON object in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON object that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling
        try:
            response = await chain.arun(
                query=query,
                report=report[:7000],  # Limit report size for token constraints
                expected_content=json.dumps(expected_content, indent=2),
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            assessment = json.loads(cleaned_response)
            logger.info(
                f"Completed completeness assessment with score: {assessment.get('completeness_score', 0)}/10"
            )
            return assessment
        except Exception as e:
            logger.error(f"Error in completeness assessment: {str(e)}")
            # Return minimal fallback assessment
            return {
                "completeness_score": 5,
                "coverage_gaps": [],
                "adequately_covered": [
                    "Unable to complete detailed completeness assessment"
                ],
                "overall_assessment": "Completeness assessment could not be completed fully",
            }

    async def _assess_bias(
        self,
        report: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Assess the bias and balance in the report.

        Args:
            report: The research report
            query: The research query

        Returns:
            Assessment results with bias scores and identified issues
        """
        # Create prompt for bias assessment with explicit JSON formatting
        prompt_template = """You are an expert research evaluator tasked with assessing bias and balance in a research report.

RESEARCH QUERY: {query}

REPORT TO EVALUATE:
{report}

Analyze the report for bias, balance, and objectivity.

Your evaluation should:
1. Identify any language that suggests bias (political, ideological, etc.)
2. Assess if multiple perspectives are presented fairly
3. Check if opinions are clearly distinguished from facts
4. Evaluate if the report uses balanced source selection

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
    "objectivity_score": 7,
    "identified_biases": [
        {{
            "bias_example": "The problematic text from the report",
            "bias_type": "Type of bias (political, methodological, etc.)",
            "severity": "high|medium|low"
        }}
    ],
    "balanced_elements": [
        "Description of well-balanced aspects"
    ],
    "overall_assessment": "Brief overall assessment of objectivity and balance"
}}

IMPORTANT: Only include the JSON object in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON object that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling
        try:
            response = await chain.arun(
                query=query,
                report=report[:7000],  # Limit report size for token constraints
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            assessment = json.loads(cleaned_response)
            logger.info(
                f"Completed bias assessment with score: {assessment.get('objectivity_score', 0)}/10"
            )
            return assessment
        except Exception as e:
            logger.error(f"Error in bias assessment: {str(e)}")
            # Return minimal fallback assessment
            return {
                "objectivity_score": 5,
                "identified_biases": [],
                "balanced_elements": ["Unable to complete detailed bias assessment"],
                "overall_assessment": "Bias assessment could not be completed fully",
            }

    async def _assess_readability(
        self,
        report: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Assess the readability and structure of the report.

        Args:
            report: The research report
            query: The research query

        Returns:
            Assessment results with readability scores and suggestions
        """
        # Create prompt for readability assessment with explicit JSON formatting
        prompt_template = """You are an expert research evaluator tasked with assessing the structure and readability of a research report.

RESEARCH QUERY: {query}

REPORT TO EVALUATE:
{report}

Analyze the report for structure, readability, and overall presentation.

Your evaluation should:
1. Assess the logical flow and organization of the report
2. Evaluate the clarity of writing and explanation
3. Check if technical terms are properly explained
4. Analyze the effectiveness of headings, lists, and formatting

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
    "readability_score": 7,
    "structural_issues": [
        {{
            "issue": "Description of the structural or readability issue",
            "location": "Where in the report this occurs",
            "severity": "high|medium|low"
        }}
    ],
    "strengths": [
        "Description of structural/readability strengths"
    ],
    "overall_assessment": "Brief overall assessment of structure and readability"
}}

IMPORTANT: Only include the JSON object in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON object that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling
        try:
            response = await chain.arun(
                query=query,
                report=report[:7000],  # Limit report size for token constraints
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            assessment = json.loads(cleaned_response)
            logger.info(
                f"Completed readability assessment with score: {assessment.get('readability_score', 0)}/10"
            )
            return assessment
        except Exception as e:
            logger.error(f"Error in readability assessment: {str(e)}")
            # Return minimal fallback assessment
            return {
                "readability_score": 5,
                "structural_issues": [],
                "strengths": ["Unable to complete detailed readability assessment"],
                "overall_assessment": "Readability assessment could not be completed fully",
            }

    async def _generate_improvements(
        self,
        report: str,
        query: str,
        factual_assessment: Dict[str, Any],
        completeness_assessment: Dict[str, Any],
        bias_assessment: Dict[str, Any],
        readability_assessment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate specific improvement suggestions based on assessments.

        Args:
            report: The research report
            query: The research query
            factual_assessment: Results of factual assessment
            completeness_assessment: Results of completeness assessment
            bias_assessment: Results of bias assessment
            readability_assessment: Results of readability assessment

        Returns:
            List of improvement suggestions with priority levels
        """
        # Compile the assessment results
        assessments = {
            "factual": factual_assessment,
            "completeness": completeness_assessment,
            "bias": bias_assessment,
            "readability": readability_assessment,
        }

        # Create prompt for improvements with explicit JSON formatting
        prompt_template = """You are an expert research editor tasked with providing actionable improvements for a research report.

RESEARCH QUERY: {query}

ASSESSMENTS OF THE REPORT:
{assessments}

Based on these assessments, provide specific, actionable suggestions to improve the report.

Your suggestions should:
1. Address the most critical issues identified in the assessments
2. Provide clear, specific instructions for improvement
3. Be prioritized by importance
4. Cover different aspects (factual accuracy, completeness, balance, structure)

FORMAT YOUR RESPONSE AS A VALID JSON LIST with the following structure:
[
    {{
        "improvement": "Clear description of the improvement needed",
        "rationale": "Why this improvement is important",
        "priority": "high|medium|low",
        "category": "factual|completeness|bias|readability"
    }},
    {{
        "improvement": "Second improvement suggestion",
        "rationale": "Why this improvement is important",
        "priority": "high|medium|low",
        "category": "factual|completeness|bias|readability"
    }}
]

Limit your response to 5-8 most important improvements.
IMPORTANT: Only include the JSON list in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON array that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling
        try:
            response = await chain.arun(
                query=query,
                assessments=json.dumps(assessments, indent=2),
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            improvements = json.loads(cleaned_response)
            logger.info(f"Generated {len(improvements)} improvement suggestions")
            return improvements
        except Exception as e:
            logger.error(f"Error generating improvements: {str(e)}")
            # Return minimal fallback improvements
            return [
                {
                    "improvement": "Review report for factual accuracy",
                    "rationale": "Ensuring factual accuracy is essential for research quality",
                    "priority": "high",
                    "category": "factual",
                },
                {
                    "improvement": "Check for completeness of coverage",
                    "rationale": "Comprehensive coverage is important for thorough research",
                    "priority": "medium",
                    "category": "completeness",
                },
            ]

    def _calculate_quality_metrics(
        self,
        factual_assessment: Dict[str, Any],
        completeness_assessment: Dict[str, Any],
        bias_assessment: Dict[str, Any],
        readability_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate overall quality metrics based on individual assessments.

        Args:
            factual_assessment: Results of factual assessment
            completeness_assessment: Results of completeness assessment
            bias_assessment: Results of bias assessment
            readability_assessment: Results of readability assessment

        Returns:
            Dictionary with quality metrics and overall score
        """
        # Extract scores from assessments
        factual_score = factual_assessment.get("factual_score", 5)
        completeness_score = completeness_assessment.get("completeness_score", 5)
        objectivity_score = bias_assessment.get("objectivity_score", 5)
        readability_score = readability_assessment.get("readability_score", 5)

        # Calculate weighted overall score
        # Factual accuracy weighted more heavily
        overall_score = (
            factual_score * 0.4
            + completeness_score * 0.3
            + objectivity_score * 0.15
            + readability_score * 0.15
        )

        # Round to one decimal place
        overall_score = round(overall_score, 1)

        # Determine quality level
        if overall_score >= 8.5:
            quality_level = "excellent"
        elif overall_score >= 7.0:
            quality_level = "good"
        elif overall_score >= 5.0:
            quality_level = "adequate"
        else:
            quality_level = "needs improvement"

        # Count issues by severity
        issues_count = {"high": 0, "medium": 0, "low": 0}

        # Count factual errors
        for error in factual_assessment.get("identified_errors", []):
            severity = error.get("severity", "medium")
            issues_count[severity] = issues_count.get(severity, 0) + 1

        # Count coverage gaps
        for gap in completeness_assessment.get("coverage_gaps", []):
            severity = gap.get("importance", "medium")
            issues_count[severity] = issues_count.get(severity, 0) + 1

        # Count bias issues
        for bias in bias_assessment.get("identified_biases", []):
            severity = bias.get("severity", "medium")
            issues_count[severity] = issues_count.get(severity, 0) + 1

        # Count structural issues
        for issue in readability_assessment.get("structural_issues", []):
            severity = issue.get("severity", "medium")
            issues_count[severity] = issues_count.get(severity, 0) + 1

        # Compile metrics
        metrics = {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "component_scores": {
                "factual_accuracy": factual_score,
                "completeness": completeness_score,
                "objectivity": objectivity_score,
                "readability": readability_score,
            },
            "issues_summary": issues_count,
            "timestamp": time.time(),
        }

        return metrics

    async def _generate_critique_summary(
        self,
        query: str,
        quality_metrics: Dict[str, Any],
        factual_assessment: Dict[str, Any],
        completeness_assessment: Dict[str, Any],
        bias_assessment: Dict[str, Any],
        readability_assessment: Dict[str, Any],
        improvement_suggestions: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a comprehensive critique summary based on all assessments.

        Args:
            query: The research query
            quality_metrics: Overall quality metrics
            factual_assessment: Results of factual assessment
            completeness_assessment: Results of completeness assessment
            bias_assessment: Results of bias assessment
            readability_assessment: Results of readability assessment
            improvement_suggestions: List of improvement suggestions

        Returns:
            Formatted critique summary as markdown text
        """
        # Extract key metrics
        overall_score = quality_metrics.get("overall_score", 0)
        quality_level = quality_metrics.get("quality_level", "adequate")

        # Create a summary prompt
        prompt_template = """You are an expert research editor providing a critique of a research report.

RESEARCH QUERY: {query}

QUALITY METRICS:
{quality_metrics}

FACTUAL ASSESSMENT:
{factual_assessment}

COMPLETENESS ASSESSMENT:
{completeness_assessment}

BIAS ASSESSMENT:
{bias_assessment}

READABILITY ASSESSMENT:
{readability_assessment}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

Based on these assessments, write a concise, constructive critique of the research report.

Your critique should:
1. Begin with an overall assessment of the report quality
2. Highlight the most significant strengths
3. Address the most critical areas for improvement
4. Provide specific examples when possible
5. End with actionable next steps for improving the report

Format your response as Markdown with appropriate headings and sections.
Be constructive and professional while also being honest about shortcomings.
Keep your critique focused on the 3-4 most important points in each category.
"""

        # Create the LLM chain with a more forgiving prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            # Use a try block with multiple attempts if needed
            max_retries = 3
            attempt = 0
            critique = None

            while attempt < max_retries and critique is None:
                try:
                    attempt += 1
                    response = await chain.arun(
                        query=query,
                        quality_metrics=json.dumps(quality_metrics, indent=2),
                        factual_assessment=json.dumps(factual_assessment, indent=2),
                        completeness_assessment=json.dumps(
                            completeness_assessment, indent=2
                        ),
                        bias_assessment=json.dumps(bias_assessment, indent=2),
                        readability_assessment=json.dumps(
                            readability_assessment, indent=2
                        ),
                        improvement_suggestions=json.dumps(
                            improvement_suggestions, indent=2
                        ),
                    )
                    critique = response.strip()

                    # Ensure it starts with a markdown heading
                    if not critique.startswith("# "):
                        critique = f"# Research Report Critique\n\n{critique}"

                    logger.info(
                        f"Generated critique summary for report with score: {overall_score}/10"
                    )

                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed: {str(e)}")
                    # Brief pause before retry
                    await asyncio.sleep(1)

            # If we have a critique, return it
            if critique:
                return critique

            # If all attempts failed, raise to be caught by outer try-except
            raise Exception("All retry attempts failed")

        except Exception as e:
            logger.error(f"Error generating critique summary: {str(e)}")
            # Generate minimal fallback critique
            return f"""# Research Report Critique

## Overall Assessment

This report on "{query}" has been assessed as **{quality_level}** with an overall score of **{overall_score}/10**.

## Key Strengths

- The report addresses the research query
- Basic structure is in place

## Areas for Improvement

- Review for factual accuracy and completeness
- Check for potential bias in presentation
- Improve structure and readability where needed

## Next Steps

We recommend reviewing the report focusing on the areas mentioned above to improve the overall quality.
"""
