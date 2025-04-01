"""
Research history management for storing and retrieving past research.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from loguru import logger


class ResearchHistoryManager:
    """Manager for research history."""

    def __init__(self, history_file: str = "research_history.json"):
        """
        Initialize the research history manager.

        Args:
            history_file: Path to the history file
        """
        self.history_file = history_file

    def _make_json_serializable(self, obj, depth=0, seen=None):
        """
        Recursively convert an object to a JSON serializable format.

        Args:
            obj: The object to convert
            depth: Current recursion depth
            seen: Set of object ids already processed to prevent infinite recursion

        Returns:
            JSON serializable version of the object
        """
        # Handle recursion depth limit
        if depth > 10:  # Limit recursion depth to prevent stack overflow
            return str(type(obj))

        # Initialize seen set to track object references
        if seen is None:
            seen = set()

        # Handle None case first
        if obj is None:
            return None

        # Handle primitive types directly
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Check for circular references
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular reference to {type(obj).__name__}>"

        # Add this object to seen set
        seen.add(obj_id)

        # Handle different types
        if isinstance(obj, dict):
            return {
                k: self._make_json_serializable(v, depth + 1, seen)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item, depth + 1, seen) for item in obj]
        elif hasattr(obj, "tolist") and callable(obj.tolist):  # Handle numpy arrays
            return obj.tolist()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return self._make_json_serializable(obj.__dict__, depth + 1, seen)
        else:
            # Try to convert to a basic type
            try:
                return str(obj)
            except Exception:
                # Last resort fallback
                return str(type(obj))

    async def save_research(self, research_result: Dict[str, Any]) -> str:
        """
        Save research results to history for later retrieval.

        Args:
            research_result: The complete research result dictionary

        Returns:
            ID of the saved research
        """
        try:
            # Make sure the research result is JSON serializable
            serializable_result = self._make_json_serializable(research_result)

            # Generate a unique ID for this research if not already present
            research_id = serializable_result.get(
                "research_id",
                serializable_result.get("metadata", {}).get(
                    "session_id", str(uuid.uuid4())
                ),
            )

            # Add timestamp and research_id
            serializable_result["saved_at"] = datetime.now().isoformat()
            serializable_result["research_id"] = research_id

            # Load existing history
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = []

            # Add new research and save
            history.append(serializable_result)

            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(
                f"Saved research '{serializable_result.get('query', 'unknown')}' to history with ID: {research_id}"
            )
            return research_id

        except Exception as e:
            logger.error(f"Error saving research history: {str(e)}")
            raise

    async def get_all_research(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all research history items.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List of research history items (most recent first)
        """
        try:
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return []

            # Sort by saved_at (most recent first) and limit
            history.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
            return history[:limit]

        except Exception as e:
            logger.error(f"Error retrieving research history: {str(e)}")
            return []

    async def get_research_by_id(self, research_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific research by ID.

        Args:
            research_id: ID of the research to retrieve

        Returns:
            Research data if found, None otherwise
        """
        try:
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

            # Find research with matching ID
            for research in history:
                if research.get("research_id") == research_id:
                    return research

            return None

        except Exception as e:
            logger.error(f"Error retrieving research by ID: {str(e)}")
            return None

    async def delete_research_by_id(self, research_id: str) -> bool:
        """
        Delete a specific research by ID.

        Args:
            research_id: ID of the research to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return False

            # Filter out the research with matching ID
            original_length = len(history)
            history = [r for r in history if r.get("research_id") != research_id]

            if len(history) == original_length:
                # No research was removed
                return False

            # Save updated history
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(f"Deleted research with ID: {research_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting research by ID: {str(e)}")
            return False
