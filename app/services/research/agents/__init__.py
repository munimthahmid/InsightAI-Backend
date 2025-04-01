"""
Specialized agent implementations for the multi-agent orchestration system.
Each agent focuses on a specific research function and coordinates through the controller.
"""

from app.services.research.agents.base_agent import BaseAgent
from app.services.research.agents.controller_agent import ControllerAgent
from app.services.research.agents.acquisition_agent import AcquisitionAgent
from app.services.research.agents.analysis_agent import AnalysisAgent
from app.services.research.agents.synthesis_agent import SynthesisAgent
from app.services.research.agents.critique_agent import CritiqueAgent

# These will be implemented later:
# from app.services.research.agents.analysis_agent import AnalysisAgent
# from app.services.research.agents.synthesis_agent import SynthesisAgent
# from app.services.research.agents.critique_agent import CritiqueAgent

__all__ = [
    "BaseAgent",
    "ControllerAgent",
    "AcquisitionAgent",
    "AnalysisAgent",
    "SynthesisAgent",
    "CritiqueAgent",
]
