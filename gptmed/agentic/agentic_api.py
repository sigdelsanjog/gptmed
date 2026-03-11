"""Backend adapter for gptmed agentic workflow.

This module is designed for local backend integration and testing.
It intentionally imports the installed ``gptmed`` package API path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import importlib
import os


class BackendAgenticService:
    """Lazy wrapper around ``gptmed.agentic.main.MedicalPrescriptionWorkflow``."""

    def __init__(self) -> None:
        self._workflow = None
        self._error: Optional[str] = None
        self._source: Optional[str] = None

        # Keep these configurable for package version differences.
        self._workflow_module = os.getenv("GPTMED_AGENTIC_WORKFLOW_MODULE", "gptmed.agentic.main")
        self._workflow_class_name = os.getenv(
            "GPTMED_AGENTIC_WORKFLOW_CLASS", "MedicalPrescriptionWorkflow"
        )

    def _import_workflow_class(self):
        """Import workflow class from installed gptmed package only."""
        module = importlib.import_module(self._workflow_module)
        workflow_class = getattr(module, self._workflow_class_name)
        self._source = "installed-package"
        return workflow_class

    def _ensure_workflow(self) -> None:
        """Initialize and cache workflow instance on first use."""
        if self._workflow is not None or self._error is not None:
            return

        try:
            workflow_class = self._import_workflow_class()
            self._workflow = workflow_class()
        except Exception as exc:
            self._error = (
                "Failed to load gptmed agentic workflow from installed package. "
                f"module={self._workflow_module}, class={self._workflow_class_name}, error={exc}"
            )

    def get_status(self) -> Dict[str, Any]:
        """Return package adapter readiness details."""
        self._ensure_workflow()
        return {
            "available": self._workflow is not None,
            "source": self._source,
            "error": self._error,
        }

    def list_agents(self) -> Dict[str, Any]:
        """List agent metadata from the loaded workflow registry."""
        self._ensure_workflow()
        if self._workflow is None:
            raise RuntimeError(self._error or "Agentic workflow is not available")

        registry = getattr(self._workflow, "registry", None)
        if registry is None or not hasattr(registry, "list_agent_info"):
            raise RuntimeError(
                "gptmed workflow loaded, but no compatible registry API was found "
                "(expected: workflow.registry.list_agent_info())."
            )

        return {
            "source": self._source,
            "agents": registry.list_agent_info(),
        }

    def process_prescription(self, prescription_data: Any) -> Dict[str, Any]:
        """Execute workflow and return JSON-serializable step results."""
        self._ensure_workflow()
        if self._workflow is None:
            raise RuntimeError(self._error or "Agentic workflow is not available")

        results = self._workflow.process_prescription(prescription_data)
        serialized_results = {
            agent_name: result.to_dict() if hasattr(result, "to_dict") else result
            for agent_name, result in results.items()
        }

        return {
            "source": self._source,
            "results": serialized_results,
        }

    def process_prescription_file(
        self,
        file_bytes: bytes,
        mime_type: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Execute full 4-agent workflow starting from a raw file upload.

        Pipeline:
            FileIngestionAgent -> PrescriptionAnalyzer -> DoctorAgent -> PharmacistAgent

        Args:
            file_bytes: Raw bytes of the uploaded PDF or image.
            mime_type: MIME type string (e.g. ``"application/pdf"``).
            filename: Original filename (used as extension fallback).

        Returns:
            Dict with ``source`` and ``results`` keys.
        """
        self._ensure_workflow()
        if self._workflow is None:
            raise RuntimeError(self._error or "Agentic workflow is not available")

        results = self._workflow.process_prescription_from_file(
            file_bytes=file_bytes,
            mime_type=mime_type,
            filename=filename,
        )
        serialized_results = {
            agent_name: result.to_dict() if hasattr(result, "to_dict") else result
            for agent_name, result in results.items()
        }

        return {
            "source": self._source,
            "results": serialized_results,
        }
