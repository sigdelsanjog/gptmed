"""Pydantic schemas for the agentic API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class PrescriptionRequest(BaseModel):
    """Payload for processing a prescription through the agentic workflow."""

    patient_name: str
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    diagnosis: str
    medications: List[Dict[str, str]]
    doctor_name: Optional[str] = None
    prescription_date: Optional[str] = None
    allergies: Optional[List[str]] = []
    special_instructions: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "patient_name": "Jane Doe",
                    "patient_age": 38,
                    "patient_gender": "F",
                    "diagnosis": "Hypertension",
                    "medications": [
                        {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily"}
                    ],
                    "doctor_name": "Dr. Adams",
                    "prescription_date": "2026-03-11",
                    "allergies": ["Aspirin"],
                    "special_instructions": "Monitor blood pressure weekly",
                }
            ]
        }
    }


# ── Responses ─────────────────────────────────────────────────────────────────

class AgentInfo(BaseModel):
    name: str
    description: str
    type: str
    enabled: str


class AgentsResponse(BaseModel):
    source: str
    agents: List[Dict[str, Any]]


class StatusResponse(BaseModel):
    available: bool
    source: Optional[str]
    error: Optional[str]


class AgentStepResult(BaseModel):
    agent_name: str
    status: str
    result: Optional[Any]
    confidence_score: float
    execution_time_ms: float
    timestamp: str
    error_message: Optional[str]
    error_type: Optional[str]
    metadata: Dict[str, Any]


class WorkflowResponse(BaseModel):
    source: str
    results: Dict[str, Any]


class UploadResponse(BaseModel):
    """Response for file-upload prescription processing."""

    source: str
    filename: str
    content_type: str
    file_size_bytes: int
    results: Dict[str, Any]
