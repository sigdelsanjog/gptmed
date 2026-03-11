"""
FastAPI application for testing the gptmed agentic framework.

Start with:
    python run.py                  # from this folder
    uvicorn gptmed.api.app:app --reload  # from the gptmed package root

Endpoints
─────────
GET  /              Health check + available routes
GET  /status        Agentic workflow load status
GET  /agents        List registered agents
POST /prescriptions/process   Run full workflow on structured data
POST /prescriptions/demo      Run workflow on a built-in sample
POST /prescriptions/upload    Upload a PDF or image prescription
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from gptmed.agentic.agentic_api import BackendAgenticService
from .models import (
    AgentsResponse,
    PrescriptionRequest,
    StatusResponse,
    UploadResponse,
    WorkflowResponse,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GptMed Agentic API",
    description=(
        "Local test API for the gptmed multi-agent prescription workflow.\n\n"
        "Four agents run in sequence:\n"
        "0. **FileIngestionAgent** – extracts text from uploaded PDF/image\n"
        "1. **PrescriptionAnalyzer** – parses & structures raw prescription data\n"
        "2. **DoctorAgent** – diagnoses, assesses risk, recommends tests\n"
        "3. **PharmacistAgent** – checks drug interactions, suggests medicines\n"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared instance – workflow initialisation is deferred to first use.
_service = BackendAgenticService()

_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    """Health check and route index."""
    return {
        "service": "gptmed-agentic-api",
        "version": "1.0.0",
        "routes": {
            "GET  /status":                   "Agentic workflow load status",
            "GET  /agents":                   "List registered agents",
            "POST /prescriptions/process":    "Run workflow on structured prescription data",
            "POST /prescriptions/demo":       "Run workflow on a built-in sample",
            "POST /prescriptions/upload":     "Upload a PDF or image prescription",
        },
    }


@app.get("/status", response_model=StatusResponse, tags=["meta"])
async def get_status():
    """Return whether the agentic workflow loaded successfully from the installed package."""
    return _service.get_status()


@app.get("/agents", response_model=AgentsResponse, tags=["agents"])
async def list_agents():
    """List all agents registered in the prescription workflow."""
    try:
        return _service.list_agents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/prescriptions/process", response_model=WorkflowResponse, tags=["workflow"])
async def process_prescription(request: PrescriptionRequest):
    """
    Execute the full agentic workflow on the provided structured prescription.

    Returns step-by-step results from all agents.
    """
    try:
        return _service.process_prescription(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")


@app.post("/prescriptions/demo", response_model=WorkflowResponse, tags=["workflow"])
async def run_demo():
    """
    Run the built-in demo prescription through the full workflow.

    Useful for verifying the agent pipeline works end-to-end without
    constructing a payload.
    """
    demo_prescription = {
        "patient_name": "John Doe",
        "patient_age": 45,
        "patient_gender": "M",
        "diagnosis": "Type 2 Diabetes Mellitus",
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
            {"name": "Insulin Glargine", "dosage": "10 units", "frequency": "At bedtime"},
        ],
        "doctor_name": "Dr. Smith",
        "prescription_date": "2026-03-11",
        "allergies": ["Penicillin"],
        "special_instructions": "Take Metformin with food",
    }

    try:
        return _service.process_prescription(demo_prescription)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")


@app.post("/prescriptions/upload", response_model=UploadResponse, tags=["workflow"])
async def upload_prescription(file: UploadFile = File(...)):
    """
    Upload a prescription PDF or image and run the full 4-agent workflow.

    Supported formats: PDF, JPEG, PNG, TIFF, BMP, WEBP.

    Pipeline: FileIngestionAgent → PrescriptionAnalyzer → DoctorAgent → PharmacistAgent
    """
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed types: {sorted(_ALLOWED_MIME_TYPES)}"
            ),
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        workflow_result = _service.process_prescription_file(
            file_bytes=file_bytes,
            mime_type=content_type,
            filename=file.filename or "upload",
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                f"OCR/PDF dependency missing: {exc}. "
                "Install via: pip install 'gptmed[agentic-api]' pytesseract && "
                "sudo apt install tesseract-ocr"
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")

    return {
        **workflow_result,
        "filename": file.filename or "upload",
        "content_type": content_type,
        "file_size_bytes": len(file_bytes),
    }
