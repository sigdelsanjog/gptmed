# Enterprise Agentic Medical Prescription Workflow

A production-ready multi-agent framework for processing medical prescriptions using autonomous AI agents. **Designed for scalability and extensibility**.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [The 3 Agents](#the-3-agents)
4. [Directory Structure](#directory-structure)
5. [Using the Framework](#using-the-framework)
6. [Extension Guide](#extension-guide)
7. [Key Design Decisions](#key-design-decisions)

---

## 🚀 Quick Start

### 1. Run Demo (Fastest Way)
```bash
cd gptmed/gptmed/agentic
python main.py --demo
```

### 2. Interactive Mode
```bash
python main.py
# Then select option 1 or 2
```

### 3. Process from File
```bash
python main.py --file prescription.json
```

### 4. List Agents
```bash
python main.py --list-agents
```

---

## 🏗️ Architecture Overview

### Core Principles

1. **Single Responsibility** - Each agent does ONE job well
2. **Fault Isolation** - Agent failures don't crash the system
3. **Observable** - All operations are logged and tracked
4. **Chainable** - Output of agent N → Input of agent N+1
5. **Configurable** - Behavior controlled via config, not hardcoded

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Input: Prescription Data                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  1️⃣ Prescription Analyzer Agent │
        │  - Parse raw prescription       │
        │  - Extract structured fields    │
        │  - Validate data                │
        └────────────────┬────────────────┘
                         │
                    Output: {
                      patient_name,
                      diagnosis,
                      medications[],
                      ...
                    }
                         │
        ┌────────────────▼────────────────┐
        │  2️⃣ Doctor Agent                │
        │  - Interpret medications        │
        │  - Suggest symptoms             │
        │  - Assess risk level            │
        │  - Recommend tests              │
        └────────────────┬────────────────┘
                         │
                    Output: {
                      possible_diseases[],
                      risk_assessment,
                      recommended_tests[],
                      ...
                    }
                         │
        ┌────────────────▼────────────────┐
        │  3️⃣ Pharmacist Agent            │
        │  - Check drug interactions      │
        │  - Verify contraindications     │
        │  - Recommend medicines          │
        │  - Validate allergies           │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  Final Output: Complete Analysis│
        │  - All recommendations          │
        │  - Safety alerts                │
        │  - Confidence scores            │
        └────────────────────────────────┘
```

---

## 👥 The 3 Agents

### Agent 1: Prescription Analyzer ✅

**Purpose**: Convert messy prescription data into clean structure

**Input**: Raw prescription text or dict
```python
{
    "patient_name": "John Doe",
    "patient_age": 45,
    "diagnosis": "Type 2 Diabetes",
    "medications": [
        {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"}
    ],
    ...
}
```

**Output**: Structured prescription
```json
{
  "patient_name": "John Doe",
  "patient_age": 45,
  "patient_gender": "M",
  "diagnosis": "Type 2 Diabetes Mellitus",
  "medications": [...],
  "doctor_name": "Dr. Smith",
  "allergies": ["Penicillin"],
  "special_instructions": "Take with food"
}
```

**Key Features**:
- Regex-based field extraction
- Validation and cleaning
- Confidence scoring
- Handles both raw text and structured input

---

### Agent 2: Doctor Agent 👨‍⚕️

**Purpose**: Analyze prescription from medical perspective

**Input**: Structured prescription from Agent 1

**Output**: Medical analysis
```json
{
  "extracted_diagnosis": "Type 2 Diabetes",
  "suggested_symptoms": [
    "Increased thirst",
    "Frequent urination",
    "Fatigue",
    "Blurred vision"
  ],
  "possible_diseases": [
    {
      "disease": "Type 2 Diabetes",
      "confidence": 0.95,
      "reason": "Metformin + Insulin prescription"
    }
  ],
  "risk_assessment": {
    "level": "Medium",
    "reason": "Multiple medications + age 45"
  },
  "recommended_tests": ["HbA1c", "Fasting Glucose"],
  "lifestyle_recommendations": [...]
}
```

**Key Features**:
- Medication → disease mapping
- Differential diagnosis
- Risk assessment based on age + medications
- Symptom analysis
- Test recommendations

---

### Agent 3: Pharmacist Agent 💊

**Purpose**: Pharmaceutical safety and recommendations

**Input**: Prescription + Doctor analysis

**Output**: Pharmacy recommendations
```json
{
  "current_medications": [
    {
      "name": "Metformin",
      "safety_status": "✅ SAFE",
      "side_effects": ["Nausea", "Diarrhea"],
      "notes": "Standard dosing"
    }
  ],
  "drug_interactions_check": {
    "interactions_found": [],
    "severity_summary": "✅ No interactions detected"
  },
  "contraindications": [],
  "recommended_medications": [
    {
      "name": "Atorvastatin",
      "priority": "PRIMARY",
      "indication": "Cardiovascular risk reduction in diabetes"
    }
  ],
  "safety_summary": "✅ SAFE: No major safety concerns detected"
}
```

**Key Features**:
- Drug-drug interaction database
- Contraindication checking
- Allergy compatibility
- Side effect warnings
- Medicine recommendations by diagnosis

---

## 📁 Directory Structure

```
gptmed/gptmed/agentic/
├─ core/                           # Framework core
│  ├─ __init__.py
│  ├─ base_agent.py               # Abstract agent interface
│  ├─ registry.py                 # Agent registry (singleton)
│  ├─ orchestrator.py             # Multi-agent orchestrator
│  ├─ result.py                   # Result wrapper (AgentResult)
│  ├─ logger.py                   # Logging system
│  └─ exceptions.py               # Custom exceptions
│
├─ agents/                         # Specialized agents
│  ├─ __init__.py
│  ├─ prescription_analyzer.py     # Agent 1
│  ├─ doctor_agent.py              # Agent 2
│  └─ pharmacist_agent.py          # Agent 3
│
├─ tools/                          # Shared utilities
│  ├─ ocr_tools.py                # (Future) OCR integration
│  ├─ drug_database.py            # (Future) Drug interactions API
│  └─ validators.py               # (Future) Data validators
│
├─ config/                         # Configuration files
│  └─ agent_config.yaml           # (Future) Agent parameters
│
├─ memory/                         # State management
│  └─ conversation_history.py     # (Future) Session history
│
├─ tests/                          # Unit & integration tests
│  ├─ test_agents.py
│  ├─ test_orchestrator.py
│  └─ test_workflow.py
│
├─ examples/                       # Sample prescriptions
│  ├─ sample_prescription.json
│  └─ expected_output.json
│
├─ logs/                           # Execution logs
│  └─ agentic_*.log
│
├─ main.py                         # Entry point (CLI)
├─ README.md                       # This file
└─ requirements.txt                # Python dependencies
```

---

## 💻 Using the Framework

### Option 1: Command Line (Recommended for Testing)

```bash
# Demo mode
python main.py --demo

# Interactive mode  
python main.py

# From file
python main.py --file my_prescription.json

# List agents
python main.py --list-agents
```

### Option 2: Python API

```python
from main import MedicalPrescriptionWorkflow

# Initialize
workflow = MedicalPrescriptionWorkflow()

# Define prescription
prescription = {
    "patient_name": "Jane Smith",
    "patient_age": 52,
    "diagnosis": "Hypertension",
    "medications": [
        {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily"}
    ]
}

# Execute workflow
results = workflow.process_prescription(prescription)

# Display results
workflow.display_results(results)
```

### Option 3: Direct Agent Usage

```python
from core import AgentRegistry, AgentOrchestrator
from agents import PrescriptionAnalyzerAgent

# Create agents
analyzer = PrescriptionAnalyzerAgent()

# Execute single agent
result = analyzer.execute({
    "raw_text": "Patient: John\nDiagnosis: Diabetes\nMeds: Metformin 500mg"
})

print(f"Status: {result.status}")
print(f"Confidence: {result.confidence_score:.2%}")
print(f"Result: {result.result}")
```

---

## 🔧 Extension Guide

### How to Add a 4th Agent

**Step 1**: Create agent file
```python
# agents/nutrition_agent.py

from ..core.base_agent import BaseAgent

class NutritionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="NutritionAgent",
            description="Provides dietary recommendations",
            enabled=True
        )
    
    def validate_input(self, input_data):
        # Validate input
        return "diagnosis" in input_data
    
    def validate_output(self, output):
        # Validate output structure
        return "recommendations" in output
    
    def process(self, input_data):
        # Your logic here
        diagnosis = input_data.get("diagnosis")
        return {
            "recommendations": ["Follow low sodium diet"],
            "_confidence": 0.85
        }
```

**Step 2**: Register in main.py
```python
from agents import NutritionAgent

def _register_agents(self):
    agents = [
        PrescriptionAnalyzerAgent(),
        DoctorAgent(),
        PharmacistAgent(),
        NutritionAgent()  # Add this
    ]
    # ...
```

**Step 3**: Add to workflow
```python
steps = [
    WorkflowStep(agent_name="PrescriptionAnalyzer"),
    WorkflowStep(agent_name="DoctorAgent"),
    WorkflowStep(agent_name="PharmacistAgent"),
    WorkflowStep(agent_name="NutritionAgent")  # Add this
]
```

### Key Points for New Agents

1. **Inherit from BaseAgent** - Ensures consistent interface
2. **Implement 3 methods**:
   - `validate_input()` - Check input is valid
   - `validate_output()` - Check output structure
   - `process()` - Core logic
3. **Return `_confidence`** - Quality metric (0.0-1.0)
4. **No try-except needed** - BaseAgent handles it
5. **Use AgentLogger** - For consistent logging

---

## 🎯 Key Design Decisions

### 1. Why Template Method Pattern?

The `BaseAgent.execute()` method defines the workflow:
```python
def execute():
    1. Validate input
    2. Check if enabled
    3. Call process()
    4. Validate output
    5. Wrap in AgentResult
    6. Handle exceptions
```

**Benefit**: Consistent error handling, logging, timing across all agents. No code duplication.

### 2. Why AgentResult Wrapper?

Agents return `AgentResult` (not raw dict) with:
- Status (COMPLETED, FAILED, SKIPPED, TIMEOUT)
- Confidence score
- Execution time
- Error details
- Metadata (lineage, dependencies)

**Benefit**: Orchestrator can chain results, track errors, and provide visibility.

### 3. Why AgentRegistry (Singleton)?

Central registry tracks all agents:
```python
registry = AgentRegistry()
registry.register(agent)
agent = registry.get("PrescriptionAnalyzer")
```

**Benefit**: Dynamic agent discovery, enable/disable at runtime, easy testing.

### 4. Why Sequential Execution?

Default: Agents run in sequence (Agent 1 → 2 → 3)

**Benefit**: 
- Simple, predictable
- Result chaining is clean
- Error handling straightforward
- Matches medical workflow (analyze → diagnose → prescribe)

**Future**: Support for parallel execution of independent agents.

### 5. Why Fail-Fast Option?

Errors can stop workflow or continue:

```python
# Stop on first failure
results = orchestrator.execute_workflow(steps, fail_fast=True)

# Continue despite failures
results = orchestrator.execute_workflow(steps, fail_fast=False)
```

**Benefit**: Flexibility for different use cases (strict QA vs best-effort)

---

## 📊 How Agents Interact

### Result Passing

```python
# Agent 1 Output → Agent 2 Input
result_1 = analyzer.execute(raw_data)
result_2 = doctor.execute(result_1.result)  # Use Agent 1's output
result_3 = pharmacist.execute(result_2.result)  # Use Agent 2's output
```

### Error Handling

```python
if result.is_failed():
    print(f"Status: {result.status}")
    print(f"Error: {result.error_message}")
    print(f"Error Type: {result.error_type}")
    # Decide: retry, skip, or escalate
```

### Confidence-Based Quality

```python
result = agent.execute(data)

if result.confidence_score < 0.7:
    print("⚠️ Low confidence - manual review recommended")
else:
    print("✅ High confidence - proceed")
```

---

## 📝 Example Workflow Execution

```python
workflow = MedicalPrescriptionWorkflow()

# Input
prescription = {
    "patient_name": "John Doe",
    "patient_age": 45,
    "diagnosis": "Type 2 Diabetes",
    "medications": [
        {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"}
    ],
    "allergies": ["Penicillin"]
}

# Execute
results = workflow.process_prescription(prescription)

# Results
{
    "PrescriptionAnalyzer": AgentResult(...),  # COMPLETED
    "DoctorAgent": AgentResult(...),           # COMPLETED
    "PharmacistAgent": AgentResult(...)        # COMPLETED
}

# Display
workflow.display_results(results)
```

**Output**:
```
════════════════════════════════════════════════════════════════════════════════
📊 WORKFLOW RESULTS
════════════════════════════════════════════════════════════════════════════════

✅ PrescriptionAnalyzer: completed (45.2ms) [confidence: 85%]

  👤 Patient: John Doe
  📅 Age: 45
  🏥 Diagnosis: Type 2 Diabetes Mellitus
  👨‍⚕️ Doctor: Not found
  
  💊 Medications (1):
     • Metformin 500mg - Twice daily

✅ DoctorAgent: completed (23.1ms) [confidence: 92%]

  🔍 Diagnosis: Type 2 Diabetes Mellitus
  
  🤒 Symptoms (6):
     • Increased thirst
     • Frequent urination
     • Fatigue
     • ...

✅ PharmacistAgent: completed (31.5ms) [confidence: 90%]

  💊 Drug Interactions: ✅ No interactions detected
  ✅ Recommended Additional Medicines:
     • Atorvastatin (PRIMARY) - Cardiovascular risk reduction
```

---

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_agents.py::test_prescription_analyzer

# With coverage
python -m pytest --cov=. tests/
```

---

## 📚 Next Steps

1. ✅ **Framework built** - Core, agents, orchestrator
2. ⏳ **OCR Integration** - Add image processing
3. ⏳ **External APIs** - PubMed, drug databases
4. ⏳ **API Server** - Flask/FastAPI REST endpoint
5. ⏳ **Database** - Prescription history, user profiles
6. ⏳ **Parallel Execution** - Support concurrent agents
7. ⏳ **Memory Enhancement** - Conversation context

---

## 💡 Key Takeaways

- **Scalable**: Add new agents without touching existing code
- **Observable**: Every operation is logged and tracked
- **Robust**: Individual agent failures don't crash system
- **Testable**: Each agent can be unit tested in isolation
- **Extensible**: Plug and play new agents, tools, databases

---

## 📞 Support

For issues, check:
1. `logs/agentic_*.log` - Detailed execution logs
2. Agent confidence scores - Quality indicator
3. AgentResult.error_message - Specific error details

Good luck! 🚀
