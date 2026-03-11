"""
Prescription Analyzer Agent - Agent 1

Converts messy prescription data into clean, structured format.
Handles OCR, text parsing, and validation.
"""

import re
import sys
import os
from typing import Any, Dict, Optional, List

# Setup imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.base_agent import BaseAgent


class PrescriptionAnalyzerAgent(BaseAgent):
    """
    Agent 1: Prescription Analyzer
    
    Input: Prescription data (raw text or dict)
    Output: Structured prescription dictionary
    
    Extracted Fields:
    - Patient info (name, age, gender)
    - Diagnosis
    - Medications (name, dosage, frequency)
    - Doctor info
    - Prescriber's date
    - Allergies
    - Special instructions
    """
    
    def __init__(self):
        super().__init__(
            name="PrescriptionAnalyzer",
            description="Converts prescription images/text into clean structured data",
            enabled=True
        )
        
        # Medication database for validation
        self.known_medications = {
            "metformin", "insulin", "aspirin", "lisinopril",
            "atorvastatin", "omeprazole", "amoxicillin", "fluoxetine",
            "levothyroxine", "albuterol", "ibuprofen", "paracetamol"
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input is either string or dict with prescription data."""
        if isinstance(input_data, str):
            # Must have at least some content
            return len(input_data.strip()) > 10
        elif isinstance(input_data, dict):
            # Must have some prescription fields
            keys = set(input_data.keys())
            required_keys = {"patient_name", "diagnosis", "medications"}
            return len(keys & required_keys) > 0
        return False
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate extracted structure has required fields."""
        required_keys = {"patient_name", "diagnosis", "medications"}
        has_required = all(key in output for key in required_keys)
        
        # Must have at least one medication
        medications = output.get("medications", [])
        has_medications_list = isinstance(medications, list) and len(medications) > 0
        
        return has_required and has_medications_list
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process prescription data into structured format.
        
        Args:
            input_data: Raw prescription text or dict
        
        Returns:
            Structured prescription dictionary
        """
        # If input is already a dict with structured prescription data, validate and return it
        if isinstance(input_data, dict) and all(
            key in input_data for key in ["patient_name", "diagnosis", "medications"]
        ):
            # Already structured - just clean it up and add confidence
            result = {}
            for key in ["patient_name", "patient_age", "patient_gender", "diagnosis", 
                       "medications", "doctor_name", "prescription_date", "allergies", 
                       "special_instructions"]:
                result[key] = input_data.get(key)
            result["_confidence"] = min(1.0, 0.9)  # High confidence for structured input
            return result
        
        # Handle as raw text
        if isinstance(input_data, dict):
            raw_text = self._dict_to_text(input_data)
        else:
            raw_text = input_data
        
        # Extract fields using regex patterns
        result = {
            "patient_name": self._extract_patient_name(raw_text),
            "patient_age": self._extract_age(raw_text),
            "patient_gender": self._extract_gender(raw_text),
            "diagnosis": self._extract_diagnosis(raw_text),
            "medications": self._extract_medications(raw_text),
            "doctor_name": self._extract_doctor_name(raw_text),
            "prescription_date": self._extract_date(raw_text),
            "allergies": self._extract_allergies(raw_text),
            "special_instructions": self._extract_special_instructions(raw_text),
        }
        
        # Add confidence score
        result["_confidence"] = self._calculate_confidence(result)
        
        return result
    
    def _dict_to_text(self, data: Dict) -> str:
        """Convert dict to text for processing."""
        lines = []
        for key, value in data.items():
            if value:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def _extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name."""
        patterns = [
            r'(?:Patient|Name):\s*([A-Za-z\s\.]*?)(?:\n|,|Age)',
            r'^([A-Z][a-z]+ [A-Z][a-z]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                if 2 <= len(name) <= 100:
                    return name
        
        return None
    
    def _extract_age(self, text: str) -> Optional[int]:
        """Extract patient age."""
        match = re.search(r'(?:Age|age):\s*(\d{1,3})', text)
        if match:
            age = int(match.group(1))
            if 0 < age < 150:
                return age
        return None
    
    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender (M/F/Male/Female)."""
        match = re.search(r'(?:Gender|Sex|gender|sex):\s*([MFmf]|Male|Female|male|female)', text)
        if match:
            val = match.group(1).upper()
            return "M" if val in ["M", "MALE"] else "F"
        return None
    
    def _extract_diagnosis(self, text: str) -> Optional[str]:
        """Extract diagnosis."""
        match = re.search(r'(?:Diagnosis|diagnosis|Chief Complaint):\s*([^\n]+)', text)
        if match:
            diagnosis = match.group(1).strip()
            if diagnosis and len(diagnosis) < 200:
                return diagnosis
        return None
    
    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        """Extract medications with dosage and frequency."""
        medications = []
        
        # Pattern: "Medication DosageUnit - Frequency/Instructions"
        # Example: "Metformin 500mg - twice daily", "1. Insulin 10 units - before meals"
        patterns = [
            r'(\d+\.\s)?([A-Za-z\s]+?)\s+(\d+(?:mg|ml|units|IU)?)\s*(?:[-–:]*)\s*(.+?)(?=\n\d+\.|Allergies|Doctor|Dr\.|$)',
            r'([A-Za-z]+)\s+(\d+(?:mg|ml|units)?)\s+(?:x|\*)?\s*(\d+(?:\s*(?:times|×|x)?)?(?:\s*daily|weekly|monthly)?)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                med_name = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
                dosage = match.group(3).strip() if len(match.groups()) >= 3 else ""
                frequency = match.group(4).strip() if len(match.groups()) >= 4 else ""
                
                # Filter to first line only
                if frequency:
                    frequency = frequency.split('\n')[0].strip()
                
                if med_name and len(med_name) < 100:
                    medications.append({
                        "name": med_name,
                        "dosage": dosage,
                        "frequency": frequency
                    })
        
        # Remove duplicates
        seen = set()
        unique_meds = []
        for med in medications:
            key = (med['name'].lower(), med['dosage'])
            if key not in seen:
                unique_meds.append(med)
                seen.add(key)
        
        return unique_meds
    
    def _extract_doctor_name(self, text: str) -> Optional[str]:
        """Extract doctor/physician name."""
        patterns = [
            r'(?:Dr\.?|Doctor|Physician)\s+([A-Za-z\s\.]+?)(?:\n|,|MD|$)',
            r'(?:Prescriber|Prescribed by):\s*([A-Za-z\s\.]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor = match.group(1).strip()
                if 2 < len(doctor) < 100:
                    return doctor
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract prescription date."""
        # Look for various date formats
        patterns = [
            r'(?:Date|date|Date Prescribed):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(?:Date|date):\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_allergies(self, text: str) -> List[str]:
        """Extract drug allergies."""
        allergies = []
        
        match = re.search(
            r'(?:Allergies?|Allergy|Drug Allergies):\s*(.+?)(?:\n(?:[A-Z]|Special|Doctor|Dr\.))',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if match:
            allergy_text = match.group(1).strip()
            # Split by comma, semicolon, or "and"
            allergy_list = re.split(r',|;|and', allergy_text, flags=re.IGNORECASE)
            allergies = [a.strip() for a in allergy_list if a.strip() and len(a.strip()) < 50]
        
        return allergies
    
    def _extract_special_instructions(self, text: str) -> Optional[str]:
        """Extract special instructions."""
        match = re.search(
            r'(?:Special\s+Instructions?|Instructions|Notes?|Remarks?):\s*(.+?)(?:\n(?:Dr\.|Allergies|$))',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if match:
            instructions = match.group(1).strip().replace('\n', ' ')
            if len(instructions) < 500:
                return instructions
        
        return None
    
    def _calculate_confidence(self, prescription: Dict) -> float:
        """Calculate confidence score based on extracted fields."""
        score = 0.0
        max_score = 7.0
        
        if prescription.get("patient_name"):
            score += 1
        if prescription.get("patient_age"):
            score += 1
        if prescription.get("diagnosis"):
            score += 1
        if prescription.get("medications") and len(prescription["medications"]) > 0:
            score += 2
        if prescription.get("doctor_name"):
            score += 1
        if prescription.get("prescription_date"):
            score += 1
        
        return score / max_score
