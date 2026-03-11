"""
Doctor Agent - Agent 2

Analyzes structured prescription and provides:
- Diagnosis confirmation/interpretation
- Symptom analysis
- Disease predictions (differential diagnosis)
- Risk assessment
- Recommended tests
- Lifestyle recommendations
"""

import sys
import os
from typing import Any, Dict, Optional, List

# Setup imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.base_agent import BaseAgent


class DoctorAgent(BaseAgent):
    """
    Agent 2: Doctor Agent
    
    Input: Structured prescription from PrescriptionAnalyzerAgent
    Output: Medical analysis including diagnosis, symptoms, risk, recommendations
    
    Responsibilities:
    - Interpret medications to infer diagnosis
    - Identify associated symptoms
    - Perform differential diagnosis
    - Assess patient risk level
    - Recommend diagnostic tests
    - Provide lifestyle guidance
    """
    
    def __init__(self):
        super().__init__(
            name="DoctorAgent",
            description="Analyzes prescription and provides medical diagnosis/interpretation",
            enabled=True
        )
        
        # Knowledge base: medication → likely conditions
        self.medication_conditions = {
            "Metformin": ["Type 2 Diabetes", "Prediabetes", "PCOS"],
            "Insulin": ["Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes"],
            "Lisinopril": ["Hypertension", "Heart Failure", "Post-MI"],
            "Atorvastatin": ["Hypercholesterolemia", "Cardiovascular Disease"],
            "Aspirin": ["Cardiovascular Protection", "Post-Stroke Prevention"],
            "Omeprazole": ["GERD", "Peptic Ulcer", "Gastritis"],
            "Amoxicillin": ["Bacterial Infection", "Strep Throat", "Pneumonia"],
            "Fluoxetine": ["Depression", "Anxiety", "OCD", "Panic Disorder"],
            "Levothyroxine": ["Hypothyroidism", "Hashimoto's Thyroiditis"],
            "Albuterol": ["Asthma", "COPD", "Reactive Airway Disease"]
        }
        
        # Knowledge base: condition → symptoms
        self.condition_symptoms = {
            "Type 2 Diabetes": [
                "Increased thirst", "Frequent urination", "Fatigue", 
                "Blurred vision", "Slow wound healing", "Tingling in extremities"
            ],
            "Hypertension": [
                "Headaches", "Dizziness", "Shortness of breath", 
                "Chest discomfort", "Nosebleeds", "Vision changes"
            ],
            "Depression": [
                "Persistent sadness", "Loss of interest in activities", 
                "Sleep disturbance", "Fatigue", "Difficulty concentrating"
            ],
            "Asthma": [
                "Shortness of breath", "Chest tightness", "Wheezing", 
                "Dry cough", "Difficulty with exercise"
            ],
            "GERD": [
                "Heartburn", "Acid reflux", "Chest pain", "Difficulty swallowing"
            ]
        }
        
        # Knowledge base: condition → recommended tests
        self.condition_tests = {
            "Type 2 Diabetes": ["HbA1c", "Fasting Glucose", "Kidney Function", "Lipid Panel"],
            "Hypertension": ["Blood Pressure Monitoring", "Kidney Function", "ECG"],
            "Depression": ["Thyroid Function", "Vitamin B12", "Liver Function"],
            "Asthma": ["Spirometry", "Chest X-Ray", "Peak Flow Test"]
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input is structured prescription data."""
        if not isinstance(input_data, dict):
            return False
        
        # Must have minimum fields from PrescriptionAnalyzer
        required = {"diagnosis", "medications"}
        return all(key in input_data for key in required)
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output has required medical analysis fields."""
        required_keys = {"extracted_diagnosis", "suggested_symptoms", "possible_diseases"}
        return all(key in output for key in required_keys)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process prescription to provide medical analysis.
        
        Args:
            input_data: Structured prescription data
        
        Returns:
            Medical analysis dictionary
        """
        diagnosis = input_data.get("diagnosis", "Unknown")
        medications = input_data.get("medications", [])
        age = input_data.get("patient_age")
        
        # Generate analysis
        possible_diseases = self._infer_diseases(medications, diagnosis)
        symptoms = self._get_symptoms(diagnosis)
        risk_level, risk_reason = self._assess_risk(diagnosis, medications, age)
        tests = self._get_recommended_tests(diagnosis)
        lifestyle = self._get_lifestyle_recommendations(diagnosis)
        
        result = {
            "extracted_diagnosis": diagnosis,
            "suggested_symptoms": symptoms,
            "possible_diseases": possible_diseases,
            "risk_assessment": {
                "level": risk_level,
                "reason": risk_reason
            },
            "recommended_tests": tests,
            "lifestyle_recommendations": lifestyle,
            "metadata": {
                "medications_count": len(medications),
                "analysis_timestamp": input_data.get("prescription_date", "Unknown")
            },
            "_confidence": self._calculate_confidence(diagnosis, medications, symptoms)
        }
        
        return result
    
    def _infer_diseases(self, medications: List[Dict[str, str]], diagnosis: str) -> List[Dict[str, Any]]:
        """
        Infer possible diseases based on prescribed medications.
        
        Returns: Sorted list of diseases with confidence scores
        """
        diseases_found = {}
        
        # Analyze each medication
        for med in medications:
            med_name = med.get("name", "").strip()
            
            # Exact match
            if med_name in self.medication_conditions:
                conditions = self.medication_conditions[med_name]
                for condition in conditions:
                    if condition not in diseases_found:
                        diseases_found[condition] = {"confidence": 0.0, "reasons": []}
                    diseases_found[condition]["confidence"] += 0.35
                    diseases_found[condition]["reasons"].append(f"Prescribed: {med_name}")
            
            # Partial match
            else:
                for med_key in self.medication_conditions:
                    if med_key.lower() in med_name.lower():
                        conditions = self.medication_conditions[med_key]
                        for condition in conditions:
                            if condition not in diseases_found:
                                diseases_found[condition] = {"confidence": 0.0, "reasons": []}
                            diseases_found[condition]["confidence"] += 0.2
                            diseases_found[condition]["reasons"].append(f"Similar to: {med_name}")
        
        # Boost confidence for explicit diagnosis
        if diagnosis and diagnosis != "Unknown":
            if diagnosis not in diseases_found:
                diseases_found[diagnosis] = {"confidence": 0.90, "reasons": ["Explicit diagnosis"]}
            else:
                diseases_found[diagnosis]["confidence"] = min(0.95, diseases_found[diagnosis]["confidence"] + 0.4)
                diseases_found[diagnosis]["reasons"].insert(0, "Explicit diagnosis")
        
        # Convert to sorted list
        result = []
        for disease, data in sorted(diseases_found.items(), 
                                    key=lambda x: x[1]["confidence"], 
                                    reverse=True):
            result.append({
                "disease": disease,
                "confidence": min(1.0, data["confidence"]),
                "reason": " + ".join(data["reasons"])
            })
        
        return result[:5]  # Top 5 diseases
    
    def _get_symptoms(self, diagnosis: str) -> List[str]:
        """Get symptoms associated with diagnosis."""
        # Direct lookup
        if diagnosis in self.condition_symptoms:
            return self.condition_symptoms[diagnosis]
        
        # Partial match
        for condition in self.condition_symptoms:
            if condition.lower() in diagnosis.lower():
                return self.condition_symptoms[condition]
        
        return []
    
    def _assess_risk(self, diagnosis: str, medications: List[Dict[str, str]], 
                    age: Optional[int] = None) -> tuple[str, str]:
        """Assess patient risk level."""
        score = 0.0
        reasons = []
        
        # Base risk from diagnosis
        high_risk = ["Heart Failure", "Acute MI", "Stroke", "Sepsis"]
        medium_risk = ["Type 2 Diabetes", "Hypertension", "COPD"]
        
        if any(cond.lower() in diagnosis.lower() for cond in high_risk):
            score += 3.0
            reasons.append("High-risk diagnosis")
        elif any(cond.lower() in diagnosis.lower() for cond in medium_risk):
            score += 1.5
            reasons.append("Medium-risk diagnosis")
        
        # Polypharmacy risk
        med_count = len(medications)
        if med_count >= 5:
            score += 1.5
            reasons.append(f"Polypharmacy ({med_count} medications)")
        elif med_count >= 3:
            score += 0.5
            reasons.append(f"Multiple medications ({med_count})")
        
        # Age risk
        if age and age > 65:
            score += 1.0
            reasons.append(f"Age {age} (senior patient)")
        
        # Determine level
        if score >= 3.0:
            level = "High"
        elif score >= 1.5:
            level = "Medium"
        else:
            level = "Low"
        
        reason = " + ".join(reasons) if reasons else "Routine monitoring recommended"
        
        return level, reason
    
    def _get_recommended_tests(self, diagnosis: str) -> List[str]:
        """Get recommended diagnostic tests."""
        # Direct lookup
        if diagnosis in self.condition_tests:
            return self.condition_tests[diagnosis]
        
        # Partial match
        for condition in self.condition_tests:
            if condition.lower() in diagnosis.lower():
                return self.condition_tests[condition]
        
        # Generic tests
        return ["Complete Blood Count", "Basic Metabolic Panel", "Liver Function Tests"]
    
    def _get_lifestyle_recommendations(self, diagnosis: str) -> List[str]:
        """Get lifestyle recommendations."""
        recommendations = {
            "Type 2 Diabetes": [
                "Regular exercise (30 min, 5 days/week)",
                "Low-carb, high-fiber diet",
                "Monitor blood sugar regularly",
                "Maintain healthy weight"
            ],
            "Hypertension": [
                "Reduce sodium (<2000mg/day)",
                "Regular aerobic exercise",
                "Healthy BMI",
                "Stress management",
                "Limit alcohol"
            ],
            "Depression": [
                "Regular physical activity",
                "Consistent sleep schedule",
                "Social engagement",
                "Avoid alcohol/drugs",
                "Professional counseling"
            ]
        }
        
        # Direct lookup
        if diagnosis in recommendations:
            return recommendations[diagnosis]
        
        # Partial match
        for condition in recommendations:
            if condition.lower() in diagnosis.lower():
                return recommendations[condition]
        
        # Generic recommendations
        return [
            "Regular physical activity",
            "Balanced diet",
            "7-9 hours sleep",
            "Stress management",
            "Regular medical follow-up"
        ]
    
    def _calculate_confidence(self, diagnosis: str, medications: List, symptoms: List) -> float:
        """Calculate analysis confidence."""
        score = 0.0
        max_score = 3.0
        
        if diagnosis and diagnosis != "Unknown":
            score += 1.0
        if medications and len(medications) > 0:
            score += 1.0
        if symptoms and len(symptoms) > 0:
            score += 1.0
        
        return score / max_score
