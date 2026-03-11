"""
Pharmacist Agent - Agent 3

Provides pharmaceutical recommendations:
- Medicine recommendations based on diagnosis
- Drug interaction checking
- Contraindication verification
- Dosage validation
- Side effect warnings
"""

import sys
import os
from typing import Any, Dict, Optional, List

# Setup imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.base_agent import BaseAgent


class PharmacistAgent(BaseAgent):
    """
    Agent 3: Pharmacist Agent
    
    Input: Structured prescription + Doctor analysis
    Output: Pharmaceutical recommendations and safety checks
    
    Responsibilities:
    - Recommend medicines based on diagnosis
    - Check drug interactions
    - Verify contraindications
    - Validate dosages
    - Identify side effects
    - Check allergy compatibility
    """
    
    def __init__(self):
        super().__init__(
            name="PharmacistAgent",
            description="Provides pharmaceutical recommendations and safety checks",
            enabled=True
        )
        
        # Drug interaction database
        self.drug_interactions = {
            ("Metformin", "Alcohol"): {"severity": "HIGH", "effect": "Lactic acidosis risk"},
            ("Warfarin", "Aspirin"): {"severity": "HIGH", "effect": "Increased bleeding risk"},
            ("Metformin", "Contrast Dye"): {"severity": "MEDIUM", "effect": "Kidney function risk"},
            ("Lisinopril", "Potassium Supplements"): {"severity": "MEDIUM", "effect": "Hyperkalemia risk"},
            ("Fluoxetine", "MAOIs"): {"severity": "HIGH", "effect": "Serotonin syndrome"},
        }
        
        # Contraindications: condition → contraindicated drugs
        self.contraindications = {
            "Pregnancy": ["Lisinopril", "Methotrexate", "Warfarin", "Aspirin (high dose)"],
            "Kidney Disease": ["NSAIDs", "Metformin (severe)", "ACE Inhibitors (caution)"],
            "Liver Disease": ["Acetaminophen", "Statins"],
            "Asthma": ["Beta Blockers", "NSAIDs"],
        }
        
        # Recommended medicines by condition
        self.disease_medications = {
            "Type 2 Diabetes": [
                {"name": "Metformin", "primary": True, "dosage": "500-1000mg", "frequency": "Twice daily"},
                {"name": "Sulfonylureas", "primary": False, "dosage": "Variable", "frequency": "Once/twice daily"},
                {"name": "GLP-1 Agonists", "primary": False, "dosage": "Variable", "frequency": "Weekly/daily"},
            ],
            "Hypertension": [
                {"name": "ACE Inhibitors", "primary": True, "dosage": "Variable", "frequency": "Once daily"},
                {"name": "Calcium Channel Blockers", "primary": True, "dosage": "Variable", "frequency": "Once daily"},
                {"name": "Beta Blockers", "primary": False, "dosage": "Variable", "frequency": "Once/twice daily"},
            ],
            "Depression": [
                {"name": "SSRIs", "primary": True, "dosage": "Variable", "frequency": "Once daily"},
                {"name": "SNRIs", "primary": True, "dosage": "Variable", "frequency": "Once daily"},
                {"name": "Tricyclic Antidepressants", "primary": False, "dosage": "Variable", "frequency": "At night"},
            ],
            "Asthma": [
                {"name": "Albuterol", "primary": True, "dosage": "2 puffs", "frequency": "As needed"},
                {"name": "Inhaled Corticosteroids", "primary": True, "dosage": "Variable", "frequency": "Daily"},
                {"name": "Leukotriene Inhibitors", "primary": False, "dosage": "Variable", "frequency": "Daily"},
            ],
        }
        
        # Side effects by medication
        self.side_effects = {
            "Metformin": ["Nausea", "Diarrhea", "Abdominal discomfort", "Metallic taste"],
            "Lisinopril": ["Dry cough", "Dizziness", "Fatigue", "Hyperkalemia"],
            "Atorvastatin": ["Muscle pain", "Liver enzyme elevation", "Headache"],
            "Aspirin": ["Bleeding risk", "Stomach upset", "Allergic reactions"],
            "Omeprazole": ["Headache", "Nausea", "Diarrhea", "Magnesium depletion"],
            "Fluoxetine": ["Insomnia", "Nausea", "Sexual dysfunction", "Weight changes"],
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input is prescription or prescription + analysis data."""
        if not isinstance(input_data, dict):
            return False
        
        # Accept either:
        # 1. Original prescription (has diagnosis + medications)
        # 2. Doctor analysis (has extracted_diagnosis + possible_diseases) 
        # 3. Combined data with any diagnosis-related field
        has_diagnosis = ("diagnosis" in input_data or "extracted_diagnosis" in input_data)
        has_meds_or_diseases = ("medications" in input_data or "possible_diseases" in input_data)
        
        return has_diagnosis and has_meds_or_diseases
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output has pharmaceutical recommendations."""
        required = {"current_medications", "recommended_medications", "drug_interactions_check"}
        return all(key in output for key in required)
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process prescription to provide pharmaceutical recommendations.
        
        Args:
            input_data: Structured prescription + doctor analysis (or either one)
        
        Returns:
            Pharmaceutical recommendations and safety checks
        """
        # Extract fields from either original prescription or doctor analysis
        medications = input_data.get("medications", [])
        
        # Handle diagnosis from either original or doctor analysis
        diagnosis = input_data.get("diagnosis") or input_data.get("extracted_diagnosis", "Unknown")
        
        # For a more complete analysis, try to infer medications from doctor's output if needed
        if not medications and "possible_diseases" in input_data:
            # This is doctor output, extract diagnosis from possible_diseases
            possible_diseases = input_data.get("possible_diseases", [])
            if possible_diseases and isinstance(possible_diseases, list) and len(possible_diseases) > 0:
                top_disease = possible_diseases[0]
                if isinstance(top_disease, dict):
                    diagnosis = top_disease.get("disease", diagnosis)
        
        allergies = input_data.get("allergies", [])
        
        # Analyze current medications
        current_med_safety = self._analyze_current_medications(medications, allergies)
        
        # Get drug interactions
        interactions = self._check_drug_interactions(medications)
        
        # Get contraindications
        contraindications = self._check_contraindications(diagnosis, medications)
        
        # Recommend additional medicines
        recommended = self._recommend_medications(diagnosis, medications)
        
        # Check allergy compatibility
        allergy_check = self._check_allergies(medications, allergies)
        
        result = {
            "current_medications": current_med_safety,
            "recommended_medications": recommended,
            "drug_interactions_check": {
                "interactions_found": interactions,
                "severity_summary": self._summarize_severity(interactions)
            },
            "contraindications": contraindications,
            "allergy_compatibility": allergy_check,
            "safety_summary": self._generate_safety_summary(
                interactions, contraindications, allergy_check
            ),
            "_confidence": self._calculate_confidence(medications, diagnosis)
        }
        
        return result
    
    def _analyze_current_medications(self, medications: List[Dict], allergies: List[str]) -> List[Dict]:
        """Analyze safety of current medications."""
        analysis = []
        
        for med in medications:
            med_name = med.get("name", "Unknown")
            
            # Get side effects
            side_effects = self.side_effects.get(med_name, ["Consult pharmacist for details"])
            
            # Check if allergy conflict
            allergy_conflict = any(allergy.lower() in med_name.lower() for allergy in allergies)
            
            analysis.append({
                "name": med_name,
                "dosage": med.get("dosage", "Not specified"),
                "frequency": med.get("frequency", "Not specified"),
                "side_effects": side_effects,
                "safety_status": "⚠️ ALLERGY CONFLICT" if allergy_conflict else "✅ SAFE",
                "notes": "Check with pharmacist about allergy history" if allergy_conflict else "Standard dosing"
            })
        
        return analysis
    
    def _check_drug_interactions(self, medications: List[Dict]) -> List[Dict]:
        """Check for drug interactions between medications."""
        interactions_found = []
        
        med_names = [med.get("name", "").strip() for med in medications]
        
        # Check all pairs
        for i, med1 in enumerate(med_names):
            for med2 in med_names[i+1:]:
                # Check both orders
                key1 = (med1, med2)
                key2 = (med2, med1)
                
                if key1 in self.drug_interactions:
                    interaction = self.drug_interactions[key1]
                    interactions_found.append({
                        "drug_1": med1,
                        "drug_2": med2,
                        "severity": interaction["severity"],
                        "effect": interaction["effect"]
                    })
                elif key2 in self.drug_interactions:
                    interaction = self.drug_interactions[key2]
                    interactions_found.append({
                        "drug_1": med2,
                        "drug_2": med1,
                        "severity": interaction["severity"],
                        "effect": interaction["effect"]
                    })
        
        return interactions_found
    
    def _check_contraindications(self, diagnosis: str, medications: List[Dict]) -> List[Dict]:
        """Check for contraindications based on diagnosis."""
        contraindications = []
        
        med_names = [med.get("name", "").strip() for med in medications]
        
        # Check each diagnosis-related contraindication
        for condition, contraindicated_drugs in self.contraindications.items():
            if condition.lower() in diagnosis.lower():
                for med_name in med_names:
                    for contra_drug in contraindicated_drugs:
                        if contra_drug.lower() in med_name.lower():
                            contraindications.append({
                                "condition": condition,
                                "contraindicated_drug": med_name,
                                "reason": f"{med_name} may worsen {condition}"
                            })
        
        return contraindications
    
    def _recommend_medications(self, diagnosis: str, current_meds: List[Dict]) -> List[Dict]:
        """Recommend additional medications based on diagnosis."""
        recommendations = []
        
        # Find matching diagnosis
        for condition, recommended_drugs in self.disease_medications.items():
            if condition.lower() in diagnosis.lower():
                # Only recommend if not already prescribed
                current_med_names = [m.get("name", "").lower() for m in current_meds]
                
                for drug in recommended_drugs:
                    if drug["name"].lower() not in current_med_names:
                        recommendations.append({
                            "name": drug["name"],
                            "recommended_dosage": drug["dosage"],
                            "frequency": drug["frequency"],
                            "priority": "PRIMARY" if drug["primary"] else "SECONDARY",
                            "indication": f"Recommended for {condition}"
                        })
        
        return recommendations
    
    def _check_allergies(self, medications: List[Dict], allergies: List[str]) -> Dict:
        """Check medication compatibility with known allergies."""
        med_names = [med.get("name", "").strip() for med in medications]
        conflicts = []
        
        for allergy in allergies:
            for med_name in med_names:
                if allergy.lower() in med_name.lower() or med_name.lower() in allergy.lower():
                    conflicts.append({
                        "allergy": allergy,
                        "medication": med_name,
                        "severity": "⚠️ POTENTIAL CONFLICT"
                    })
        
        return {
            "allergies_on_file": allergies,
            "conflicts_detected": conflicts,
            "status": "⚠️ REVIEW NEEDED" if conflicts else "✅ NO CONFLICTS"
        }
    
    def _summarize_severity(self, interactions: List[Dict]) -> str:
        """Summarize severity of interactions."""
        if not interactions:
            return "✅ No interactions detected"
        
        severities = [i.get("severity", "").upper() for i in interactions]
        
        if "HIGH" in severities:
            high_count = severities.count("HIGH")
            return f"⚠️ HIGH RISK: {high_count} serious interaction(s) detected"
        elif "MEDIUM" in severities:
            med_count = severities.count("MEDIUM")
            return f"⚠️ MODERATE RISK: {med_count} interaction(s) detected"
        else:
            return "ℹ️ Low-risk interactions detected"
    
    def _generate_safety_summary(self, interactions: List, contraindications: List, 
                                allergy_check: Dict) -> str:
        """Generate overall safety summary."""
        issues = []
        
        if interactions:
            issues.append(f"{len(interactions)} drug interaction(s)")
        
        if contraindications:
            issues.append(f"{len(contraindications)} contraindication(s)")
        
        if allergy_check["conflicts_detected"]:
            issues.append(f"{len(allergy_check['conflicts_detected'])} allergy conflict(s)")
        
        if not issues:
            return "✅ SAFE: No major safety concerns detected. Proceed with current plan."
        
        return f"⚠️ CAUTION: {', '.join(issues)}. Consult with healthcare provider before dispensing."
    
    def _calculate_confidence(self, medications: List, diagnosis: str) -> float:
        """Calculate recommendation confidence."""
        score = 0.0
        max_score = 2.0
        
        if medications and len(medications) > 0:
            score += 1.0
        if diagnosis and diagnosis != "Unknown":
            score += 1.0
        
        return score / max_score
