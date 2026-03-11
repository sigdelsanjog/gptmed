# -*- coding: utf-8 -*-
"""
Main Entry Point - Interactive CLI for Testing Agents

Usage:
    python main.py                   # Interactive mode
    python main.py --demo            # Run demo workflow
    python main.py --file data.json  # Process from file
"""

import sys
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from .core import AgentLogger, AgentRegistry, AgentOrchestrator, WorkflowStep
    from .agents import PrescriptionAnalyzerAgent, DoctorAgent, PharmacistAgent
except ImportError:
    # Fallback for direct execution: `python main.py` from this folder.
    from core import AgentLogger, AgentRegistry, AgentOrchestrator, WorkflowStep
    from agents import PrescriptionAnalyzerAgent, DoctorAgent, PharmacistAgent


class MedicalPrescriptionWorkflow:
    """Main workflow orchestrator for medical prescription processing."""
    
    def __init__(self):
        """Initialize workflow components."""
        # Setup logging
        AgentLogger.setup(level="INFO")
        self.logger = AgentLogger
        
        # Setup registry and orchestrator
        self.registry = AgentRegistry()
        self.orchestrator = AgentOrchestrator(self.registry)
        
        # Register all agents
        self._register_agents()
        
        self.logger.info("=" * 60)
        self.logger.info("🏥 Medical Prescription Workflow Initialized")
        self.logger.info("=" * 60)
    
    def _register_agents(self) -> None:
        """Register all available agents."""
        agents = [
            PrescriptionAnalyzerAgent(),
            DoctorAgent(),
            PharmacistAgent()
        ]
        
        for agent in agents:
            self.registry.register(agent)
        
        self.logger.info(f"✅ Registered {len(agents)} agents")
    
    def process_prescription(self, prescription_data: Any) -> Dict[str, Any]:
        """
        Execute full prescription workflow: Analyze → Diagnose → Recommend Medicine
        
        Args:
            prescription_data: Raw prescription data
        
        Returns:
            Complete workflow results
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📋 PROCESSING PRESCRIPTION")
        self.logger.info("=" * 60)
        
        # Execute workflow: Analyzer → Doctor → Pharmacist
        steps = [
            WorkflowStep(agent_name="PrescriptionAnalyzer"),
            WorkflowStep(agent_name="DoctorAgent"),
            WorkflowStep(agent_name="PharmacistAgent")
        ]
        
        results = self.orchestrator.execute_workflow(
            steps=steps,
            initial_input=prescription_data,
            fail_fast=False,
            timeout_sec=30
        )
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display workflow results in human-readable format."""
        print("\n" + "=" * 80)
        print("📊 WORKFLOW RESULTS")
        print("=" * 80)
        
        for agent_name, result in results.items():
            print(f"\n{result.to_summary()}")
            
            if result.is_success():
                self._print_agent_output(agent_name, result.result)
            elif result.error_message:
                print(f"   ❌ Error: {result.error_message}")
        
        print("\n" + "=" * 80)
        print(self.orchestrator.get_execution_summary())
        print("=" * 80)
    
    def _print_agent_output(self, agent_name: str, output: Dict[str, Any]) -> None:
        """Pretty print agent output."""
        if agent_name == "PrescriptionAnalyzer":
            self._print_prescription(output)
        elif agent_name == "DoctorAgent":
            self._print_diagnosis(output)
        elif agent_name == "PharmacistAgent":
            self._print_pharmacist_recommendations(output)
    
    def _print_prescription(self, prescription: Dict) -> None:
        """Pretty print prescription analysis."""
        print(f"\n  👤 Patient: {prescription.get('patient_name', 'N/A')}")
        print(f"  📅 Age: {prescription.get('patient_age', 'N/A')}")
        print(f"  🏥 Diagnosis: {prescription.get('diagnosis', 'N/A')}")
        print(f"  👨‍⚕️ Doctor: {prescription.get('doctor_name', 'N/A')}")
        
        print(f"\n  💊 Medications ({len(prescription.get('medications', []))}):") 
        for med in prescription.get('medications', []):
            print(f"     • {med['name']} {med.get('dosage', '')} - {med.get('frequency', '')}")
        
        allergies = prescription.get('allergies', [])
        if allergies:
            print(f"\n  ⚠️ Allergies: {', '.join(allergies)}")
        
        instructions = prescription.get('special_instructions')
        if instructions:
            print(f"\n  📝 Instructions: {instructions}")
    
    def _print_diagnosis(self, diagnosis: Dict) -> None:
        """Pretty print doctor's diagnosis."""
        print(f"\n  🔍 Diagnosis: {diagnosis.get('extracted_diagnosis', 'N/A')}")
        
        symptoms = diagnosis.get('suggested_symptoms', [])
        if symptoms:
            print(f"\n  🤒 Symptoms ({len(symptoms)}):")
            for symptom in symptoms[:5]:
                print(f"     • {symptom}")
        
        diseases = diagnosis.get('possible_diseases', [])
        if diseases:
            print(f"\n  🦠 Possible Diseases:")
            for disease in diseases[:3]:
                confidence = disease.get('confidence', 0) * 100
                print(f"     • {disease['disease']} ({confidence:.0f}%) - {disease.get('reason', '')}")
        
        risk = diagnosis.get('risk_assessment', {})
        print(f"\n  ⚠️ Risk Level: {risk.get('level', 'N/A')}")
        print(f"  Reason: {risk.get('reason', 'N/A')}")
        
        tests = diagnosis.get('recommended_tests', [])
        if tests:
            print(f"\n  🧪 Recommended Tests: {', '.join(tests[:3])}")
    
    def _print_pharmacist_recommendations(self, recommendations: Dict) -> None:
        """Pretty print pharmacist recommendations."""
        print(f"\n  💊 Current Medications:")
        for med in recommendations.get('current_medications', [])[:3]:
            print(f"     • {med['name']} - {med['safety_status']}")
        
        interactions = recommendations.get('drug_interactions_check', {})
        print(f"\n  🔗 Drug Interactions: {interactions.get('severity_summary', 'None detected')}")
        
        recommended = recommendations.get('recommended_medications', [])
        if recommended:
            print(f"\n  ✅ Recommended Additional Medicines:")
            for med in recommended[:3]:
                print(f"     • {med['name']} ({med['priority']}) - {med.get('indication', '')}")
        
        allergy_check = recommendations.get('allergy_compatibility', {})
        print(f"\n  🏥 Allergy Check: {allergy_check.get('status', 'N/A')}")
        
        print(f"\n  📋 Safety Summary: {recommendations.get('safety_summary', 'N/A')}")
    
    def interactive_mode(self) -> None:
        """Interactive mode for testing."""
        print("\n" + "=" * 80)
        print("🏥 MEDICAL PRESCRIPTION WORKFLOW - INTERACTIVE MODE")
        print("=" * 80)
        print("\nCommands:")
        print("  1. Run demo            - Execute with sample prescription")
        print("  2. Enter prescription  - Input custom prescription")
        print("  3. Show agents         - List registered agents")
        print("  4. Exit                - Quit")
        print("\n" + "-" * 80)
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                self._run_demo()
            elif choice == "2":
                self._input_custom_prescription()
            elif choice == "3":
                self._show_agents()
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Try again.")
    
    def _run_demo(self) -> None:
        """Run demonstration workflow with sample data."""
        sample_prescription = {
            "patient_name": "John Doe",
            "patient_age": 45,
            "patient_gender": "M",
            "diagnosis": "Type 2 Diabetes Mellitus",
            "medications": [
                {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
                {"name": "Insulin Glargine", "dosage": "10 units", "frequency": "At bedtime"}
            ],
            "doctor_name": "Dr. Smith",
            "prescription_date": "2026-03-11",
            "allergies": ["Penicillin"],
            "special_instructions": "Take Metformin with food"
        }
        
        print("\n📋 Sample Prescription:")
        print(json.dumps(sample_prescription, indent=2))
        
        results = self.process_prescription(sample_prescription)
        self.display_results(results)
    
    def _input_custom_prescription(self) -> None:
        """Allow user to input custom prescription."""
        print("\n📝 Enter Prescription Data (JSON format)")
        print("Example: {\"patient_name\": \"Jane\", \"diagnosis\": \"Diabetes\", ...}")
        
        try:
            raw_input = input("Prescription JSON: ").strip()
            prescription = json.loads(raw_input)
            
            results = self.process_prescription(prescription)
            self.display_results(results)
        
        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def _show_agents(self) -> None:
        """Display registered agents."""
        print("\n📋 Registered Agents:")
        print("-" * 80)
        
        for info in self.registry.list_agent_info():
            status = "✅" if info["enabled"] == "True" else "⊘"
            print(f"{status} {info['name']:<25} ({info['type']})")
            print(f"   {info['description']}")
        
        print("\n" + "-" * 80)
        print(f"Total: {len(self.registry.list_agents())} agents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Prescription Workflow - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo              # Run demo workflow
  python main.py --file data.json    # Process prescription from file
  python main.py                     # Interactive mode
        """
    )
    
    parser.add_argument("--demo", action="store_true", help="Run demo workflow")
    parser.add_argument("--file", type=str, help="Process prescription from JSON file")
    parser.add_argument("--list-agents", action="store_true", help="List registered agents")
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = MedicalPrescriptionWorkflow()
    
    # Handle commands
    if args.demo:
        workflow._run_demo()
    
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                prescription = json.load(f)
            results = workflow.process_prescription(prescription)
            workflow.display_results(results)
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in file: {args.file}")
    
    elif args.list_agents:
        workflow._show_agents()
    
    else:
        # Interactive mode
        workflow.interactive_mode()


if __name__ == "__main__":
    main()
