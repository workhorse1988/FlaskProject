import os
import json
import re
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
import pandas as pd
from datetime import datetime
import uuid


class AnalysisResult:
    """Class to hold and process analysis results of fire sprinkler system layouts."""

    def __init__(self, description=""):
        self.description = description
        self.hydraulic_reference_points = None
        self.flow_rates = None
        self.pressures = None
        self.areas = None
        self.nodes = None
        self.hazen_williams_coefficients = None
        self.slopes = None
        self.flow_rates_with_margin = None
        self.water_supply_data = None
        self.pipe_sizes = None
        self.device_data = None
        self.compliance_issues = []

    def to_dict(self):
        return {
            'description': self.description,
            'hydraulic_reference_points': self.hydraulic_reference_points,
            'flow_rates': self.flow_rates,
            'pressures': self.pressures,
            'areas': self.areas,
            'nodes': self.nodes,
            'hazen_williams_coefficients': self.hazen_williams_coefficients,
            'slopes': self.slopes,
            'flow_rates_with_margin': self.flow_rates_with_margin,
            'water_supply_data': self.water_supply_data,
            'pipe_sizes': self.pipe_sizes,
            'device_data': self.device_data,
            'compliance_issues': self.compliance_issues
        }


# Function to read and parse the fire plan PDF file using pdfplumber

def parse_fire_plan_file(file_path):
    extracted_data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        extracted_data.append(text)
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
    except Exception as e:
        print(f"Error reading PDF with pdfplumber: {e}")
        return None
    return extracted_data


# Enhanced parsing function for extracting design elements from the fire sprinkler layout

def parse_fire_sprinkler_design(pdf_path):
    extracted_elements = {"sprinkler_locations": [], "pipe_sizes": [], "device_types": [], "node_elevations": [], "k_factors": []}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        # Extract sprinkler locations
                        sprinkler_matches = re.findall(r'Sprinkler Location\s*:\s*(.*)', text)
                        extracted_elements["sprinkler_locations"].extend(sprinkler_matches)

                        # Extract pipe sizes
                        pipe_size_matches = re.findall(r'Pipe Size\s*:\s*(\d+\.?\d*)\s*in', text)
                        extracted_elements["pipe_sizes"].extend(pipe_size_matches)

                        # Extract device types
                        device_type_matches = re.findall(r'Valve Type\s*:\s*(.*)', text)
                        extracted_elements["device_types"].extend(device_type_matches)

                        # Extract node elevations
                        elevation_matches = re.findall(r'Node Elevation\s*:\s*(\d+\.?\d*)\s*ft', text)
                        extracted_elements["node_elevations"].extend(elevation_matches)

                        # Extract K-factors
                        k_factor_matches = re.findall(r'K-Factor\s*:\s*(\d+\.?\d*)', text)
                        extracted_elements["k_factors"].extend(k_factor_matches)
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
    except Exception as e:
        print(f"Error reading PDF with pdfplumber: {e}")
        return None
    return extracted_elements


# Function to parse hydraulic calculation report

def parse_hydraulic_calculations(pdf_path):
    extracted_hydraulic_data = {
        "flow_rates": [],
        "pressures": [],
        "pipe_lengths": [],
        "node_elevations": [],
        "k_factors": [],
        "hose_demand": False,
        "fixed_loss_devices": [],
        "flow_test_data": {}
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        # Extract flow rates
                        flow_rate_matches = re.findall(r'Flow Rate\s*:\s*(\d+\.?\d*)\s*gpm', text)
                        extracted_hydraulic_data["flow_rates"].extend(flow_rate_matches)

                        # Extract pressures
                        pressure_matches = re.findall(r'Pressure\s*:\s*(\d+\.?\d*)\s*psi', text)
                        extracted_hydraulic_data["pressures"].extend(pressure_matches)

                        # Extract pipe lengths
                        pipe_length_matches = re.findall(r'Pipe Length\s*:\s*(\d+\.?\d*)\s*ft', text)
                        extracted_hydraulic_data["pipe_lengths"].extend(pipe_length_matches)

                        # Extract node elevations
                        elevation_matches = re.findall(r'Node Elevation\s*:\s*(\d+\.?\d*)\s*ft', text)
                        extracted_hydraulic_data["node_elevations"].extend(elevation_matches)

                        # Extract K-factors
                        k_factor_matches = re.findall(r'K-Factor\s*:\s*(\d+\.?\d*)', text)
                        extracted_hydraulic_data["k_factors"].extend(k_factor_matches)

                        # Check for hose demand inclusion
                        if re.search(r'Hose Demand\s*:\s*Included', text, re.IGNORECASE):
                            extracted_hydraulic_data["hose_demand"] = True

                        # Extract fixed loss devices
                        fixed_loss_matches = re.findall(r'Fixed Loss Device\s*:\s*(.*)', text)
                        extracted_hydraulic_data["fixed_loss_devices"].extend(fixed_loss_matches)

                        # Extract flow test data
                        static_pressure_match = re.search(r'Static Pressure\s*:\s*(\d+\.?\d*)\s*psi', text)
                        residual_pressure_match = re.search(r'Residual Pressure\s*:\s*(\d+\.?\d*)\s*psi', text)
                        flow_at_test_match = re.search(r'Flow at Test\s*:\s*(\d+\.?\d*)\s*gpm', text)
                        if static_pressure_match and residual_pressure_match and flow_at_test_match:
                            extracted_hydraulic_data["flow_test_data"] = {
                                "static_pressure": float(static_pressure_match.group(1)),
                                "residual_pressure": float(residual_pressure_match.group(1)),
                                "flow_at_test": float(flow_at_test_match.group(1))
                            }
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
    except Exception as e:
        print(f"Error reading PDF with pdfplumber: {e}")
        return None

    # Handle missing data if needed.
    # If these lists end up empty, we can optionally prompt the user, or handle in some other way.
    if not extracted_hydraulic_data["flow_rates"]:
        extracted_hydraulic_data["flow_rates"] = [input("Flow rate data is missing. Please enter flow rate (gpm): ")]
    if not extracted_hydraulic_data["pressures"]:
        extracted_hydraulic_data["pressures"] = [input("Pressure data is missing. Please enter pressure (psi): ")]
    if not extracted_hydraulic_data["node_elevations"]:
        extracted_hydraulic_data["node_elevations"] = [input("Node elevation data is missing. Please enter node elevation (ft): ")]
    if not extracted_hydraulic_data["pipe_lengths"]:
        extracted_hydraulic_data["pipe_lengths"] = [input("Pipe length data is missing. Please enter pipe length (ft): ")]

    return extracted_hydraulic_data


def recalculate_k_factor(flow_rate, pressure_loss, length, hazen_williams_coefficient, diameter, elevation_change):
    """
    Recalculate the K-Factor for arm-overs, drops, and riser nipples.
    This formula is an example approach. Adjust the constants as needed.
    """
    try:
        # Example formula for demonstration only:
        k_factor = (4.52 * flow_rate * length) / (hazen_williams_coefficient ** 0.85 * diameter ** 4.87)
        return k_factor
    except Exception as e:
        print(f"Error recalculating K-factor: {e}")
        return None


def verify_hydraulic_reference_points(plans_data, calculation_output_data):
    """
    Verify if the hydraulic reference points and node information match.
    Also verify if design requirements such as node elevations, K-factors, hose demand,
    and flow test data align with the design.
    """
    compliance_issues = []

    # Verify node elevations
    if plans_data.get("node_elevations") != calculation_output_data.get("node_elevations"):
        compliance_issues.append("Error: Node elevations do not match between plans and hydraulic calculations.")

    # Verify K-factors
    if plans_data.get("k_factors") != calculation_output_data.get("k_factors"):
        compliance_issues.append("Error: K-factors do not match between plans and hydraulic calculations.")

    # Verify all flowing sprinklers are included
    if not set(plans_data.get("sprinkler_locations")).issubset(set(calculation_output_data.get("flow_rates"))):
        compliance_issues.append("Error: Not all flowing sprinklers in the remote area are accounted for in the hydraulic calculations.")

    # Verify hose demand inclusion
    if not calculation_output_data.get("hose_demand"):
        compliance_issues.append("Error: Hose demand is not included in the hydraulic calculations when required.")

    # Verify static pressure and residual pressure against flow test data
    flow_test_data = calculation_output_data.get("flow_test_data")
    if flow_test_data:
        static_pressure = flow_test_data.get("static_pressure")
        residual_pressure = flow_test_data.get("residual_pressure")

        # Example checks
        if static_pressure < 20:
            compliance_issues.append("Warning: Static pressure is below acceptable limits.")
        if residual_pressure < 10:
            compliance_issues.append("Warning: Residual pressure is below acceptable limits.")

    return compliance_issues


def generate_analysis_report(analysis_result, output_path):
    """Generate a PDF report summarizing the analysis."""
    try:
        c = canvas.Canvas(output_path)
        c.setFont("Helvetica", 12)

        # Title
        c.drawString(100, 800, f"Fire Sprinkler System Analysis Report - {analysis_result.description}")

        # Description
        c.drawString(100, 780, f"Description: {analysis_result.description}")

        y_position = 760

        # Print compliance issues
        if analysis_result.compliance_issues:
            c.drawString(100, y_position, "Compliance Issues:")
            y_position -= 20
            for issue in analysis_result.compliance_issues:
                c.drawString(100, y_position, f"- {issue}")
                y_position -= 20
        else:
            c.drawString(100, y_position, "No compliance issues found.")
            y_position -= 20

        # Flow Rates
        if analysis_result.flow_rates:
            c.drawString(100, y_position, f"Flow Rates: {', '.join(map(str, analysis_result.flow_rates))}")
            y_position -= 20

        # Pipe Sizes
        if analysis_result.pipe_sizes:
            c.drawString(100, y_position, f"Pipe Sizes: {', '.join(map(str, analysis_result.pipe_sizes))}")
            y_position -= 20

        c.save()
        print(f"Report generated successfully at {output_path}")
    except Exception as e:
        print(f"Error generating report: {e}")


def save_results_to_csv(analysis_result, output_file):
    """Save the analysis results to a CSV file."""
    try:
        data = analysis_result.to_dict()
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Results saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def run_fire_sprinkler_analysis(fire_plan_pdf, hydraulic_calculation_pdf, report_pdf_output, csv_output):
    """
    Main function to run the complete analysis and generate reports.
    """
    # Parse fire plan file
    fire_plan_data = parse_fire_plan_file(fire_plan_pdf)
    if not fire_plan_data:
        print("Error: Unable to parse fire plan PDF.")
        return

    # Parse fire sprinkler design
    sprinkler_design = parse_fire_sprinkler_design(fire_plan_pdf)
    if not sprinkler_design:
        print("Error: Unable to parse sprinkler design PDF.")
        return

    # Parse hydraulic calculations
    hydraulic_data = parse_hydraulic_calculations(hydraulic_calculation_pdf)
    if not hydraulic_data:
        print("Error: Unable to parse hydraulic calculations PDF.")
        return

    # Create the analysis result object
    analysis_result = AnalysisResult(description="Fire Sprinkler System Analysis")

    # Populate the analysis result object with extracted data
    analysis_result.flow_rates = hydraulic_data["flow_rates"]
    analysis_result.pressures = hydraulic_data["pressures"]
    analysis_result.pipe_sizes = sprinkler_design["pipe_sizes"]
    analysis_result.nodes = sprinkler_design.get("node_elevations", [])

    # We can store the K-factors from either set, or compare both.
    # For now, let's use the ones from the hydraulic calculations.
    analysis_result.device_data = sprinkler_design["device_types"]

    # Verify hydraulic reference points and check compliance
    compliance_issues = verify_hydraulic_reference_points(sprinkler_design, hydraulic_data)
    analysis_result.compliance_issues.extend(compliance_issues)

    # Generate report in PDF format
    generate_analysis_report(analysis_result, report_pdf_output)

    # Save results to CSV
    save_results_to_csv(analysis_result, csv_output)


if __name__ == "__main__":
    # Example usage:
    run_fire_sprinkler_analysis(
        fire_plan_pdf=r"A:\FPE\Pages from (SLB-FIRE25-0002) Solana Highlands MULTI-FAMILY - FIRE Sprinkler 2.pdf",
        hydraulic_calculation_pdf=r"A:\FPE\SLB FIRE25-0002 SET 1 CALCS.pdf",
        report_pdf_output=r"A:\FPE\output_report.pdf",
        csv_output=r"A:\FPE\analysis_results.csv"
    )
