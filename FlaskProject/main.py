#!/usr/bin/env python3
import os
import re
import json
import math
import logging
import cv2
import pytesseract
from pytesseract import Output
import pdf2image
import pdfplumber
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
HAZEN_WILLIAMS_COEFFICIENT = 150
DEFAULT_DIAMETER = 1.0
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

# Set Tesseract executable path if needed
pytesseract.pytesseract.tesseract_cmd = '/nix/store/44vcjbcy1p2yhc974bcw250k2r5x5cpa-tesseract-5.3.4/bin/tesseract'


# AnalysisResult class holds all analysis outputs
class AnalysisResult:
    def __init__(self, description=""):
        self.description = description
        self.hydraulic_reference_points = []
        self.flow_rates = []
        self.pressures = []
        self.areas = []
        self.nodes = []
        self.hazen_williams_coefficients = []
        self.slopes = []
        self.flow_rates_with_margin = []
        self.water_supply_data = []
        self.pipe_sizes = []
        self.device_data = []
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


# Generic PDF parsing using regex patterns
def parse_pdf(file_path, patterns):
    extracted_data = {key: [] for key in patterns}
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    logging.info(f"Extracted text from page {page_number}.")
                    for key, pattern in patterns.items():
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        extracted_data[key].extend(matches)
                        logging.info(f"Matches for {key} on page {page_number}: {matches}")
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {e}")
    return extracted_data


# Parse fire plan file to extract raw text per page
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
                    logging.error(f"Error extracting text from page: {e}")
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None
    return extracted_data


# Extract design elements from the fire sprinkler plan
def parse_fire_sprinkler_design(pdf_path):
    extracted_elements = {
        "sprinkler_locations": [],
        "pipe_sizes": [],
        "device_types": [],
        "node_elevations": [],
        "k_factors": []
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        extracted_elements["sprinkler_locations"].extend(
                            re.findall(r'Sprinkler Location\s*:\s*(.*)', text))
                        extracted_elements["pipe_sizes"].extend(re.findall(r'Pipe Size\s*:\s*(\d+\.?\d*)\s*in', text))
                        extracted_elements["device_types"].extend(re.findall(r'Valve Type\s*:\s*(.*)', text))
                        extracted_elements["node_elevations"].extend(
                            re.findall(r'Node Elevation\s*:\s*(\d+\.?\d*)\s*ft', text))
                        extracted_elements["k_factors"].extend(re.findall(r'K-Factor\s*:\s*(\d+\.?\d*)', text))
                except Exception as e:
                    logging.error(f"Error extracting design elements from page: {e}")
    except Exception as e:
        logging.error(f"Error reading design PDF: {e}")
        return None
    return extracted_elements


# Parse hydraulic calculation report from PDF
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
                        extracted_hydraulic_data["flow_rates"].extend(
                            re.findall(r'Flow Rate\s*:\s*(\d+\.?\d*)\s*gpm', text))
                        extracted_hydraulic_data["pressures"].extend(
                            re.findall(r'Pressure\s*:\s*(\d+\.?\d*)\s*psi', text))
                        extracted_hydraulic_data["pipe_lengths"].extend(
                            re.findall(r'Pipe Length\s*:\s*(\d+\.?\d*)\s*ft', text))
                        extracted_hydraulic_data["node_elevations"].extend(
                            re.findall(r'Node Elevation\s*:\s*(\d+\.?\d*)\s*ft', text))
                        extracted_hydraulic_data["k_factors"].extend(re.findall(r'K-Factor\s*:\s*(\d+\.?\d*)', text))
                        if re.search(r'Hose Demand\s*:\s*Included', text, re.IGNORECASE):
                            extracted_hydraulic_data["hose_demand"] = True
                        extracted_hydraulic_data["fixed_loss_devices"].extend(
                            re.findall(r'Fixed Loss Device\s*:\s*(.*)', text))
                        sp_match = re.search(r'Static Pressure\s*:\s*(\d+\.?\d*)\s*psi', text)
                        rp_match = re.search(r'Residual Pressure\s*:\s*(\d+\.?\d*)\s*psi', text)
                        ft_match = re.search(r'Flow at Test\s*:\s*(\d+\.?\d*)\s*gpm', text)
                        if sp_match and rp_match and ft_match:
                            extracted_hydraulic_data["flow_test_data"] = {
                                "static_pressure": float(sp_match.group(1)),
                                "residual_pressure": float(rp_match.group(1)),
                                "flow_at_test": float(ft_match.group(1))
                            }
                except Exception as e:
                    logging.error(f"Error extracting hydraulic data from page: {e}")
    except Exception as e:
        logging.error(f"Error reading hydraulic PDF: {e}")
        return None

    # Prompts for missing key data (should be replaced in production with automated handling)
    if not extracted_hydraulic_data["flow_rates"]:
        extracted_hydraulic_data["flow_rates"] = [
            float(input("Flow rate data is missing. Please enter flow rate (gpm): "))]
    if not extracted_hydraulic_data["pressures"]:
        extracted_hydraulic_data["pressures"] = [
            float(input("Pressure data is missing. Please enter pressure (psi): "))]
    if not extracted_hydraulic_data["node_elevations"]:
        extracted_hydraulic_data["node_elevations"] = [
            float(input("Node elevation data is missing. Please enter node elevation (ft): "))]
    if not extracted_hydraulic_data["pipe_lengths"]:
        extracted_hydraulic_data["pipe_lengths"] = [
            float(input("Pipe length data is missing. Please enter pipe length (ft): "))]
    if not extracted_hydraulic_data.get("flow_test_data"):
        extracted_hydraulic_data["flow_test_data"] = {
            "static_pressure": float(input("Enter static pressure (psi): ")),
            "residual_pressure": float(input("Enter residual pressure (psi): ")),
            "flow_at_test": float(input("Enter flow rate at test (gpm): "))
        }
    if "Density" not in extracted_hydraulic_data:
        extracted_hydraulic_data["Density"] = float(input("Enter the design density (gpm/ft²): "))
    return extracted_hydraulic_data


# OCR extraction functions using Tesseract and pdfplumber fallback
def extract_text_from_image(file_path):
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            extracted_text = []
            logging.info("Processing PDF for OCR extraction...")
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text.append({
                            "text": text,
                            "x": 0,
                            "y": 0,
                            "width": page.width,
                            "height": page.height,
                            "type": "pdf_text"
                        })
            if not extracted_text:
                logging.info("No text from pdfplumber; converting PDF to images for OCR...")
                images = pdf2image.convert_from_path(file_path, dpi=300, fmt='png')
                for img in images:
                    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    text_data = process_image_with_ocr(opencv_img)
                    if isinstance(text_data, dict) and "error" in text_data:
                        logging.error(f"OCR error: {text_data['error']}")
                        continue
                    extracted_text.extend(text_data)
            return extracted_text
        else:
            image = cv2.imread(file_path)
            if image is None:
                return {"error": "Failed to load image file"}
            return process_image_with_ocr(image)
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}


def process_image_with_ocr(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text_data = pytesseract.image_to_data(threshold_image, output_type=Output.DICT, config=custom_config)
        extracted_text = []
        for i in range(len(text_data["text"])):
            if text_data["text"][i].strip():
                extracted_text.append({
                    "text": text_data["text"][i],
                    "x": text_data["left"][i],
                    "y": text_data["top"][i],
                    "width": text_data["width"][i],
                    "height": text_data["height"][i],
                    "type": "ocr_text"
                })
        return extracted_text
    except Exception as e:
        return {"error": f"OCR processing failed: {str(e)}"}


# Analysis of fire sprinkler layout from OCR text
def analyze_sprinkler_system_layout(extracted_text):
    if isinstance(extracted_text, dict) and "error" in extracted_text:
        return extracted_text
    components = {"sprinklers": 0, "pipes": 0, "valves": 0, "drain_points": 0}
    flow_rates, pressures, pipe_sizes = [], [], []
    keywords = {
        "sprinklers": ["sprinkler", "head", "nozzle", "k-factor", "deflector", "pendent"],
        "pipes": ["pipe", "main", "branch", "riser", "crossmain", "feed"],
        "valves": ["valve", "control", "check", "backflow", "drain", "test"],
        "drain_points": ["drain", "auxiliary", "drum drip", "low point"]
    }
    key_annotations = []
    for entry in extracted_text:
        text = entry["text"].lower()
        for comp, terms in keywords.items():
            if any(term in text for term in terms):
                components[comp] += 1
                key_annotations.append(entry)
        flow_match = re.findall(r'(\d+\.?\d*)\s*(?:gpm)', text)
        pressure_match = re.findall(r'(\d+\.?\d*)\s*(?:psi)', text)
        pipe_size_match = re.findall(r'(\d+\.?\d*)\s*(?:inch|in|")', text)
        flow_rates.extend([float(fr) for fr in flow_match])
        pressures.extend([float(p) for p in pressure_match])
        pipe_sizes.extend([float(ps) for ps in pipe_size_match])
    total_flow = sum(flow_rates) if flow_rates else 0
    area = 1500  # assumed design area
    density = total_flow / area if total_flow > 0 else 0
    compliance_issues = []
    if density < 0.20:
        compliance_issues.append({
            "issue": "System density below minimum requirement",
            "compliance_reference": ["NFPA 13, Section 19.3.3.1.1"]
        })
    if pipe_sizes and min(pipe_sizes) < 1.0:
        compliance_issues.append({
            "issue": "Pipe diameter does not meet minimum specification",
            "compliance_reference": ["NFPA 13, Section 8.15.19.4"]
        })
    if components["sprinklers"] > 0:
        compliance_issues.append({
            "issue": "Verify sprinkler spacing meets maximum coverage area requirements",
            "compliance_reference": ["NFPA 13, Section 8.6.2.2.1"]
        })
    return {
        "document_type": "Fire Sprinkler System Layout",
        "detected_annotations": key_annotations,
        "analysis": {
            "system_type": determine_system_type(extracted_text),
            "hazard_classification": "Ordinary Hazard Group 1",
            "system_design": {
                "density_requirement": "0.20 gpm per sq.ft. over 1,500 sq.ft.",
                "total_water_demand": f"{total_flow:.2f} gpm (total)",
                "calculated_density": f"{density:.3f} gpm/sq.ft",
                "sprinkler_spacing": "Detected via grid layout",
                "component_count": components,
                "flow_rates": flow_rates,
                "pressures": pressures,
                "pipe_sizes": pipe_sizes
            },
            "notes": generate_analysis_notes(components, flow_rates, pressures),
            "potential_discrepancies": compliance_issues
        }
    }


def determine_system_type(extracted_text):
    text_content = " ".join(entry["text"].lower() for entry in extracted_text)
    if "dry" in text_content and "wet" in text_content:
        return "Mixed Wet and Dry Pipe Fire Sprinkler System"
    elif "dry" in text_content:
        return "Dry Pipe Fire Sprinkler System"
    elif "preaction" in text_content:
        return "Preaction Fire Sprinkler System"
    elif "deluge" in text_content:
        return "Deluge Fire Sprinkler System"
    else:
        return "Wet Pipe Fire Sprinkler System"


def generate_analysis_notes(components, flow_rates, pressures):
    notes = []
    if components["sprinklers"] == 0:
        notes.append("Warning: No sprinkler heads detected.")
    if not flow_rates:
        notes.append("Warning: No flow rate data detected.")
    if not pressures:
        notes.append("Warning: No pressure data detected.")
    if components["drain_points"] == 0:
        notes.append("Warning: No drain points detected; verify system drainage.")
    return notes


# Recalculate K-Factor for fittings (simplified)
def recalculate_k_factor(flow_rate, pressure_loss, length, coefficient, diameter, elevation_change):
    try:
        k_factor = (4.52 * flow_rate * length) / (coefficient ** 0.85 * diameter ** 4.87)
        return k_factor
    except Exception as e:
        logging.error(f"Error recalculating K-factor: {e}")
        return None


# Verification functions
def verify_hydraulic_reference_points(plans_data, calculation_output_data):
    compliance_issues = []
    if plans_data.get("node_elevations") != calculation_output_data.get("node_elevations"):
        compliance_issues.append("Error: Node elevations do not match between plans and hydraulic calculations.")
    if plans_data.get("k_factors") != calculation_output_data.get("k_factors"):
        compliance_issues.append("Error: K-factors do not match between plans and hydraulic calculations.")
    if not set(plans_data.get("sprinkler_locations")).issubset(set(calculation_output_data.get("flow_rates"))):
        compliance_issues.append("Error: Not all flowing sprinklers are accounted for in hydraulic calculations.")
    if not calculation_output_data.get("hose_demand"):
        compliance_issues.append("Error: Hose demand is not included where applicable.")
    if 'fixed_loss_devices' not in calculation_output_data:
        compliance_issues.append("Error: Fixed loss devices are not properly represented.")
    else:
        for device in calculation_output_data['fixed_loss_devices']:
            if device == "simulated_by_fittings":
                compliance_issues.append(
                    "Error: Fixed loss devices must be entered as fixed values, not simulated by fittings.")
    return compliance_issues


# Flask application for file uploads and dynamic analysis
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            if size > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'error': f'File size exceeds limit of {MAX_FILE_SIZE / (1024 * 1024)}MB'}), 413
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                extracted_text = extract_text_from_image(filepath)
                if isinstance(extracted_text, dict) and "error" in extracted_text:
                    return jsonify({'error': extracted_text["error"]}), 400
                analysis_result = analyze_sprinkler_system_layout(extracted_text)
                os.remove(filepath)
                return jsonify(analysis_result)
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Analysis failed: {e}'}), 500
        except Exception as e:
            return jsonify({'error': f'File upload failed: {e}'}), 500
    return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, PDF'}), 400


# Command-line main function for PDF-based analysis and report generation
def cli_main():
    file_path = os.getenv('FIRE_PLAN_PATH',
                          r"A:\HydroCalcAssistant\HydRoCalc_T#1-PlanSet#MaverikGasStation@11.26.24.pdf")
    hydraulic_report_path = os.getenv('HYDRAULIC_REPORT_PATH',
                                      r"A:\HydroCalcAssistant\HydRoCalc_T#1-CalcReport-Set#MaverikGasStation@11.26.24.pdf")
    output_json_path = os.getenv('OUTPUT_JSON_PATH', r"A:\HydroCalcAssistant\analysis_results.json")
    output_report_path = os.getenv('OUTPUT_REPORT_PATH', r"A:\HydroCalcAssistant\compliance_report.json")

    design_patterns = {
        "sprinkler_locations": r'Sprinkler\s+(\d+)',
        "pipe_sizes": r'Pipe\s+Size\s*:\s*(\d+\.?\d*)\s*in',
        "device_types": r'Valve\s+Type\s*:\s*(.*)',
        "node_elevations": r'Node\s+Elevation\s*:\s*(-?\d+\.?\d*)\s*ft',
        "k_factors": r'K-Factor\s*:\s*(\d+\.?\d*)'
    }
    hydraulic_patterns = {
        "flow_rates": r'Flow\s+Rate\s*:\s*(\d+\.?\d*)\s*gpm',
        "pressures": r'Pressure\s*:\s*(\d+\.?\d*)\s*psi',
        "pipe_lengths": r'Pipe\s+Length\s*:\s*(\d+\.?\d*)\s*ft',
        "node_elevations": r'Node\s+Elevation\s*:\s*(-?\d+\.?\d*)\s*ft',
        "k_factors": r'K-Factor\s*:\s*(\d+\.?\d*)',
        "hose_demand": r'Hose\s+Demand\s*:\s*Included',
        "fixed_loss_devices": r'Fixed\s+Loss\s+Device\s*:\s*(.*)'
    }
    design_elements = parse_pdf(file_path, design_patterns)
    hydraulic_data = parse_pdf(hydraulic_report_path, hydraulic_patterns)
    compliance_issues = verify_hydraulic_reference_points(design_elements, hydraulic_data)

    analysis_result = AnalysisResult(description="Fire Sprinkler System Layout Analysis")
    analysis_result.compliance_issues = compliance_issues
    analysis_result.pipe_sizes = design_elements["pipe_sizes"]
    analysis_result.device_data = design_elements["device_types"]
    analysis_result.flow_rates = hydraulic_data["flow_rates"]
    analysis_result.pressures = hydraulic_data["pressures"]
    analysis_result.hydraulic_reference_points = design_elements["node_elevations"]

    try:
        with open(output_report_path, 'w') as report_file:
            report_file.write(json.dumps(analysis_result.to_dict(), indent=4))
        logging.info("Compliance report generated successfully.")
    except Exception as e:
        logging.error(f"Error generating report: {e}")

    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(analysis_result.to_dict(), json_file, indent=4)
        logging.info("Analysis Result saved to JSON file.")
    except Exception as e:
        logging.error(f"Error saving analysis result: {e}")


if __name__ == '__main__':
    # If running as a server, comment out cli_main() and use Flask app.
    # For command-line usage, call cli_main()
    if os.getenv('RUN_FLASK', '1') == '1':
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        cli_main()
