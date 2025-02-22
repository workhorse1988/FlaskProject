import os
import cv2
import pytesseract
from pytesseract import Output
import pdf2image
import pdfplumber
import numpy as np
from datetime import datetime
import re

# Use the correct absolute path for Tesseract
pytesseract.pytesseract.tesseract_cmd = '/nix/store/44vcjbcy1p2yhc974bcw250k2r5x5cpa-tesseract-5.3.4/bin/tesseract'

def extract_text_from_image(image_path):
    """
    Extracts text annotations from the fire sprinkler system layout image using Tesseract OCR.
    Supports both image files (PNG, JPG) and PDFs.
    """
    try:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}

        file_extension = os.path.splitext(image_path)[1].lower()

        if file_extension == '.pdf':
            try:
                extracted_text = []
                print("Processing PDF file...")

                # First try with pdfplumber for text extraction
                with pdfplumber.open(image_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            # Extract key information using regex
                            extracted_text.append({
                                "text": text,
                                "x": 0,
                                "y": 0,
                                "width": page.width,
                                "height": page.height,
                                "type": "pdf_text"
                            })

                # If no text was extracted, try OCR
                if not extracted_text:
                    print("No text extracted with pdfplumber, trying OCR...")
                    images = pdf2image.convert_from_path(image_path, dpi=300, fmt='png')

                    for img in images:
                        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        text_data = process_image_with_ocr(opencv_img)
                        if isinstance(text_data, dict) and "error" in text_data:
                            print(f"OCR error: {text_data['error']}")
                            continue
                        extracted_text.extend(text_data)

                return extracted_text

            except Exception as e:
                return {"error": f"Failed to process PDF: {str(e)}"}
        else:
            # Process image files
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Failed to load image file"}

            return process_image_with_ocr(image)

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

def process_image_with_ocr(image):
    """Helper function to process image with OCR"""
    try:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        threshold_image = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Perform OCR with custom configuration
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text_data = pytesseract.image_to_data(threshold_image, output_type=Output.DICT, config=custom_config)

        # Process OCR results
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

def analyze_sprinkler_system_layout(extracted_text):
    """Analyzes fire sprinkler layout based on extracted OCR text."""
    if isinstance(extracted_text, dict) and "error" in extracted_text:
        return extracted_text

    # Initialize components and metrics
    components = {
        "sprinklers": 0,
        "pipes": 0,
        "valves": 0,
        "drain_points": 0
    }

    # Initialize data collection
    flow_rates = []
    pressures = []
    pipe_sizes = []
    elevations = []
    k_factors = []

    # Enhanced keywords for better component detection
    keywords = {
        "sprinklers": ["sprinkler", "head", "nozzle", "k-factor", "deflector", "upright", "pendent"],
        "pipes": ["pipe", "main", "branch", "riser", "crossmain", "feed", "diameter", "schedule"],
        "valves": ["valve", "control", "check", "backflow", "drain", "test", "inspector", "os&y"],
        "drain_points": ["drain", "auxiliary", "drum drip", "low point"]
    }

    # Analyze text entries
    key_annotations = []
    for entry in extracted_text:
        text = entry["text"].lower()

        # Count components
        for component, terms in keywords.items():
            if any(term in text for term in terms):
                components[component] += 1
                key_annotations.append(entry)

        # Extract numerical values with enhanced patterns
        flow_match = re.findall(r'(\d+\.?\d*)\s*(?:gpm|GPM)', text)
        pressure_match = re.findall(r'(\d+\.?\d*)\s*(?:psi|PSI)', text)
        pipe_size_match = re.findall(r'(\d+\.?\d*)\s*(?:inch|in|")', text)
        elevation_match = re.findall(r'(\d+\.?\d*)\s*(?:ft|FT|\')', text)
        k_factor_match = re.findall(r'[kK][-\s]*(?:factor|Factor)?\s*[=:]?\s*(\d+\.?\d*)', text)

        # Convert and store numerical values
        flow_rates.extend([float(fr) for fr in flow_match])
        pressures.extend([float(p) for p in pressure_match])
        pipe_sizes.extend([float(ps) for ps in pipe_size_match])
        elevations.extend([float(e) for e in elevation_match])
        k_factors.extend([float(k) for k in k_factor_match])

    # Calculate total water demand
    total_flow = sum(flow_rates) if flow_rates else 0
    area = 1500  # Standard design area in sq ft
    density = total_flow / area if total_flow > 0 else 0

    # Compliance checks based on NFPA 13
    compliance_issues = []

    # Check density requirement
    if density < 0.20:  # Required density of 0.20 gpm/sq ft
        compliance_issues.append({
            "issue": "System density below minimum requirement",
            "compliance_reference": ["NFPA 13, Section 19.3.3.1.1"]
        })

    # Check pipe sizes
    if pipe_sizes:
        min_pipe_size = min(pipe_sizes)
        if min_pipe_size < 1.0:  # Minimum pipe size requirement
            compliance_issues.append({
                "issue": "Pipe diameter does not meet minimum specification",
                "compliance_reference": ["NFPA 13, Section 8.15.19.4"]
            })

    # Check sprinkler spacing
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
                "density_requirement": "0.20 gpm per sq. ft. over 1,500 sq. ft.",
                "total_water_demand": f"{total_flow:.2f} gpm (total)",
                "calculated_density": f"{density:.3f} gpm/sq.ft",
                "sprinkler_spacing": "Detected via grid layout",
                "component_count": components,
                "flow_rates": flow_rates,
                "pressures": pressures,
                "pipe_sizes": pipe_sizes,
                "elevations": elevations,
                "k_factors": k_factors
            },
            "notes": generate_analysis_notes(components, flow_rates, pressures),
            "potential_discrepancies": compliance_issues
        }
    }

def determine_system_type(extracted_text):
    """Determine the type of sprinkler system based on extracted text."""
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
    """Generate analysis notes based on the extracted data."""
    notes = []

    if components["sprinklers"] == 0:
        notes.append("Warning: No sprinkler heads detected in the layout")

    if not flow_rates:
        notes.append("Warning: No flow rates detected in the layout")

    if not pressures:
        notes.append("Warning: No pressure values detected in the layout")

    if components["drain_points"] == 0:
        notes.append("Warning: No drain points detected - verify system drainage provisions")

    return notes