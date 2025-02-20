import re
from typing import Optional, Dict, Any
import pdfplumber


def parse_hydraulic_calculations(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Extracts hydraulic calculation data from PDF report.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing extracted hydraulic data or None if processing fails
    """
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
    
    if not pdf_path or not isinstance(pdf_path, str):
        print("Invalid PDF path provided")
        return None
        
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Compile regex patterns once for better performance
            patterns = {
                "flow_rate": re.compile(r'Flow Rate\s*:\s*(\d+\.?\d*)\s*gpm'),
                "pressure": re.compile(r'Pressure\s*:\s*(\d+\.?\d*)\s*psi'),
                "pipe_length": re.compile(r'Pipe Length\s*:\s*(\d+\.?\d*)\s*ft'),
                "elevation": re.compile(r'Node Elevation\s*:\s*(\d+\.?\d*)\s*ft'),
                "k_factor": re.compile(r'K-Factor\s*:\s*(\d+\.?\d*)'),
                "hose_demand": re.compile(r'Hose Demand\s*:\s*Included', re.IGNORECASE),
                "fixed_loss": re.compile(r'Fixed Loss Device\s*:\s*(.*)'),
                "static_pressure": re.compile(r'Static Pressure\s*:\s*(\d+\.?\d*)\s*psi'),
                "residual_pressure": re.compile(r'Residual Pressure\s*:\s*(\d+\.?\d*)\s*psi'),
                "flow_at_test": re.compile(r'Flow at Test\s*:\s*(\d+\.?\d*)\s*gpm')
            }
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    # Extract and convert numerical values using list comprehension
                    for key, pattern in patterns.items():
                        if key in ["flow_rate", "pressure", "pipe_length", "elevation", "k_factor"]:
                            matches = pattern.findall(text)
                            if matches:
                                extracted_hydraulic_data[f"{key}s"].extend(float(x) for x in matches)

                    # Check for hose demand
                    if patterns["hose_demand"].search(text):
                        extracted_hydraulic_data["hose_demand"] = True

                    # Extract fixed loss devices
                    fixed_losses = patterns["fixed_loss"].findall(text)
                    if fixed_losses:
                        extracted_hydraulic_data["fixed_loss_devices"].extend(fixed_losses)

                    # Extract flow test data if all required values are present
                    static_pressure = patterns["static_pressure"].search(text)
                    residual_pressure = patterns["residual_pressure"].search(text)
                    flow_at_test = patterns["flow_at_test"].search(text)
                    
                    if all([static_pressure, residual_pressure, flow_at_test]):
                        extracted_hydraulic_data["flow_test_data"] = {
                            "static_pressure": float(static_pressure.group(1)),
                            "residual_pressure": float(residual_pressure.group(1)),
                            "flow_at_test": float(flow_at_test.group(1))
                        }
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None
        
    # Validate extracted data - check if any numerical data was found
    if not any([
        extracted_hydraulic_data["flow_rates"],
        extracted_hydraulic_data["pressures"],
        extracted_hydraulic_data["pipe_lengths"],
        extracted_hydraulic_data["node_elevations"],
        extracted_hydraulic_data["k_factors"],
        extracted_hydraulic_data["flow_test_data"]
    ]):
        print("No hydraulic data was extracted from the PDF")
        return None
        
    return extracted_hydraulic_data


def run_fire_sprinkler_analysis(fire_plan_pdf, hydraulic_calculation_pdf, report_pdf_output, csv_output):
    pass


if __name__ == "__main__":
    # Example usage with proper path handling
    from pathlib import Path
    
    base_path = Path("A:/FPE")
    run_fire_sprinkler_analysis(
        fire_plan_pdf=base_path / "Pages from (SLB-FIRE25-0002) Solana Highlands MULTI-FAMILY - FIRE Sprinkler 2.pdf",
        hydraulic_calculation_pdf=base_path / "SLB FIRE25-0002 SET 1 CALCS.pdf",
        report_pdf_output=base_path / "output_report.pdf",
        csv_output=base_path / "analysis_results.csv"
    )
