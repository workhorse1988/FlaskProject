import json


def process_discrepancies_to_json(input_file, output_file):
    # Store the resulting discrepancies
    discrepancies = []

    # Initialize placeholders for each entry
    entry = {
        "DeficiencyType": "",
        "Reference": "",
        "CorrectiveAction": ""
    }

    # Read the .txt file as a list of lines
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Iterate through all lines
    current_field = None  # Tracks whether we're in DeficiencyType / Reference / CorrectiveAction

    for line in lines:
        # Strip whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Detect DeficiencyType
        if line.startswith("Deficiency Details:"):
            # If entry is already populated, save to list and start a new block
            if any(entry.values()):
                discrepancies.append(entry)
                entry = {  # Reset for the new entry
                    "DeficiencyType": "",
                    "Reference": "",
                    "CorrectiveAction": ""
                }
            # Start a new DeficiencyType
            entry["DeficiencyType"] = line.split(":", 1)[-1].strip()
            current_field = "DeficiencyType"
            continue

        # Detect Reference
        if line.startswith("Code References:"):
            entry["Reference"] = line.split(":", 1)[-1].strip()
            current_field = "Reference"
            continue

        # Detect CorrectiveAction
        if line.startswith("Corrective Actions:"):
            entry["CorrectiveAction"] = line.split(":", 1)[-1].strip()
            current_field = "CorrectiveAction"
            continue

        # Handle multi-line text for the currently active field
        if current_field and line:
            entry[current_field] += " " + line.strip()

    # Add the last entry if any after finishing iteration
    if any(entry.values()):
        discrepancies.append(entry)

    # Write results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(discrepancies, json_file, indent=4, ensure_ascii=False)

    print(f"JSON successfully created with {len(discrepancies)} entries at: {output_file}")