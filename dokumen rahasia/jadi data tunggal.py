import re
import csv

def extract_between(text, left, right):
    pattern = re.escape(left) + r"(.*?)" + re.escape(right)
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

def extract_number_after(text, keyword):
    pattern = re.escape(keyword) + r".*?([\d\.]+)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""

def extract_roman_after(text, keyword):
    """Ambil angka romawi pertama setelah keyword."""
    pattern = re.escape(keyword) + r".*?([IVXLCDM]+)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""

def process_file(input_path, output_csv):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text = "".join(lines)

    # ---------------------------------------
    # ship_name & ship_id (line 9)
    # ---------------------------------------
    line9 = lines[8].strip() if len(lines) >= 9 else ""
    ship_name_raw = line9.split('-')[0].strip() if '-' in line9 else line9
    ship_name = ship_name_raw
    ship_id = ship_name_raw.replace(" ", "_")

    # ---------------------------------------
    # country (line 193 → inside |...|)
    # ---------------------------------------
    line193 = lines[192] if len(lines) >= 193 else ""
    country = extract_between(line193, "|", "|")

    # ---------------------------------------
    # maximum_dispersion
    # ---------------------------------------
    maximum_dispersion = extract_number_after(text, "Maximum Dispersion")

    # ---------------------------------------
    # tier → roman numeral after first 'Tier'
    # ---------------------------------------
    tier = extract_roman_after(text, "Tier")

    # ---------------------------------------
    # firing_range
    # ---------------------------------------
    firing_range = extract_number_after(text, "Firing Range")

    # ---------------------------------------
    # initial_he_velocity (from Firing Range)
    # ---------------------------------------
    initial_he_velocity = extract_number_after(text, "Firing Range")

    # ---------------------------------------
    # initial_ap_velocity
    # ---------------------------------------
    initial_ap_velocity = extract_number_after(text, "Initial HE Shell Velocity")

    # ---------------------------------------
    # maximum_speed (from ManeuverabilityMaximum)
    # ---------------------------------------
    maximum_speed = extract_number_after(text, "ManeuverabilityMaximum")

    # ---------------------------------------
    # Save to CSV
    # ---------------------------------------
    headers = [
        "ship_name", "ship_id", "country", "maximum_dispersion",
        "tier", "firing_range", "initial_he_velocity",
        "initial_ap_velocity", "maximum_speed"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow({
            "ship_name": ship_name,
            "ship_id": ship_id,
            "country": country,
            "maximum_dispersion": maximum_dispersion,
            "tier": tier,
            "firing_range": firing_range,
            "initial_he_velocity": initial_he_velocity,
            "initial_ap_velocity": initial_ap_velocity,
            "maximum_speed": maximum_speed
        })

    print(f"Done! CSV saved to: {output_csv}")


# ---- Run ----
process_file("ship.txt", "ships.csv")
