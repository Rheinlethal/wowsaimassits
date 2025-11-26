import requests
from bs4 import BeautifulSoup
import csv
import os
import re

# ==========================================================
#  FUNGSI EKSTRAK DATA
# ==========================================================

def extract_value(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def clean_spaces(text):
    return re.sub(r"\s+", "_", text.strip())

# ==========================================================
#  PROSES SEBUAH URL
# ==========================================================

def process_url(url):
    print(f"Scraping: {url}")

    try:
        res = requests.get(url, timeout=15)
    except Exception as e:
        print("Gagal request:", e)
        return None

    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # ======================================================
    # ship_name
    # aturan:
    # - ambil kata sebelum '-' pertama
    # - jika huruf awal Z,I,S,U → ambil sebelum '-' kedua
    # ======================================================

    # contoh URL format: https://.../Ship:USS_Somers-1944-Refit
    raw_name = url.split("Ship:")[-1]

    parts = raw_name.split("-")

    # aturan khusus
    if raw_name[0] in ("Z", "I", "S", "U") and len(parts) > 2:
        ship_name = parts[0] + "-" + parts[1]
    else:
        ship_name = parts[0]

    ship_name = ship_name.strip()

    # ======================================================
    # ship_id
    # ambil sebelum '-' pertama, ubah spasi jadi _
    # ======================================================
    ship_id = clean_spaces(parts[0])

    # ======================================================
    # country
    # ambil teks antara | ... |
    # ======================================================
    country = extract_value(r"\|([^|]+)\|", text)

    # ======================================================
    # maximum dispersion
    # ======================================================
    maximum_dispersion = extract_value(r"Maximum Dispersion[^0-9]*([0-9]+)", text)

    # ======================================================
    # firing range
    # ======================================================
    firing_range = extract_value(r"Firing Range[^0-9]*([0-9]+)", text)

    # ======================================================
    # initial velocities (HE / AP)
    # ======================================================
    initial_he_velocity = extract_value(r"Initial HE Shell Velocity[^0-9]*([0-9]+)", text)
    initial_ap_velocity = extract_value(r"Initial AP Shell Velocity[^0-9]*([0-9]+)", text)

    # ======================================================
    # maximum speed
    # ======================================================
    maximum_speed = extract_value(r"Maximum Speed[^0-9]*([0-9]+)", text)

    if not maximum_speed:  # fallback tanpa spasi
        maximum_speed = extract_value(r"MaximumSpeed[^0-9]*([0-9]+)", text)

    return [
        ship_name,
        ship_id,
        country,
        maximum_dispersion,
        firing_range,
        initial_he_velocity,
        initial_ap_velocity,
        maximum_speed
    ]

# ==========================================================
#  MAIN SCRIPT
# ==========================================================

urls_file = r"splitted\urls__3.txt"
csv_file = "ship_data3.csv"

# buat CSV kalau belum ada
file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # header kalau file baru
    if not file_exists:
        writer.writerow([
            "ship_name", "ship_id", "country",
            "maximum_dispersion", "firing_range",
            "initial_he_velocity", "initial_ap_velocity",
            "maximum_speed"
        ])

    # baca URL dari file
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [x.strip() for x in f.readlines() if x.strip()]

    for url in urls:
        data = process_url(url)
        if data:
            writer.writerow(data)
        else:
            print("❌ Gagal extract")

print("\nSelesai bro! Cek file: ship_data.csv")
