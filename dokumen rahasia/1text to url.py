import re

input_file = "nama kapal.txt"
output_file = "dokumen rahasia\\urls.txt"

URL_PREFIX = "https://wiki.wargaming.net/en/Ship:"

def clean_name(text):
    text = text.replace("[", "").replace("]", "").replace("★", "").strip()
    text = text.replace("Doubloons", "").strip()             # hapus Doubloons

    # ❗hapus teks dalam tanda kurung, termasuk (< 06.03.2017)
    text = re.sub(r"\(.*?\)", "", text).strip()

    # spasi → underscore
    text = re.sub(r"\s+", "_", text)

    return text

hasil = []
sudah_ada = set()

def tambah(nama):
    nama = clean_name(nama)
    if nama not in sudah_ada:
        sudah_ada.add(nama)
        hasil.append(URL_PREFIX + nama)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # SKIP baris yang mengandung .png
        if ".png" in line:
            continue

        # 1. link=Ship:[Nama]
        match = re.search(r"link=Ship:\[?([^]\|]+)\]?", line)
        if match:
            tambah(match.group(1))
            continue

        # 2. [[Ship:...|Nama]]
        match = re.search(r"\[\[Ship:[^\|]+\|([^\]]+)\]\]", line)
        if match:
            tambah(match.group(1))
            continue

        # 3. Tier I/II/X + Nama
        match = re.search(r"^(★|[IVX]+)\s+(.+)$", line)
        if match:
            tambah(match.group(2))
            continue

        # 4. Uppercase tier (VIII, X, VI)
        match = re.search(r"^[A-Z]+\s+(.+)$", line)
        if match:
            tambah(match.group(1))
            continue

with open(output_file, "w", encoding="utf-8") as f:
    for item in hasil:
        f.write(item + "\n")

print("Selesai! Cek hasil.txt")
