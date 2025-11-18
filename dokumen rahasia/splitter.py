import os

# === PENGATURAN ===
input_file = "dokumen rahasia\\urls.txt"        # file sumber
lines_per_file = 172           # jumlah baris per file
output_prefix = "urls__"   # nama awal output
output_folder = "splitted"     # folder keluaran

# === BUAT FOLDER KALO BELUM ADA ===
os.makedirs(output_folder, exist_ok=True)

# === BACA SEMUA BARIS ===
with open(input_file, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

total_lines = len(lines)
file_count = (total_lines + lines_per_file - 1) // lines_per_file

print(f"Total baris: {total_lines}")
print(f"Membuat {file_count} file...")

# === SPLIT ===
for i in range(file_count):
    start = i * lines_per_file
    end = start + lines_per_file
    chunk = lines[start:end]

    output_file = os.path.join(output_folder, f"{output_prefix}{i+1}.txt")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n".join(chunk))

    print(f"File dibuat: {output_file} ({len(chunk)} baris)")

print("Selesai! 🎉")
