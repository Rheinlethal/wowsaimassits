import pandas as pd
import os

def excel_to_csv(input_file):
    # ambil semua sheet
    xls = pd.ExcelFile(input_file)
    
    # ambil nama file tanpa extension
    base_name = os.path.splitext(input_file)[0]
    
    for sheet_name in xls.sheet_names:
        # baca tiap sheet
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        
        # bikin nama csv dari nama sheet
        safe_sheet = sheet_name.replace(" ", "_")  # type: ignore # biar ga ribut kalau ada spasi
        output_file = f"{base_name}_{safe_sheet}_DATA_.csv"
        
        # export
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved: {output_file}")

# contoh pemakaian
excel_to_csv("data tembak2.xlsx")
