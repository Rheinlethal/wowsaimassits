import requests
from bs4 import BeautifulSoup

url = ""
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")

all_text = soup.get_text()

# simpan ke file txt
with open("ship.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("udah kelar bro, cek ship.txt 👀")
