import requests

url = "https://monsterhunter.fandom.com/wiki/Cephadrome"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com",
}

resp = requests.get(url, headers=headers)

print("Status code:", resp.status_code)
print("Preview of what was returned:\n")
print(resp.text[:1000])  # print first 1000 characters of the page
