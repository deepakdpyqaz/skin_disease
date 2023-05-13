# IMPORTS
import gdown
import os
from zipfile import ZipFile
import shutil
from pathlib import Path
import json

settings = json.load(open("settings.json"))


# Downloading file
id = settings["data_id"]
output = "data.zip"
gdown.download(id=id, output=output, quiet=False)

print("Data downloaded successfully!")
# Extracting base file
with ZipFile("data.zip") as zip:
    zip.extractall("temp")

print("Base file extracted successfully!")
# Delete base file
os.remove("data.zip")

# Extracting subdirectories
for f in Path("temp").glob("**/**/*.zip"):
    with ZipFile(f) as zip:
        zip.extractall()
print("Subdirectories extracted successfully!")
# Deleting subdirectories
shutil.rmtree("temp")
print("Data setup complete!")

cred_id = settings["cred_id"]
cred_output = ".credentials.zip"
gdown.download(id=cred_id, output=cred_output, quiet=False)
with ZipFile(".credentials.zip") as zip:
    zip.extractall(pwd=input("Enter Password: ").encode())
os.remove(".credentials.zip")
print("Credentials setup complete!")
