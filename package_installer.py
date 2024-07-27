import subprocess

def install(package):
    subprocess.check_call(["pip3", "install", package])

packages = ["pandas", "torch", "transformers", "nlp","imblearn","datasets","bs4","matplotlib","nltk"]

for package in packages:
    install(package)
