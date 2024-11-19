
import requests
from bs4 import BeautifulSoup
import csv
url = 'https://olympics.com/en/paris-2024/athletes'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
athletes = soup.find_all('div', class_='athlete-card')
with open('athletes.csv', 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Nationality', 'Sport']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for athlete in athletes:
        name = athlete.find('h2', class_='athlete-name').text.strip()
        nationality = athlete.find('span', class_='athlete-country').text.strip()
        sport = athlete.find('span', class_='athlete-sport').text.strip()
        writer.writerow({'Name': name, 'Nationality': nationality, 'Sport': sport})
