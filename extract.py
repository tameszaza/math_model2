from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from webdriver_manager.chrome import ChromeDriverManager

# Function to scrape a single page
def scrape_page(driver: WebDriver, writer):
    rows = driver.find_elements(By.CSS_SELECTOR, "tbody[role='rowgroup'] tr")
    print(f"Scraping {len(rows)} rows on this page...")

    for row in rows:
        try:
            name = row.find_element(By.CSS_SELECTOR, ".competitor-long-name").text
            nationality = row.find_element(By.CSS_SELECTOR, "div.wrsNoc").text.split('\n')[-1]
            sport = row.find_element(By.CSS_SELECTOR, ".discipline-sport span.text-start").text
            writer.writerow({'Name': name, 'Nationality': nationality, 'Sport': sport})
            print(f"Recorded: {name}, {nationality}, {sport}")
        except Exception as e:
            print(f"Error processing row data: {e}")

# Set up WebDriver
try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    print("WebDriver set up successfully.")
except Exception as e:
    print(f"WebDriver setup error: {e}")
    exit()

url = 'https://olympics.com/en/paris-2024/athletes'
csv_filename = 'athletes.csv'

try:
    driver.get(url)
    print(f"Page loaded successfully: {url}")
    time.sleep(5)  # Adjust this wait as needed

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Nationality', 'Sport']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            # Scrape data from the current page
            scrape_page(driver, writer)

            # Check if the "Next" button is available
            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='next page']"))
                )
                next_button.click()
                time.sleep(3)  # Wait for the next page to load
            except Exception as e:
                print(f"Pagination error or no more pages: {e}")
                break

except Exception as e:
    print(f"Error during scraping: {e}")
finally:
    driver.quit()
    print(f"WebDriver closed, and data saved to {csv_filename}.")
