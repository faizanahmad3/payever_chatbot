from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
import requests
import os
import undetected_chromedriver as uc


def initialize_driver():
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options)
    return driver


def get_soup(driver, url):
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    return BeautifulSoup(driver.page_source, 'html.parser')


def get_links(soup, base_url):
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/'):
            # Ensure the link is constructed with the base URL
            links.append(base_url + href)
        elif href.startswith(base_url):
            links.append(href)
        else:
            # Ignore external or malformed links
            continue
    return list(set(links))


def scrape_content(soup):
    base_url = "https://support.payever.org"
    text = soup.get_text(separator=' ', strip=True)
    images = [base_url + img['src'] for img in soup.find_all('img', src=True)]
    return text, images


def download_image(img_url, folder):
    try:
        response = requests.get(img_url)
        if response.status_code == 200:
            filename = os.path.join(folder, img_url.split('/')[-1] + ".png")
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
    return None


def crawl_and_scrape(driver, base_url, visited=None, data=None, max_depth=3, current_depth=0):
    if visited is None:
        visited = set()
    if data is None:
        data = []

    if current_depth > max_depth:
        return data

    if base_url in visited:
        return data

    visited.add(base_url)
    print(f"Visiting: {base_url}")

    try:
        soup = get_soup(driver, base_url)

        # Check for captcha
        if "captcha" in soup.title.string.lower():
            print("Captcha detected. Please solve it manually.")
            input("Press Enter after solving the captcha...")

            # Retry getting the soup after captcha is solved
            soup = get_soup(driver, base_url)

        text, images = scrape_content(soup)

        # Create a folder for images
        folder_name = f"images_{base_url.split('/')[-1]}"
        folder_name = "images_of_articles/" + folder_name
        os.makedirs(folder_name, exist_ok=True)

        # Download images
        downloaded_images = [download_image(img, folder_name) for img in images]

        data.append({
            'url': base_url,
            'text': text,
            'images': downloaded_images
        })

        links = get_links(soup, 'https://support.payever.org')  # Use the base URL here
        for link in links:
            time.sleep(2)
            crawl_and_scrape(driver, link, visited, data, max_depth, current_depth + 1)

    except Exception as e:
        print(f"Error processing {base_url}: {e}")

    return data


def main():
    base_url = 'https://support.payever.org/hc/en-us'
    driver = initialize_driver()

    try:
        scraped_data = crawl_and_scrape(driver, base_url)

        # Save data to CSV
        df = pd.DataFrame(scraped_data)
        df.to_csv('scraped_data.csv', index=False)

        print("Scraping completed. Data saved to scraped_data.csv")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
