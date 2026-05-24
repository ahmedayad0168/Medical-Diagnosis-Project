import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from urllib.parse import urljoin
from src.utils.helpers import save_to_csv, save_to_mongo

class MayoClinicScraper:
    BASE_URL = "https://www.mayoclinic.org"
    INDEX_URL = f"{BASE_URL}/diseases-conditions"

    def __init__(self, delay=1, use_mongo=False):
        self.delay = delay
        self.use_mongo = use_mongo
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_letter_links(self):
        """Get all letter index URLs (A-Z, #)"""
        soup = self._get_soup(self.INDEX_URL)
        letter_links = []
        for a in soup.find_all("a", href=True):
            if "index?letter=" in a["href"]:
                full = urljoin(self.BASE_URL, a["href"])
                letter_links.append(full)
        return letter_links

    def get_disease_links(self, letter_url):
        """Extract disease detail URLs from a letter page"""
        soup = self._get_soup(letter_url)
        links = []
        for a in soup.find_all("a", class_="cmp-anchor--plain cmp-button cmp-button__link cmp-results-with-primary-name__see-link"):
            links.append(urljoin(self.BASE_URL, a["href"]))
        return list(set(links))  # remove duplicates

    def scrape_disease_page(self, url):
        """Extract sections: Overview, Symptoms, Causes, Risk factors"""
        soup = self._get_soup(url)
        disease_name = soup.find("h1").get_text(strip=True) if soup.find("h1") else "Unknown"

        sections = {"overview": "", "symptoms": "", "causes": "", "risk factors": ""}
        wanted = list(sections.keys())

        for h2 in soup.find_all("h2"):
            title = h2.get_text(strip=True).lower()
            if title in wanted:
                content = []
                for sibling in h2.find_next_siblings():
                    if sibling.name == "h2":
                        break
                    content.append(sibling.get_text(" ", strip=True))
                sections[title] = " ".join(content).strip()

        return {
            "disease_name": disease_name,
            "url": url,
            "sections": sections,
            "scraped_at": time.time()
        }

    def _get_soup(self, url):
        resp = self.session.get(url)
        resp.raise_for_status()
        time.sleep(self.delay)
        return BeautifulSoup(resp.text, "html.parser")

    def run(self, output_csv=None):
        all_diseases = []
        letter_links = self.get_letter_links()
        for letter_url in tqdm(letter_links, desc="Letters"):
            disease_links = self.get_disease_links(letter_url)
            for dl in tqdm(disease_links, desc="Diseases", leave=False):
                try:
                    data = self.scrape_disease_page(dl)
                    all_diseases.append(data)
                    if self.use_mongo:
                        save_to_mongo(data)
                except Exception as e:
                    print(f"Failed {dl}: {e}")
                time.sleep(self.delay)

        if output_csv:
            save_to_csv(all_diseases, output_csv)
        return all_diseases