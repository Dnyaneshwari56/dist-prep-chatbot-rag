import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
from datetime import datetime
from urllib.parse import urljoin, urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DisasterPrepScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.scraped_data = []
        
        # Trusted sources configuration
        self.sources = {
            'FEMA': {
                'base_url': 'https://www.fema.gov',
                'targets': [
                    '/emergency-managers/practitioners/planning',
                    '/individuals-communities/emergency-preparedness',
                    '/disaster/how-to-prepare'
                ]
            },
            'Ready.gov': {
                'base_url': 'https://www.ready.gov',
                'targets': [
                    '/plan',
                    '/kit',
                    '/informed',
                    '/involved'
                ]
            },
            'CDC': {
                'base_url': 'https://www.cdc.gov',
                'targets': [
                    '/disasters/index.html',
                    '/disasters/hurricanes/index.html',
                    '/disasters/earthquakes/index.html',
                    '/disasters/floods/index.html'
                ]
            },
            'NOAA': {
                'base_url': 'https://www.weather.gov',
                'targets': [
                    '/safety',
                    '/wrn/force-preparedness',
                    '/safety/hurricane',
                    '/safety/tornado'
                ]
            },
            'Red Cross': {
                'base_url': 'https://www.redcross.org',
                'targets': [
                    '/get-help/how-to-prepare-for-emergencies',
                    '/get-help/disaster-relief-and-recovery-services',
                    '/about-us/our-work/disaster-relief'
                ]
            },
            'WHO': {
                'base_url': 'https://www.who.int',
                'targets': [
                    '/emergencies/diseases',
                    '/emergencies/surveillance',
                    '/emergencies/preparedness'
                ]
            },
            'UNDRR': {
                'base_url': 'https://www.undrr.org',
                'targets': [
                    '/building-resilience',
                    '/reducing-disaster-risk',
                    '/understanding-disaster-risk'
                ]
            }
        }
    
    def setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # Use webdriver-manager to automatically download chromedriver
            # Force win64 architecture
            try:
                service = Service(ChromeDriverManager(os_type="win64").install())
            except:
                # Fallback to default
                service = Service(ChromeDriverManager().install())
            
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Chrome WebDriver setup successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup WebDriver: {e}")
            # Try alternative approach with requests for basic scraping
            logger.info("Attempting to use requests-based scraping as fallback")
            self.driver = None
    
    def scrape_url(self, url, source_name):
        """Scrape content from a specific URL"""
        if self.driver:
            return self.scrape_url_selenium(url, source_name)
        else:
            return self.scrape_url_requests(url, source_name)
    
    def scrape_url_selenium(self, url, source_name):
        """Scrape content using Selenium"""
        try:
            logger.info(f"Scraping {url} from {source_name} using Selenium")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page content
            html = self.driver.page_source
            return self.parse_html(html, url, source_name)
                
        except TimeoutException:
            logger.error(f"Timeout scraping {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_url_requests(self, url, source_name):
        """Scrape content using requests as fallback"""
        try:
            logger.info(f"Scraping {url} from {source_name} using requests")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return self.parse_html(response.text, url, source_name)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping {url} with requests: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def parse_html(self, html, url, source_name):
        """Parse HTML content and extract relevant information"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract main content
            content_selectors = [
                'main', '.main-content', '.content', '.article', 
                'article', '.post', '.page-content', '[role="main"]'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, get body text
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            
            # Clean up content
            content = ' '.join(content.split())
            
            if len(content) > 200:  # Only save if content is substantial
                document = {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'source': source_name,
                    'scraped_date': datetime.now().isoformat(),
                    'type': 'disaster_preparedness'
                }
                self.scraped_data.append(document)
                logger.info(f"Successfully scraped {len(content)} characters from {url}")
                return document
            else:
                logger.warning(f"Content too short from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {e}")
            return None
    
    def scrape_source(self, source_name, max_pages_per_target=3):
        """Scrape all target URLs for a specific source"""
        source_config = self.sources.get(source_name)
        if not source_config:
            logger.error(f"Unknown source: {source_name}")
            return []
        
        source_data = []
        base_url = source_config['base_url']
        
        for target_path in source_config['targets']:
            try:
                full_url = urljoin(base_url, target_path)
                document = self.scrape_url(full_url, source_name)
                if document:
                    source_data.append(document)
                
                # Add small delay between requests
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing {target_path} for {source_name}: {e}")
                continue
        
        logger.info(f"Scraped {len(source_data)} documents from {source_name}")
        return source_data
    
    def scrape_all_sources(self):
        """Scrape all configured sources"""
        try:
            self.setup_driver()
            
            for source_name in self.sources.keys():
                logger.info(f"Starting to scrape {source_name}")
                self.scrape_source(source_name)
                time.sleep(3)  # Longer delay between sources
            
            logger.info(f"Scraping completed. Total documents: {len(self.scraped_data)}")
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
        finally:
            if self.driver:
                self.driver.quit()
    
    def save_data(self, output_file="scraped_disaster_prep_data.json"):
        """Save scraped data to JSON file"""
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.scraped_data)} documents to {output_path}")
        return output_path

def main():
    """Main function to run the scraper"""
    scraper = DisasterPrepScraper(headless=True)
    
    try:
        scraper.scrape_all_sources()
        output_file = scraper.save_data()
        print(f"Scraping completed! Data saved to: {output_file}")
        print(f"Total documents scraped: {len(scraper.scraped_data)}")
        
        # Print summary by source
        sources_summary = {}
        for doc in scraper.scraped_data:
            source = doc['source']
            sources_summary[source] = sources_summary.get(source, 0) + 1
        
        print("\nDocuments by source:")
        for source, count in sources_summary.items():
            print(f"  {source}: {count} documents")
            
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        print(f"Scraping failed: {e}")

if __name__ == "__main__":
    main()
