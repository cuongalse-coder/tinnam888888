"""
TinNam Scraper - Selenium-based scraper for ketquadientoan.com
Scrapes all historical results for Mega 6/45 and Power 6/55.
"""
import time
import re
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from scraper.data_manager import insert_mega645, insert_power655, get_count

MEGA_URL = "https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html"
POWER_URL = "https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html"


def create_driver():
    """Create Chrome WebDriver with headless option."""
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--log-level=3')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception:
        # Fallback: try without webdriver-manager
        driver = webdriver.Chrome(options=options)
    
    driver.implicitly_wait(10)
    return driver


def parse_date(date_str):
    """Parse Vietnamese date string to YYYY-MM-DD format.
    Input: 'T5, 12/03/2026' or '12/03/2026' or similar
    """
    # Extract dd/mm/yyyy pattern
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return date_str


def scrape_mega645(driver=None):
    """Scrape all Mega 6/45 results."""
    own_driver = driver is None
    if own_driver:
        driver = create_driver()
    
    print("[Scraper] Loading Mega 6/45 page...")
    driver.get(MEGA_URL)
    time.sleep(3)
    
    try:
        # Set date range to get all historical data
        date_from = driver.find_element(By.CSS_SELECTOR, '#datef')
        date_to = driver.find_element(By.CSS_SELECTOR, '#datet')
        
        # Clear and set from date to a very early date
        driver.execute_script("arguments[0].value = '01-07-2016'", date_from)
        time.sleep(0.5)
        
        # Click search button
        btn = driver.find_element(By.CSS_SELECTOR, '.btnxemthongke')
        btn.click()
        
        print("[Scraper] Waiting for Mega 6/45 data to load...")
        time.sleep(5)
        
        # Wait for table to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table tbody tr'))
        )
        time.sleep(2)
        
        # Parse table rows
        rows_data = driver.execute_script('''
            var results = [];
            var rows = document.querySelectorAll('table.table tbody tr, table.table-striped tbody tr');
            if (rows.length === 0) {
                rows = document.querySelectorAll('table tbody tr');
            }
            for (var i = 0; i < rows.length; i++) {
                var tds = rows[i].querySelectorAll('td');
                if (tds.length >= 2) {
                    var dateText = tds[0].textContent.trim();
                    var balls = tds[1].querySelectorAll('span.home-mini-whiteball');
                    var numbers = [];
                    for (var j = 0; j < balls.length; j++) {
                        var num = parseInt(balls[j].textContent.trim());
                        if (!isNaN(num)) numbers.push(num);
                    }
                    var jackpot = tds.length >= 3 ? tds[2].textContent.trim() : '';
                    if (numbers.length >= 6) {
                        results.push({
                            date: dateText,
                            numbers: numbers.slice(0, 6),
                            jackpot: jackpot
                        });
                    }
                }
            }
            return results;
        ''')
        
        print(f"[Scraper] Found {len(rows_data)} Mega 6/45 draws")
        
        # Convert to tuples for database
        db_rows = []
        for r in rows_data:
            date = parse_date(r['date'])
            nums = r['numbers']
            jackpot = r['jackpot']
            if len(nums) >= 6:
                db_rows.append((date, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], jackpot))
        
        if db_rows:
            insert_mega645(db_rows)
        
    except Exception as e:
        print(f"[Scraper] Error scraping Mega 6/45: {e}")
        import traceback
        traceback.print_exc()
    
    if own_driver:
        driver.quit()
    
    return get_count('mega')


def scrape_power655(driver=None):
    """Scrape all Power 6/55 results."""
    own_driver = driver is None
    if own_driver:
        driver = create_driver()
    
    print("[Scraper] Loading Power 6/55 page...")
    driver.get(POWER_URL)
    time.sleep(3)
    
    try:
        # Set date range
        date_from = driver.find_element(By.CSS_SELECTOR, '#datef')
        driver.execute_script("arguments[0].value = '01-01-2017'", date_from)
        time.sleep(0.5)
        
        btn = driver.find_element(By.CSS_SELECTOR, '.btnxemthongke')
        btn.click()
        
        print("[Scraper] Waiting for Power 6/55 data to load...")
        time.sleep(5)
        
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table tbody tr'))
        )
        time.sleep(2)
        
        # Parse table rows - Power has bonus ball with class 'jphu'
        rows_data = driver.execute_script('''
            var results = [];
            var rows = document.querySelectorAll('table.table tbody tr, table.table-striped tbody tr');
            if (rows.length === 0) {
                rows = document.querySelectorAll('table tbody tr');
            }
            for (var i = 0; i < rows.length; i++) {
                var tds = rows[i].querySelectorAll('td');
                if (tds.length >= 2) {
                    var dateText = tds[0].textContent.trim();
                    var regularBalls = tds[1].querySelectorAll('span.home-mini-whiteball:not(.jphu)');
                    var bonusBall = tds[1].querySelector('span.home-mini-whiteball.jphu');
                    
                    var numbers = [];
                    for (var j = 0; j < regularBalls.length; j++) {
                        var num = parseInt(regularBalls[j].textContent.trim());
                        if (!isNaN(num)) numbers.push(num);
                    }
                    var bonus = bonusBall ? parseInt(bonusBall.textContent.trim()) : 0;
                    var jackpot = tds.length >= 3 ? tds[2].textContent.trim() : '';
                    
                    if (numbers.length >= 6) {
                        results.push({
                            date: dateText,
                            numbers: numbers.slice(0, 6),
                            bonus: bonus,
                            jackpot: jackpot
                        });
                    }
                }
            }
            return results;
        ''')
        
        print(f"[Scraper] Found {len(rows_data)} Power 6/55 draws")
        
        db_rows = []
        for r in rows_data:
            date = parse_date(r['date'])
            nums = r['numbers']
            bonus = r['bonus']
            jackpot = r['jackpot']
            if len(nums) >= 6:
                db_rows.append((date, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], bonus, jackpot))
        
        if db_rows:
            insert_power655(db_rows)
        
    except Exception as e:
        print(f"[Scraper] Error scraping Power 6/55: {e}")
        import traceback
        traceback.print_exc()
    
    if own_driver:
        driver.quit()
    
    return get_count('power')


def scrape_all():
    """Scrape both lottery types."""
    print("=" * 60)
    print("  TINNAM DATA SCRAPER")
    print("  Source: ketquadientoan.com")
    print("=" * 60)
    
    driver = create_driver()
    
    try:
        mega_count = scrape_mega645(driver)
        print(f"[Result] Mega 6/45: {mega_count} total draws in database")
        
        power_count = scrape_power655(driver)
        print(f"[Result] Power 6/55: {power_count} total draws in database")
    finally:
        driver.quit()
    
    print("=" * 60)
    print("  SCRAPING COMPLETE!")
    print(f"  Mega 6/45: {mega_count} draws")
    print(f"  Power 6/55: {power_count} draws")
    print("=" * 60)


if __name__ == '__main__':
    scrape_all()
