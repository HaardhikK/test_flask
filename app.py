from flask import Flask, request, jsonify
import time
import io
import os
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from environment variables
GPT4_API_KEY = os.getenv('GPT4_API_KEY')

# Validate required environment variables
if not GPT4_API_KEY:
    raise ValueError("Missing required environment variable: GPT4_API_KEY")

class IECScraperException(Exception):
    """Custom exception for IEC scraping errors"""
    pass

def create_driver():
    """Create a Chrome WebDriver instance configured for Render deployment"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-extensions')
    options.add_argument('--single-process')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--disable-setuid-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.page_load_strategy = 'eager'
    
    try:
        # Use ChromeDriverManager to automatically handle driver installation
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        raise IECScraperException(f"Failed to create Chrome driver: {str(e)}")

def capture_captcha_section(driver):
    try:
        captcha_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="captcha"]'))
        )
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", captcha_element)
        
        pixel_ratio = driver.execute_script("return window.devicePixelRatio;")
        location = captcha_element.location_once_scrolled_into_view
        size = captcha_element.size
        
        png = driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(png))
        
        padding = 5
        left = int(location['x'] * pixel_ratio) - padding
        top = int(location['y'] * pixel_ratio) - padding
        right = int((location['x'] + size['width']) * pixel_ratio) + padding
        bottom = int((location['y'] + size['height']) * pixel_ratio) + padding
        
        captcha_image = screenshot.crop((left, top, right, bottom))
        
        if captcha_image.size[0] < 100:
            new_width = 200
            ratio = new_width / captcha_image.size[0]
            new_height = int(captcha_image.size[1] * ratio)
            captcha_image = captcha_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return captcha_image
    except TimeoutException:
        return None

def solve_captcha_with_gpt4(captcha_image):
    if not captcha_image:
        return None
        
    buffered = io.BytesIO()
    captcha_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT4_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract and return only the text from this captcha image. Return only the 4+ alphanumeric characters, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10
    }
    
    try:
        with requests.Session() as session:
            response = session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise IECScraperException(f"GPT-4 API error: {str(e)}")

def extract_table_data_with_bs4(driver, table_id, delimiter=";"):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {'id': table_id})
    
    if not table:
        return ""
    
    rows = []
    
    # Extract headers
    thead = table.find('thead')
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all('th')]
        if any(headers):
            rows.append(delimiter.join(headers))
    
    # Extract rows
    tbody = table.find('tbody')
    if tbody:
        for row in tbody.find_all('tr'):
            cols = row.find_all('td')
            row_data = [' '.join(col.get_text(strip=True).split()) for col in cols]
            if any(row_data):
                rows.append(delimiter.join(row_data))
    
    return rows

def extract_table_data_with_pagination(driver, table_id, next_button_id, delimiter=";"):
    """Extract table data with pagination support and return as newline-separated string"""
    all_rows = []
    wait = WebDriverWait(driver, 10)
    page = 1
    
    while True:
        try:
            # Wait for table to be visible and loaded
            wait.until(EC.presence_of_element_located((By.ID, table_id)))
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, f"#{table_id} tbody tr")) > 0)
            
            # Extract current page data
            current_page_rows = extract_table_data_with_bs4(driver, table_id, delimiter)
            
            # For first page, include headers; for subsequent pages, skip headers
            if page == 1:
                all_rows.extend(current_page_rows)
            else:
                all_rows.extend(current_page_rows[1:] if current_page_rows else [])
            
            # Check for next button
            try:
                next_button = wait.until(EC.presence_of_element_located((By.ID, next_button_id)))
                if "disabled" in next_button.get_attribute("class"):
                    break
                    
                # Click next button and wait for table update
                next_button.click()
                time.sleep(1)  # Short delay to allow table to update
                page += 1
                
            except (NoSuchElementException, TimeoutException):
                break
                
        except Exception as e:
            print(f"Error on page {page}: {str(e)}")
            break
    
    # Join all rows with newline character
    return "\n".join(all_rows)

def extract_iec_details(driver):
    details = []
    wait = WebDriverWait(driver, 10)
    
    try:
        form_groups = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "form-group")))
        for form_group in form_groups:
            try:
                label = form_group.find_element(By.TAG_NAME, "label").text.strip()
                value = form_group.find_element(By.TAG_NAME, "p").text.strip()
                if label and value:
                    details.append(f"{label};{value}")
            except NoSuchElementException:
                continue
    except Exception as e:
        raise IECScraperException(f"Error extracting IEC details: {str(e)}")
    
    return "\n".join(details)

def handle_captcha_submission(driver):
    max_attempts = 5
    attempt = 0
    wait = WebDriverWait(driver, 3)
    
    while attempt < max_attempts:
        try:
            captcha_image = capture_captcha_section(driver)
            captcha_text = solve_captcha_with_gpt4(captcha_image)
            
            if not captcha_text:
                attempt += 1
                continue
            
            captcha_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="txt_Captcha"]')))
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)
            
            submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='View IEC']")))
            submit_button.click()
            
            try:
                success = wait.until(EC.presence_of_element_located(
                    (By.XPATH, '/html/body/div[2]/div[9]/div/div/div[1]/div/div/div[1]/div[1]/h6')))
                if "IEC Details" in success.text:
                    return True
            except TimeoutException:
                pass
            
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {str(e)}")
        
        attempt += 1
    
    return False

@app.route('/get_iec_details', methods=['POST'])
def get_iec_details():
    try:
        data = request.get_json()
        if not data or 'iec_code' not in data or 'name' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: iec_code and name'
            }), 400

        iec_code = data['iec_code']
        name = data['name']
        
        driver = None
        try:
            driver = create_driver()
            wait = WebDriverWait(driver, 5)
            
            driver.get("https://dgft.gov.in/CP/?opt=view-any-ice")
            
            view_any_iec_button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "/html/body/div[2]/div[9]/div[3]/div/div[2]/div[1]/div/a")))
            view_any_iec_button.click()
            
            iec_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="iecNo"]')))
            name_input = driver.find_element(By.XPATH, '//*[@id="entity"]')
            
            iec_input.send_keys(iec_code)
            name_input.send_keys(name)
            
            if not handle_captcha_submission(driver):
                return jsonify({
                    'success': False,
                    'error': 'Failed to solve captcha'
                }), 400
            
            iec_details = extract_iec_details(driver)
            branch_details = extract_table_data_with_pagination(driver, "branchTable", "branchTable_next")
            
            return jsonify({
                'success': True,
                'details': {
                    'iec_details': iec_details,
                    'branch_details': branch_details
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        finally:
            if driver:
                driver.quit()
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
