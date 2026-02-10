"""
Web Scraping Utilities
Functions for extracting data from websites
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse
import time

def validate_url(url):
    """Validate URL format and accessibility"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False, "Invalid URL format"
        
        # Check if URL is accessible
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code >= 400:
            return False, f"URL returned status code {response.status_code}"
        
        return True, "Valid URL"
    except Exception as e:
        return False, str(e)

def scrape_url(url, method='simple', selectors=None, timeout=30):
    """
    Scrape data from URL
    
    Args:
        url: Target URL
        method: 'simple' for static HTML, 'table' for auto-detect tables
        selectors: Dict of CSS selectors for custom extraction
        timeout: Request timeout in seconds
    
    Returns:
        dict: {
            'success': bool,
            'data': list of dicts or DataFrame,
            'columns': list of column names,
            'rows': row count,
            'error': error message if failed
        }
    """
    try:
        # Fetch page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract data based on method
        if method == 'table':
            result = extract_tables(soup)
        elif method == 'custom' and selectors:
            result = extract_custom(soup, selectors)
        else:
            result = extract_simple(soup)
        
        if result['success']:
            # Normalize data
            normalized = normalize_data(result['data'])
            return {
                'success': True,
                'data': normalized,
                'columns': list(normalized.columns) if isinstance(normalized, pd.DataFrame) else [],
                'rows': len(normalized),
                'preview': normalized.head(50).to_dict('records') if isinstance(normalized, pd.DataFrame) else normalized[:50]
            }
        else:
            return result
            
    except requests.Timeout:
        return {'success': False, 'error': 'Request timed out'}
    except requests.RequestException as e:
        return {'success': False, 'error': f'Network error: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Scraping error: {str(e)}'}

def extract_tables(soup):
    """Auto-detect and extract HTML tables"""
    tables = soup.find_all('table')
    
    if not tables:
        return {'success': False, 'error': 'No tables found on page'}
    
    all_data = []
    
    for i, table in enumerate(tables):
        try:
            # Try pandas first (handles complex tables)
            df = pd.read_html(str(table))[0]
            
            # Add table identifier
            df['_table_index'] = i
            all_data.append(df)
        except:
            # Fallback to manual parsing
            rows = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    rows.append([cell.get_text(strip=True) for cell in cells])
            
            if rows:
                df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
                df['_table_index'] = i
                all_data.append(df)
    
    if all_data:
        # Combine all tables
        combined = pd.concat(all_data, ignore_index=True)
        return {'success': True, 'data': combined}
    else:
        return {'success': False, 'error': 'Could not parse tables'}

def extract_simple(soup):
    """Extract all visible text and links"""
    # Remove script and style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Extract links
    links = []
    for a in soup.find_all('a', href=True):
        links.append({
            'text': a.get_text(strip=True),
            'url': a['href']
        })
    
    # Create simple dataset
    data = []
    for i, line in enumerate(lines[:500]):  # Limit to 500 lines
        data.append({
            'line_number': i + 1,
            'content': line
        })
    
    df = pd.DataFrame(data)
    return {'success': True, 'data': df}

def extract_custom(soup, selectors):
    """
    Extract data using custom CSS selectors
    
    Args:
        selectors: dict like {'column_name': 'css selector'}
    """
    data = []
    
    # Find maximum rows across all selectors
    max_rows = 0
    all_elements = {}
    
    for col_name, selector in selectors.items():
        elements = soup.select(selector)
        all_elements[col_name] = elements
        max_rows = max(max_rows, len(elements))
    
    # Build rows
    for i in range(max_rows):
        row = {}
        for col_name, elements in all_elements.items():
            if i < len(elements):
                row[col_name] = elements[i].get_text(strip=True)
            else:
                row[col_name] = ''
        data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        return {'success': True, 'data': df}
    else:
        return {'success': False, 'error': 'No data found with provided selectors'}

def normalize_data(data):
    """Clean and normalize scraped data"""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)
    
    # Clean column names
    df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def save_scraped_data(data, filename, format='csv'):
    """Save scraped data to file"""
    import os
    from flask import current_app
    
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    filepath = os.path.join(upload_folder, filename)
    
    if isinstance(data, pd.DataFrame):
        if format == 'csv':
            data.to_csv(filepath, index=False)
        elif format == 'json':
            data.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            data.to_excel(filepath, index=False)
    
    return filepath

def get_page_title(url):
    """Get page title for automatic dataset naming"""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'lxml')
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)[:100]  # Limit length
        return None
    except:
        return None
