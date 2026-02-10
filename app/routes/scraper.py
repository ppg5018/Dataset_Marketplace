"""
Web Scraper Routes
API endpoints for web scraping functionality
"""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app.utils import scraper as scraper_utils
from app.models import Dataset
from app.extensions import db
import os
from datetime import datetime

scraper_bp = Blueprint('scraper', __name__, url_prefix='/scraper')

@scraper_bp.route('/')
@login_required
def index():
    """Web scraping interface"""
    return render_template('scraper.html')

@scraper_bp.route('/api/scrape', methods=['POST'])
@login_required
def scrape():
    """Execute web scraping"""
    data = request.json
    url = data.get('url')
    method = data.get('method', 'table')
    selectors = data.get('selectors', {})
    
    # Validate URL
    is_valid, message = scraper_utils.validate_url(url)
    if not is_valid:
        return jsonify({'success': False, 'error': message}), 400
    
    # Scrape data
    result = scraper_utils.scrape_url(url, method=method, selectors=selectors if selectors else None)
    
    if result['success']:
        return jsonify({
            'success': True,
            'preview': result['preview'],
            'columns': result['columns'],
            'rows': result['rows']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500

@scraper_bp.route('/api/save', methods=['POST'])
@login_required
def save_dataset():
    """Save scraped data as dataset"""
    try:
        data = request.json
        url = data.get('url')
        method = data.get('method', 'table')
        selectors = data.get('selectors', {})
        title = data.get('title')
        description = data.get('description', '')
        price = float(data.get('price', 0))
        
        # Re-scrape to get full data
        result = scraper_utils.scrape_url(url, method=method, selectors=selectors if selectors else None)
        
        if not result['success']:
            return jsonify({'success': False, 'error': result.get('error')}), 500
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scraped_{timestamp}.csv"
        
        # Save file
        filepath = scraper_utils.save_scraped_data(result['data'], filename)
        
        # Create dataset record
        new_dataset = Dataset(
            owner_id=current_user.id,
            title=title,
            description=f"{description}\n\nScraped from: {url}",
            price=price,
            file_path=filepath
        )
        db.session.add(new_dataset)
        db.session.commit()
        
        # Calculate quality score
        try:
            from app.utils import spark as spark_utils
            quality_data = spark_utils.calculate_quality_score(filepath)
            new_dataset.quality_score = quality_data.get('overall_score', 0)
            db.session.commit()
        except:
            pass
        
        return jsonify({
            'success': True,
            'dataset_id': new_dataset.id,
            'message': 'Dataset created successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
