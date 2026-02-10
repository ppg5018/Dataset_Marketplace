from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from extensions import db
from models import Dataset, Transaction, Review
import os
import spark_utils

marketplace_bp = Blueprint('marketplace', __name__)

@marketplace_bp.route('/')
def index():
    return render_template('home.html')

@marketplace_bp.route('/browse')
def browse():
    datasets = Dataset.query.all()
    return render_template('index.html', datasets=datasets)

@marketplace_bp.route('/dashboard')
@login_required
def dashboard():
    # Unified dashboard for all users
    my_datasets = Dataset.query.filter_by(owner_id=current_user.id).all()
    purchased = Transaction.query.filter_by(buyer_id=current_user.id).all()
    return render_template('dashboard.html', my_datasets=my_datasets, purchased=purchased)



@marketplace_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset():
    # Allow all users to upload
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        price = float(request.form.get('price'))
        file = request.files['file']
        
        if file:
            filename = secure_filename(file.filename)
            # Save to uploads folder
            from flask import current_app
            save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            new_dataset = Dataset(owner_id=current_user.id, title=title, 
                                  description=description, price=price, 
                                  file_path=save_path)
            db.session.add(new_dataset)
            db.session.commit()
            
            # Auto-calculate quality score
            try:
                quality_data = spark_utils.calculate_quality_score(save_path)
                new_dataset.quality_score = quality_data.get('overall_score', 0)
                db.session.commit()
            except Exception as e:
                print(f"Error calculating quality score: {e}")
            
            flash('Dataset uploaded successfully')
            return redirect(url_for('marketplace.dashboard'))
            
    return render_template('upload.html')

@marketplace_bp.route('/dataset/<int:dataset_id>')
def dataset_detail(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access (is owner or bought it)
    has_access = False
    if current_user.is_authenticated:
        if dataset.owner_id == current_user.id:
            has_access = True
        elif Transaction.query.filter_by(buyer_id=current_user.id, dataset_id=dataset.id).first():
            has_access = True
        elif dataset.price == 0:
            has_access = True
            
    # Increment view count
    from extensions import db
    dataset.view_count = (dataset.view_count or 0) + 1
    db.session.commit()
            
    # Get basic stats if file exists
    stats = None
    if os.path.exists(dataset.file_path):
        try:
            stats = spark_utils.get_dataset_stats(dataset.file_path)
        except Exception as e:
            print(f"Error getting stats: {e}")
            
    return render_template('dataset_detail.html', dataset=dataset, has_access=has_access, stats=stats)

@marketplace_bp.route('/buy/<int:dataset_id>', methods=['POST'])
@login_required
def buy_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Simulate payment
    transaction = Transaction(buyer_id=current_user.id, dataset_id=dataset.id, amount=dataset.price)
    db.session.add(transaction)
    db.session.commit()
    
    flash('Dataset purchased successfully!')
    return redirect(url_for('marketplace.dataset_detail', dataset_id=dataset.id))

@marketplace_bp.route('/dataset/<int:dataset_id>/review', methods=['POST'])
@login_required
def submit_review(dataset_id):
    from flask import request, jsonify
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access and is not the owner
    if dataset.owner_id == current_user.id:
        return jsonify({'error': 'Cannot review your own dataset'}), 403
    
    has_access = False
    if Transaction.query.filter_by(buyer_id=current_user.id, dataset_id=dataset.id).first():
        has_access = True
    elif dataset.price == 0:
        has_access = True
        
    if not has_access:
        return jsonify({'error': 'You must purchase the dataset first'}), 403
    
    # Check if user already reviewed
    existing_review = Review.query.filter_by(user_id=current_user.id, dataset_id=dataset_id).first()
    if existing_review:
        return jsonify({'error': 'You have already reviewed this dataset'}), 400
    
    data = request.json
    rating = data.get('rating')
    comment = data.get('comment', '')
    
    if not rating or rating < 1 or rating > 5:
        return jsonify({'error': 'Invalid rating'}), 400
    
    review = Review(dataset_id=dataset_id, user_id=current_user.id, rating=rating, comment=comment)
    db.session.add(review)
    db.session.commit()
    
    return jsonify({'success': True})

@marketplace_bp.route('/seller/<username>')
def seller_profile(username):
    from models import User
    seller = User.query.filter_by(username=username).first_or_404()
    datasets = Dataset.query.filter_by(owner_id=seller.id).all()
    
    # Calculate average rating
    total_ratings = 0
    total_reviews = 0
    for dataset in datasets:
        for review in dataset.reviews:
            total_ratings += review.rating
            total_reviews += 1
    
    avg_rating = total_ratings / total_reviews if total_reviews > 0 else 0
    
    return render_template('seller_profile.html', seller=seller, datasets=datasets, avg_rating=avg_rating, total_reviews=total_reviews)

@marketplace_bp.route('/analytics')
@login_required
def analytics():
    # Show analytics based on role
    if current_user.role == 'seller':
        my_datasets = Dataset.query.filter_by(owner_id=current_user.id).all()
        total_downloads = sum(d.download_count for d in my_datasets)
        total_views = sum(d.view_count for d in my_datasets)
        total_queries = sum(d.query_count for d in my_datasets)
        total_revenue = sum(d.price * d.download_count for d in my_datasets)
        
        return render_template('analytics.html', 
                             datasets=my_datasets,
                             total_downloads=total_downloads,
                             total_views=total_views,
                             total_queries=total_queries,
                             total_revenue=total_revenue)
    else:
        # For buyers, show popular datasets
        popular_datasets = Dataset.query.order_by(Dataset.view_count.desc()).limit(10).all()
        return render_template('analytics.html', 
                             popular_datasets=popular_datasets,
                             is_buyer=True)
