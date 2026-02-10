from .extensions import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False) # 'buyer' or 'seller'
    bio = db.Column(db.Text, nullable=True)
    datasets = db.relationship('Dataset', backref='owner', lazy=True)
    transactions = db.relationship('Transaction', backref='buyer', lazy=True)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, default=0.0)
    file_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quality and Analytics fields
    quality_score = db.Column(db.Float, default=0.0)  # 0-100 quality score
    download_count = db.Column(db.Integer, default=0)  # Number of downloads
    query_count = db.Column(db.Integer, default=0)  # SQL queries executed
    view_count = db.Column(db.Integer, default=0)  # Detail page views
    
    transactions = db.relationship('Transaction', backref='dataset', lazy=True)
    reviews = db.relationship('Review', backref='dataset', lazy=True)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False) # 1-5
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='reviews', lazy=True)

