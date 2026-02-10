from flask import Flask
from flask_login import LoginManager
from pyspark.sql import SparkSession
import os
from .extensions import db, login_manager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'new-secret-key-2024-features' # Changed to invalidate old sessions
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///marketplace.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Initialize Spark Session
    # We create a single SparkSession for the app. 
    # In a real production app, you might manage this differently (e.g. Livy).
    global spark
    spark = SparkSession.builder \
        .appName("DatasetMarketplace") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Register Blueprints
    from .routes.auth import auth_bp
    from .routes.marketplace import marketplace_bp
    from .routes.data import data_bp
    from .routes.scraper import scraper_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(marketplace_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(scraper_bp)

    # Create database tables
    with app.app_context():
        # Import models to ensure they are registered with SQLAlchemy
        from . import models
        db.create_all()

    return app
