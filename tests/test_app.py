import pytest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from extensions import db
from models import User, Dataset

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'DataMarket' in response.data

def test_register_login(client):
    # Register
    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'password'
    }, follow_redirects=True)
    assert response.status_code == 200
    
    # Login
    response = client.post('/login', data={
        'email': 'test@example.com',
        'password': 'password'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Logout' in response.data

def test_dataset_upload_access(client):
    # Register seller (now just a user)
    client.post('/register', data={
        'username': 'seller',
        'email': 'seller@example.com',
        'password': 'password'
    }, follow_redirects=True)
    
    # Login as seller
    client.post('/login', data={
        'email': 'seller@example.com',
        'password': 'password'
    }, follow_redirects=True)
    
    # Upload requires file, mocking might be needed for full integration test
    # For now, check if upload page loads
    response = client.get('/upload')
    assert response.status_code == 200
    assert b'Upload New Dataset' in response.data
