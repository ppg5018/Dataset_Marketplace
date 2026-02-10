#!/usr/bin/env python
"""Reset the database with new schema"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from extensions import db

# Remove old database files
db_files = ['marketplace.db', 'marketplace.db-journal', 'marketplace.db-shm', 'marketplace.db-wal']
for f in db_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

# Create new database
app = create_app()
with app.app_context():
    db.create_all()
    print("✅ Database recreated successfully with new schema!")
    print("All tables created:")
    print("  - User (with bio column)")
    print("  - Dataset")
    print("  - Transaction")
    print("  - Review")
