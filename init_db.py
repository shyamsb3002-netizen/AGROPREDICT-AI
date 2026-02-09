from app import app
from extensions import db
from utils.db_models import User, UserAdmin, ContactUs

with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
