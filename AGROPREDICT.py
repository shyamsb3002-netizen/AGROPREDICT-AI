import os
import sqlite3
from flask import Flask, session
from extensions import db, bcrypt, login_manager
from utils.db_models import User, UserAdmin
from utils.i18n import translations

# Import Blueprints
from blueprints.main import main_bp
from blueprints.auth import auth_bp
from blueprints.prediction import prediction_bp
from blueprints.admin import admin_bp
from blueprints.api import api_bp
from blueprints.mandi import mandi_bp

def create_app():
    app = Flask(__name__)
    
    # --- Configuration ---
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "database.db")
    app.config["SECRET_KEY"] = 'thisissecretkey'
    
    # --- Initialize Extensions ---
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    
    
    # --- Register Blueprints ---
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(mandi_bp)
    
    # --- i18n Context Processor ---
    @app.context_processor
    def inject_translate():
        def t(key):
            lang = session.get('lang', 'en')
            return translations.get(lang, translations['en']).get(key, key)
        return dict(t=t)

    # --- User Loader ---
    @login_manager.user_loader
    def load_user(user_id):
        user = User.query.get(int(user_id))
        if user:
            return user
        return UserAdmin.query.get(int(user_id))
    
    # --- Offline DB Initialization ---
    def init_offline_db():
        offline_db_path = os.path.join(basedir, 'offline_logs.db')
        try:
            conn = sqlite3.connect(offline_db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS predictions 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          prediction TEXT, 
                          confidence REAL, 
                          timestamp DATETIME,
                          synced INTEGER DEFAULT 0)''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing offline DB: {e}")

    with app.app_context():
        db.create_all()
        init_offline_db()
        
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
