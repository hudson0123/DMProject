from flask import Flask
from .routes import home_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")

    # Register blueprints
    app.register_blueprint(home_bp)

    return app
