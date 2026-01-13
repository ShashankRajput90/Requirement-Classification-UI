import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
    
    # API Rate Limiting (requests per minute)
    RATELIMIT_STORAGE_URL = "memory://"
    RATELIMIT_DEFAULT = "100 per minute"
    
    # Model API Keys (load from environment)
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
    COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # Application Settings
    CLASSIFICATION_HISTORY_LIMIT = 1000  # Maximum history entries per session
    BATCH_PROCESSING_LIMIT = 100  # Maximum stories per batch
    DEFAULT_MODEL = 'Groq LLaMA 3.1 8B'
    DEFAULT_PROMPT = 'Zero Shot'
    
    # CORS Settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Create session directory if using filesystem sessions
        session_dir = os.path.join(app.instance_path, 'sessions')
        os.makedirs(session_dir, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'
    
    # Development specific settings
    SESSION_COOKIE_SECURE = False
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Development specific initialization
        print("Flask app running in DEVELOPMENT mode")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Enhanced rate limiting for production
    RATELIMIT_DEFAULT = "50 per minute"
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Setup logging
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/classification_app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Classification application startup (Production)')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False
    
    # Use in-memory session for testing
    SESSION_TYPE = 'null'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}