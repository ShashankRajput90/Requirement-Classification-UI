from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(255), nullable=False)
    email         = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)
    google_id     = db.Column(db.String(255), unique=True, nullable=True)
    avatar_url    = db.Column(db.String(500), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)


class BatchRun(db.Model):
    __tablename__ = "batch_runs"

    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    model               = db.Column(db.String(100))
    prompting_technique = db.Column(db.String(100))
    total_stories       = db.Column(db.Integer)
    created_at          = db.Column(db.DateTime, server_default=db.func.current_timestamp())

    results = db.relationship("BatchResult", backref="run", lazy=True)


class BatchResult(db.Model):
    __tablename__ = "batch_results"

    id             = db.Column(db.Integer, primary_key=True)
    batch_run_id   = db.Column(db.Integer, db.ForeignKey("batch_runs.id"), nullable=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    story          = db.Column(db.Text)
    model          = db.Column(db.String(100))
    classification = db.Column(db.String(50))
    category       = db.Column(db.String(255))
    latency        = db.Column(db.Float)