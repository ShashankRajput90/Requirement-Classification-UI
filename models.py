from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


# ==========================
# USER MODEL
# ==========================
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(255), nullable=False)
    email         = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)
    google_id     = db.Column(db.String(255), unique=True, nullable=True)
    avatar_url    = db.Column(db.String(500), nullable=True)

    # relationships
    feedbacks = db.relationship("Feedback", backref="user", lazy=True)
    edits     = db.relationship("RequirementHistory", back_populates="editor", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)


# ==========================
# BATCH RUN MODEL
# ==========================
class BatchRun(db.Model):
    __tablename__ = "batch_runs"

    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    model               = db.Column(db.String(100))
    prompting_technique = db.Column(db.String(100))
    total_stories       = db.Column(db.Integer)
    created_at          = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )

    results = db.relationship("BatchResult", backref="run", lazy=True)


# ==========================
# BATCH RESULT MODEL
# ==========================
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
    confidence     = db.Column(db.Integer) 
    is_correct     = db.Column(db.Integer)
    true_label     = db.Column(db.String(10))

    # history tracking
    history = db.relationship(
        "RequirementHistory",
        backref="result",
        lazy=True,
        cascade="all, delete-orphan"
    )

    # feedback tracking (NEW FEATURE)
    feedback_entries = db.relationship(
        "Feedback",
        backref="batch_result",
        lazy=True,
        cascade="all, delete-orphan"
    )


# ==========================
# FEEDBACK MODEL (NEW)
# ==========================
class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)

    # who gave feedback
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=True
    )

    # which prediction was corrected
    batch_result_id = db.Column(
        db.Integer,
        db.ForeignKey("batch_results.id", ondelete="CASCADE"),
        nullable=False
    )

    requirement_text = db.Column(db.Text, nullable=False)

    predicted_label = db.Column(db.String(100), nullable=False)
    corrected_label = db.Column(db.String(100), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)  
    created_at = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )


# ==========================
# REQUIREMENT HISTORY MODEL
# ==========================
class RequirementHistory(db.Model):
    __tablename__ = "requirement_history"

    id = db.Column(db.Integer, primary_key=True)

    batch_result_id = db.Column(
        db.Integer,
        db.ForeignKey("batch_results.id", ondelete="CASCADE"),
        nullable=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=True
    )

    previous_story = db.Column(db.Text)
    new_story = db.Column(db.Text)

    previous_classification = db.Column(db.String(50))
    new_classification = db.Column(db.String(50))

    changed_at = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )

    editor = db.relationship(
        "User",
        back_populates="edits",
        foreign_keys=[user_id]
    )