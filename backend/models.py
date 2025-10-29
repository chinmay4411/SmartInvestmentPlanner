from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InvestmentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)  # not nullable
    income = db.Column(db.Float, nullable=False)
    expenses = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    risk = db.Column(db.Integer, nullable=False)
    sip = db.Column(db.Float)
    fd = db.Column(db.Float)
    stocks = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
