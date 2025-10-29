from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token

def hash_password(pw):
    return generate_password_hash(pw)

def verify_password(hash_pw, password):
    return check_password_hash(hash_pw, password)

def make_token(identity):
    return create_access_token(identity=identity)
