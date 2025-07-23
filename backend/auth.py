import json

USERS_FILE = 'users.json'

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def verify_login(email, password):
    users = load_users()
    if email in users and users[email]['password'] == password:
        return True
    return False
