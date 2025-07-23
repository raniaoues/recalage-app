import json
from flask_pymongo import PyMongo
from flask import Flask

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/medapp"
mongo = PyMongo(app)

# Charger le fichier users.json
with open('Doctors.json', 'r') as f:
    users_data = json.load(f)

# Insérer les utilisateurs dans MongoDB
with app.app_context():
    for email, info in users_data.items():
        existing_user = mongo.db.Doctors.find_one({'email': email})
        if not existing_user:
            mongo.db.Doctors.insert_one({
                'email': email,
                'password': info['password']  # à hasher plus tard !
            })
            print(f"Utilisateur {email} inséré.")
        else:
            print(f"Utilisateur {email} existe déjà.")
