from flask import Blueprint, request, jsonify
from db import mongo

patient_bp = Blueprint('patients', __name__)

@patient_bp.route('/add', methods=['POST'])
def add_patient():
    data = request.json
    mongo.db.patients.insert_one(data)
    return jsonify({"message": "Patient ajouté"})

@patient_bp.route('/', methods=['GET'])
def get_patient(id):
    patient = mongo.db.patients.find_one({"patient_id": id})
    if patient:
        patient["_id"] = str(patient["_id"])
        return jsonify(patient)
    return jsonify({"error": "Patient introuvable"}), 404
