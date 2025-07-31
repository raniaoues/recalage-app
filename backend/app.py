from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from datetime import datetime
from bson import ObjectId
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import io
import base64
from sklearn.metrics import mean_squared_error
from skimage.metrics import normalized_mutual_information as mutual_information
from PIL import Image
from registration_model import (
    create_pyramids, prepare_multi_resolution_data,
    HomographyNet, MINE, multi_resolution_loss, AffineTransform
)
import torch.optim as optim
import torch
import threading
import gridfs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MONGO_URI"] = "mongodb://localhost:27017/medapp"
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)
progress_dict = {'value': 0}
result_dict = {'matrix': None, 'mi_list': None, 'done': False}

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = mongo.db.Doctors.find_one({'email': email, 'password': password})
    if user:
        return jsonify({
            "success": True,
            "doctor_id": str(user['_id']),
            "message": "Connexion réussie"
        })
    else:
        return jsonify({"success": False, "message": "Identifiants invalides"}), 401
@app.route('/get-doctor/<doctor_id>', methods=['GET'])
def get_doctor(doctor_id):
    doctor = mongo.db.Doctors.find_one({'_id': ObjectId(doctor_id)})
    if doctor:
        return jsonify({"name": doctor.get("name", ""), "email": doctor.get("email", "")})
    return jsonify({"error": "Médecin non trouvé"}), 404

@app.route('/add-patient', methods=['POST'])
def add_patient():
    try:
        data = request.json
        pathologie = data.get('pathologie')
        if not pathologie:
            return jsonify({"error": "Champ pathologie requis"}), 400

        annee = str(datetime.now().year)
        compteur = mongo.db.patients.count_documents({
            "patient_id": {"$regex": f"^{annee}-"}
        }) + 1
        id_patient = f"{annee}-{pathologie}-{str(compteur).zfill(4)}"
        doctor_id = data.get('doctor_id')

        patient_doc = {
            "patient_id": id_patient,
            'dossier': data.get('dossier'),
            'date_naissance': data.get('date_naissance'),
            'pathologie': pathologie,
            'infos': data.get('infos'),
            'doctor_id': doctor_id,
            'cases': []  # Ajout du champ pour stocker les cas
        }

        result = mongo.db.patients.insert_one(patient_doc)
        return jsonify({'message': 'Patient enregistré avec succès !'}), 200
    except Exception as e:
        return jsonify({'message': "Erreur lors de l'enregistrement du patient."}), 500

@app.route('/add-case/<patient_id>', methods=['POST'])
def add_case(patient_id):
    try:
        data = request.json
        case = {
            "image1": data.get("image1"),
            "image2": data.get("image2"),
            "matrix": data.get("matrix"),
            "created_at": datetime.now()
        }
        mongo.db.patients.update_one(
            {"_id": ObjectId(patient_id)},
            {"$push": {"cases": case}}
        )
        return jsonify({"message": "Cas ajouté avec succès"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-cases/<patient_id>', methods=['GET'])
def get_cases(patient_id):
    try:
        patient = mongo.db.patients.find_one({'_id': ObjectId(patient_id)})
        if patient:
            return jsonify(patient.get('cases', []))
        return jsonify({"error": "Patient non trouvé"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-patients', methods=['GET'])
def get_patients():
    doctor_id = request.args.get('doctor_id')
    if not doctor_id:
        return jsonify({"error": "doctor_id manquant"}), 400

    patients = mongo.db.patients.find({'doctor_id': doctor_id})
    patient_list = [{**patient, '_id': str(patient['_id'])} for patient in patients]
    return jsonify(patient_list)

@app.route('/get-patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    try:
        patient = mongo.db.patients.find_one({'_id': ObjectId(patient_id)})
        if patient:
            patient['_id'] = str(patient['_id'])
            return jsonify(patient)
        else:
            return jsonify({"error": "Patient non trouvé"}), 404
    except Exception:
        return jsonify({"error": "ID patient invalide"}), 400

@app.route('/get-my-patients', methods=['POST'])
def get_my_patients():
    data = request.json
    doctor_id = data.get('doctor_id')
    if not doctor_id:
        return jsonify({"error": "doctor_id manquant"}), 400

    patients = mongo.db.patients.find({'doctor_id': doctor_id})
    patient_list = [{**patient, '_id': str(patient['_id'])} for patient in patients]
    return jsonify(patient_list)

@app.route('/update-patient/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    data = request.json
    update_fields = {k: data[k] for k in ['pathologie', 'date_naissance', 'dossier', 'infos'] if k in data}

    if update_fields:
        try:
            result = mongo.db.patients.update_one({'_id': ObjectId(patient_id)}, {'$set': update_fields})
            if result.matched_count > 0:
                return jsonify({"message": "Patient mis à jour avec succès"})
            else:
                return jsonify({"error": "Patient non trouvé"}), 404
        except Exception:
            return jsonify({"error": "ID patient invalide"}), 400
    else:
        return jsonify({"error": "Aucune donnée à mettre à jour"}), 400

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        fixed = request.files['fixed']
        moved = request.files['moved']

        fixed_path = os.path.join(UPLOAD_FOLDER, 'fixed_resized.png')
        moved_path = os.path.join(UPLOAD_FOLDER, 'moved_resized.png')
        fixed_img = cv2.resize(cv2.imdecode(np.frombuffer(fixed.read(), np.uint8), -1), (512, 512))
        moved_img = cv2.resize(cv2.imdecode(np.frombuffer(moved.read(), np.uint8), -1), (512, 512))

        cv2.imwrite(fixed_path, fixed_img)
        cv2.imwrite(moved_path, moved_img)

        diff = np.abs(fixed_img.astype(np.float32) - moved_img.astype(np.float32))
        diff = (diff / np.max(diff) * 255).astype(np.uint8)

        _, buffer = cv2.imencode('.png', diff)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"message": "Images resized and difference calculated.", "registered_image": encoded_image}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-matrix", methods=["POST"])
def generate_matrix():
    data = request.get_json()
    neurons = data.get("neurons", 300)
    epochs = data.get("epochs", 100)  # Pour test, 100 epochs
    direction = data.get("direction", "forward")

    fixed_b64 = data.get("fixed_image")
    moved_b64 = data.get("moved_image")

    if not fixed_b64 or not moved_b64:
        return jsonify({"error": "Images manquantes"}), 400

    def decode_base64_image(b64_string):
        if b64_string.startswith('data:image'):
            b64_string = b64_string.split(',')[1]
        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image).astype(np.float32) / 255.0

    try:
        I = decode_base64_image(fixed_b64)
        J = decode_base64_image(moved_b64)
    except Exception as e:
        return jsonify({"error": f"Erreur de décodage des images : {str(e)}"}), 400


    I = cv2.resize(I, (512, 512))
    J = cv2.resize(J, (512, 512))

    pyramid_I, pyramid_J, nChannel = create_pyramids(I, J)
    I_lst, J_lst, h_lst, w_lst, xy_lst, ind_lst = prepare_multi_resolution_data(pyramid_I, pyramid_J, nChannel)

    homography_net = HomographyNet().to(device)
    mine_net = MINE(nChannel=nChannel, n_neurons=neurons, dropout_rate=0.5, bsize=20).to(device)

    optimizer = optim.Adam([
        {'params': mine_net.parameters(), 'lr': 1e-3},
        {'params': homography_net.vL, 'lr': 5e-3},
        {'params': homography_net.v1, 'lr': 1e-4}
    ], amsgrad=True)

    # Réinitialisation des variables globales
    progress_dict['value'] = 0
    result_dict['matrix'] = None
    result_dict['mi_list'] = []
    result_dict['done'] = False

    def training_task():
        mi_list = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = multi_resolution_loss(I_lst, J_lst, xy_lst, ind_lst, homography_net, mine_net, L=6, nChannel=nChannel)
            mi_list.append(-loss.item())
            loss.backward()
            optimizer.step()
            progress_dict['value'] = int((epoch + 1) / epochs * 100)
            print(f"[Matrix Gen - {direction}] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        with torch.no_grad():
            H = homography_net(0).cpu().numpy()
        result_dict['matrix'] = H.tolist()
        result_dict['mi_list'] = mi_list
        result_dict['done'] = True
        progress_dict['value'] = 100

    thread = threading.Thread(target=training_task)
    thread.start()

    return jsonify({"status": "started", "direction": direction})


@app.route('/progress')
def get_progress():
    return jsonify({"progress": progress_dict['value']})


@app.route('/result')
def get_result():
    if result_dict['done']:
        return jsonify({
            "transformation_matrix": result_dict['matrix'],
            "mi_list": result_dict['mi_list']
        })
    else:
        return jsonify({"status": "processing"}), 202

def dice_coefficient(img1, img2):
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())

def hd95(img1, img2):
    _, binary_img1 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)
    _, binary_img2 = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(binary_img1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary_img2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours1 or not contours2:
        return float('nan')
    points1 = np.vstack(contours1).squeeze()
    points2 = np.vstack(contours2).squeeze()
    if points1.ndim == 1:
        points1 = points1.reshape(-1, 2)
    if points2.ndim == 1:
        points2 = points2.reshape(-1, 2)
    distances1_to_2 = np.array([np.min(np.linalg.norm(p1 - points2, axis=1)) for p1 in points1])
    distances2_to_1 = np.array([np.min(np.linalg.norm(p2 - points1, axis=1)) for p2 in points2])
    return max(np.percentile(distances1_to_2, 95), np.percentile(distances2_to_1, 95))

def hausdorff_distance(img1, img2):
    points1 = np.column_stack(np.where(img1 > 0))
    points2 = np.column_stack(np.where(img2 > 0))
    if len(points1) == 0 or len(points2) == 0:
        return float('nan')
    from scipy.spatial.distance import directed_hausdorff
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]
    return max(hd1, hd2)

def normalized_cross_correlation(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    if img1_flat.size != img2_flat.size:
        return float('nan')
    corr = np.corrcoef(img1_flat, img2_flat)[0, 1]
    return corr


# Helper function to store large images in GridFS
def store_image_in_gridfs(image_data, filename):
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    file_id = fs.put(base64.b64decode(image_data), filename=filename)
    return str(file_id)

# Helper function to retrieve images from GridFS
def get_image_from_gridfs(file_id):
    file_data = fs.get(ObjectId(file_id))
    return base64.b64encode(file_data.read()).decode('utf-8')

@app.route('/register', methods=['POST'])
def register_images():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"error": "patient_id manquant"}), 400

    try:
        data = request.get_json()
        fixed_b64 = data.get("fixed_image")
        moved_b64 = data.get("moved_image")
        matrix = data.get("matrix")
        mi_list = data.get("mi_list", [])

        if not fixed_b64 or not moved_b64 or not matrix:
            return jsonify({"error": "Images ou matrice manquantes"}), 400

        def decode_base64_image(base64_str):
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img = img.resize((512, 512))  # resize identique à training
            return np.array(img).astype(np.float32) / 255.0

        I = decode_base64_image(fixed_b64)
        J = decode_base64_image(moved_b64)
        assert I.shape == (512, 512, 3)
        assert J.shape == (512, 512, 3)

        H_np = np.array(matrix)
        H = torch.tensor(H_np, dtype=torch.float32).to(device)

        def smooth(img): return cv2.GaussianBlur(img, (21, 21), 0)
        I_s, J_s = smooth(I), smooth(J)

        pyramid_I, pyramid_J, nChannel = create_pyramids(I_s, J_s)
        I_lst, J_lst, h_lst, w_lst, xy_lst, ind_lst = prepare_multi_resolution_data(
            pyramid_I, pyramid_J, nChannel)

        with torch.no_grad():
            I_tensor = torch.tensor(I, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            J_tensor = torch.tensor(J, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            xy = xy_lst[0]
            Jw = AffineTransform(J_tensor, H, xy[:, :, 0], xy[:, :, 1])
            Jw = Jw.permute(1, 2, 0).cpu().numpy()
            Jw = np.clip(Jw, 0, 1)
            diff = np.abs(I - Jw)
            diff = diff / np.max(diff)
            overlay = 0.5 * I + 0.5 * Jw

            I_gray_uint8 = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            Jw_gray_uint8 = cv2.cvtColor((Jw * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, bin_fixed = cv2.threshold(I_gray_uint8, 50, 255, cv2.THRESH_BINARY)
        _, bin_registered = cv2.threshold(Jw_gray_uint8, 50, 255, cv2.THRESH_BINARY)

        dice = dice_coefficient(bin_fixed, bin_registered)
        hd_95 = hd95(bin_fixed, bin_registered)
        hd = hausdorff_distance(bin_fixed, bin_registered)
        mse = mean_squared_error(I_gray_uint8.flatten(), Jw_gray_uint8.flatten())
        mi = mutual_information(I_gray_uint8, Jw_gray_uint8)
        ncc = normalized_cross_correlation(I_gray_uint8, Jw_gray_uint8)

        def encode_image(img):
            _, buffer = cv2.imencode('.png', (img * 255).astype(np.uint8))
            return base64.b64encode(buffer).decode('utf-8')

        # Store large images in GridFS instead of directly in the document
        fixed_image_id = store_image_in_gridfs(fixed_b64, f"{patient_id}_fixed.png")
        moved_image_id = store_image_in_gridfs(moved_b64, f"{patient_id}_moved.png")
        registered_image_id = store_image_in_gridfs(encode_image(Jw), f"{patient_id}_registered.png")

        case = {
            "created_at": datetime.now(),
            "fixed_image_id": fixed_image_id,
            "moved_image_id": moved_image_id,
            "registered_image_id": registered_image_id,
            "transformation_matrix": H_np.tolist(),
            "mi_list": mi_list,
            "metrics": {
                "dice_coefficient": dice,
                "hd95_distance": hd_95,
                "hausdorff_distance": hd,
                "mean_squared_error": mse,
                "mutual_information": mi,
                "normalized_cross_correlation": ncc
            }
        }

        patient = mongo.db.patients.find_one({"patient_id": patient_id})
        if not patient:
            return jsonify({"error": "Patient non trouvé"}), 404

        mongo.db.patients.update_one(
            {"patient_id": patient_id},
            {"$push": {"cases": case}}
        )

        # Return the case data with image URLs that can be fetched separately
        return jsonify({
            "message": "Recalage effectué et enregistré avec succès.",
            "fixed_image": encode_image(I),
            "registered_image": encode_image(Jw),
            "difference_image": encode_image(diff),
            "overlay_image": encode_image(overlay),
            "gray_fixed": encode_image(I_gray_uint8 / 255.0),
            "gray_registered": encode_image(Jw_gray_uint8 / 255.0),
            "metrics": case["metrics"]
        }), 200

    except Exception as e:
        print("Erreur dans /register:", str(e))
        return jsonify({"error": str(e)}), 500

# Add endpoint to fetch images from GridFS
@app.route('/get-image/<file_id>', methods=['GET'])
def get_image(file_id):
    try:
        image_data = get_image_from_gridfs(file_id)
        return jsonify({"image": f"data:image/png;base64,{image_data}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    
def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY

    normX = np.sqrt((X0**2.).sum())
    normY = np.sqrt((Y0**2.).sum())
    X0 /= normX
    Y0 /= normY

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        if (np.linalg.det(T) < 0) ^ (not reflection):
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    scale = s.sum() * normX / normY if scaling else 1
    translation = muX - scale * np.dot(muY, T)

    return {
        'rotation': T,
        'scale': scale,
        'translation': translation
    }

def build_transform_matrix(tform):
    R = np.eye(3)
    R[0:2, 0:2] = tform['rotation']
    
    S = np.eye(3) * tform['scale']
    S[2, 2] = 1
    
    t = np.eye(3)
    t[0:2, 2] = tform['translation']
    
    return (R @ S @ t.T).T

def apply_transform(image, matrix, output_shape):
    return cv2.warpAffine(
        image, 
        matrix[0:2, :], 
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# ============ ALIGN + RECENTRAGE FOND NOIR ============

def crop_and_center(image, canvas_size=(512, 512)):
    """Croppe la zone utile et recentre avec du noir autour."""
    coords = cv2.findNonZero((image > 5).astype(np.uint8))  # Tolère un petit bruit
    if coords is None:
        return np.zeros(canvas_size, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]

    result = np.zeros(canvas_size, dtype=np.uint8)
    ch, cw = cropped.shape
    cy, cx = canvas_size[0] // 2, canvas_size[1] // 2

    y1 = cy - ch // 2
    x1 = cx - cw // 2
    y2 = y1 + ch
    x2 = x1 + cw

    if 0 <= y1 < canvas_size[0] and 0 <= x1 < canvas_size[1]:
        result[y1:y2, x1:x2] = cropped
    return result

# =================== FLASK ROUTE ====================

@app.route('/manual-register', methods=['POST'])
def manual_register():
    try:
        data = request.get_json()
        
        ct_points = np.array(data.get('ct_points', []), dtype=np.float32)
        mri_points = np.array(data.get('mri_points', []), dtype=np.float32)
        
        if len(ct_points) != len(mri_points):
            return jsonify({'error': 'Le nombre de points doit être identique'}), 400

        ct = base64_to_cv2_img(data['fixed_image'])
        mri = base64_to_cv2_img(data['moving_image'])

        # Resize (standard)
        ct = cv2.resize(ct, (512, 512))
        mri = cv2.resize(mri, (512, 512))

        # Alignement via Procrustes
        tform = procrustes(ct_points, mri_points)
        M = build_transform_matrix(tform)
        mri_aligned = apply_transform(mri, M, ct.shape)

        # Crop et recentrage
        mri_aligned_centered = crop_and_center(mri_aligned, canvas_size=(512, 512))

        # Visualisation
        diff = create_diff_image(ct, mri_aligned_centered)
        overlay = create_overlay(ct, mri_aligned_centered)

        return jsonify({
            'fixedImageUrl': f"data:image/png;base64,{cv2_img_to_base64(ct)}",
            'registeredImageUrl': f"data:image/png;base64,{cv2_img_to_base64(mri_aligned_centered)}",
            'diffImageUrl': f"data:image/png;base64,{cv2_img_to_base64(diff)}",
            'overlayImageUrl': f"data:image/png;base64,{cv2_img_to_base64(overlay)}",
            'transformation_matrix': M.tolist(),
            'message': 'Recalage et recentrage terminés avec succès.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =================== UTILS ====================

def create_diff_image(fixed, moved):
    diff = np.abs(fixed.astype(float) - moved.astype(float))
    if diff.max() > 0:
        diff = (diff / diff.max() * 255)
    return diff.astype(np.uint8)

def create_overlay(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def base64_to_cv2_img(b64_str):
    if b64_str.startswith('data:image'):
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)

def cv2_img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/calculate-metrics', methods=['POST'])
def calculate_metrics():
    try:
        data = request.get_json()
        fixed_b64 = data.get('fixed_image')
        registered_b64 = data.get('registered_image')
        patient_id = data.get('patient_id')

        if not fixed_b64 or not registered_b64:
            return jsonify({"error": "Images manquantes"}), 400

        def decode_base64_image(b64_string):
            if b64_string.startswith('data:image'):
                b64_string = b64_string.split(',')[1]
            img_data = base64.b64decode(b64_string)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img = img.resize((512, 512))
            return np.array(img).astype(np.float32) / 255.0

        fixed_img = decode_base64_image(fixed_b64)
        registered_img = decode_base64_image(registered_b64)

        # Convertir en niveaux de gris
        fixed_gray = cv2.cvtColor((fixed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        registered_gray = cv2.cvtColor((registered_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Seuillage pour les métriques basées sur les contours
        _, bin_fixed = cv2.threshold(fixed_gray, 50, 255, cv2.THRESH_BINARY)
        _, bin_registered = cv2.threshold(registered_gray, 50, 255, cv2.THRESH_BINARY)

        # Calcul des métriques
        dice = dice_coefficient(bin_fixed, bin_registered)
        hd_95 = hd95(bin_fixed, bin_registered)
        hd = hausdorff_distance(bin_fixed, bin_registered)
        mse = mean_squared_error(fixed_gray.flatten(), registered_gray.flatten())
        mi = mutual_information(fixed_gray, registered_gray)
        ncc = normalized_cross_correlation(fixed_gray, registered_gray)

        """ # Enregistrer dans la base de données
        if patient_id:
            case = {
                "created_at": datetime.now(),
                "metrics": {
                    "dice_coefficient": dice,
                    "hd95_distance": hd_95,
                    "hausdorff_distance": hd,
                    "mean_squared_error": mse,
                    "mutual_information": mi,
                    "normalized_cross_correlation": ncc
                },
                "registration_type": "manual"
            }

            mongo.db.patients.update_one(
                {"patient_id": patient_id},
                {"$push": {"cases": case}}
            ) """

        return jsonify({
            "dice_coefficient": dice,
            "hd95_distance": hd_95,
            "hausdorff_distance": hd,
            "mean_squared_error": mse,
            "mutual_information": mi,
            "normalized_cross_correlation": ncc
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/api/stats', methods=['GET'])
def get_statistics():
    try:
        nb_patients = mongo.db.patients.count_documents({})
        all_patients = mongo.db.patients.find()

        nb_exams = 0
        exams_by_pathology = {}

        print("---- PATIENTS DÉTECTÉS ----")

        for p in all_patients:
            patho = p.get("pathologie")
            print("→ pathologie :", patho)
            num_cases = len(p.get("cases", []))
            nb_exams += num_cases
            if patho:
                exams_by_pathology[patho] = exams_by_pathology.get(patho, 0) + num_cases

        doctor = mongo.db.doctors.find_one({}, {"name": 1})
        doctor_name = doctor.get("name", "Inconnu") if doctor else "Inconnu"

        return jsonify({
            "nb_patients": nb_patients,
            "nb_exams": nb_exams,
            "doctor_name": doctor_name,
            "exams_by_pathology": exams_by_pathology
        })
    except Exception as e:
        print("🔥 ERREUR DANS /api/stats:", str(e))
        return jsonify({"error": "Erreur interne"}), 500


if __name__ == '__main__':
    app.run(debug=True)