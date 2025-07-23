import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './PatientList.css';

export default function PatientList() {
const [patients, setPatients] = useState([]);
const [error, setError] = useState('');
const navigate = useNavigate();

useEffect(() => {
const doctor_id = localStorage.getItem('doctor_id');
if (!doctor_id) {
    setError("Aucun médecin connecté.");
    return;
}

axios.get('http://localhost:5000/get-patients', {
    params: { doctor_id },
})
.then(response => {
    setPatients(response.data);
})
.catch(error => {
    console.error(error);
    setError("Erreur lors du chargement des patients.");
});
}, []);

const handleRowClick = (id) => {
navigate(`/patients/${id}`);
};

const checkRecalages = async (patient_id) => {
try {
    const res = await axios.get('http://localhost:5000/check-recalages', {
    params: { patient_id },
    });
    if (res.data && res.data.exists) {
    alert("Des recalages ont déjà été effectués pour ce patient.");
    } else {
    alert("Aucun recalage trouvé pour ce patient.");
    }
} catch (error) {
    alert("Erreur lors de la vérification des recalages.");
}
};

return (
<div className="list-container">
    <h2>Liste des patients</h2>
    {error && <p style={{ color: "red" }}>{error}</p>}

    <table>
    <thead>
        <tr>
        <th>ID</th>
        <th>Dossier</th>
        <th>Date naissance</th>
        <th>Pathologie</th>
        <th>Infos</th>
        <th>Recalages</th> {/* ✅ nouvelle colonne */}
        </tr>
    </thead>
    <tbody>
        {patients.map((p, i) => (
        <tr key={i} style={{ cursor: 'pointer' }}>
            <td onClick={() => handleRowClick(p._id)}>{p.patient_id}</td>
            <td onClick={() => handleRowClick(p._id)}>{p.dossier}</td>
            <td onClick={() => handleRowClick(p._id)}>{p.date_naissance}</td>
            <td onClick={() => handleRowClick(p._id)}>{p.pathologie}</td>
            <td onClick={() => handleRowClick(p._id)}>{p.infos}</td>
            <td className="recalage-buttons">
            <button
                className="btn-blue new"
                onClick={(e) => {
                e.stopPropagation();
                localStorage.setItem('cas_patient_id', p._id);
                navigate('/upload');
                }}
            >
                Nouveau
            </button>
            <button
                className="btn-blue history"
                onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/cases/${p._id}`);
                }}
                >
                Historique
                </button>

            </td>


        </tr>
        ))}
    </tbody>
    </table>
</div>
);
}
