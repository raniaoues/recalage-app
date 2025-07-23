import React, { useState } from 'react';
import axios from 'axios';
import './Patient.css';

export default function PatientForm() {
const [dossier, setDossier] = useState('');
const [dateNaissance, setDateNaissance] = useState('');
const [pathologie, setPathologie] = useState('');
const [infos, setInfos] = useState('');
const [message, setMessage] = useState('');
const [error, setError] = useState('');

const handleSubmit = async (e) => {
e.preventDefault();
setMessage('');
setError('');

const doctor_id = localStorage.getItem('doctor_id'); // récupération id médecin connecté
console.log("Doctor ID:", doctor_id); // debug

if (!doctor_id) {
    setError("Erreur : ID du médecin non trouvé. Veuillez vous reconnecter.");
    return;
}

try {
    const res = await axios.post('http://localhost:5000/add-patient', {
    dossier,
    date_naissance: dateNaissance,
    pathologie,
    infos,
    doctor_id
    });

    if(res.data.message) {
    setMessage(res.data.message);
    setDossier('');
    setDateNaissance('');
    setPathologie('');
    setInfos('');
    } else {
    setError('Erreur inattendue du serveur.');
    }
} catch (err) {
    console.error(err);
    setError("Erreur lors de l'enregistrement du patient.");
}
};

return (
<div className="form-container">
    <h2>Créer un nouveau patient</h2>
    <form onSubmit={handleSubmit}>
    <input
        type="text"
        placeholder="Numéro de dossier"
        value={dossier}
        onChange={e => setDossier(e.target.value)}
        required
    />
    <input
        type="date"
        placeholder="Date de naissance"
        value={dateNaissance}
        onChange={e => setDateNaissance(e.target.value)}
        required
    />
    <input
        type="text"
        placeholder="Pathologie"
        value={pathologie}
        onChange={e => setPathologie(e.target.value)}
        required
    />
    <textarea
        placeholder="Infos supplémentaires"
        value={infos}
        onChange={e => setInfos(e.target.value)}
    />
    <button type="submit">Enregistrer le patient</button>
    </form>

    {message && <p style={{ marginTop: '15px', color: 'green' }}>{message}</p>}
    {error && <p style={{ marginTop: '15px', color: 'red' }}>{error}</p>}
</div>
);
}
