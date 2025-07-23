import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import './PatientDetail.css';

export default function PatientDetail() {
  const { id } = useParams();
  const [patient, setPatient] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    axios.get(`http://localhost:5000/get-patient/${id}`)
      .then(res => setPatient(res.data))
      .catch(err => setError("Patient non trouvé"));
  }, [id]);

  const handleChange = (e) => {
    setPatient({ ...patient, [e.target.name]: e.target.value });
  };

  const handleUpdate = () => {
    axios.put(`http://localhost:5000/update-patient/${id}`, patient)
      .then(res => alert("Patient mis à jour"))
      .catch(err => alert("Erreur de mise à jour"));
  };

  if (error) return <p style={{ color: 'red' }}>{error}</p>;
  if (!patient) return <p>Chargement...</p>;

  return (
  <div className="detail-container">
    <h2>Détails du patient : {patient.patient_id}</h2>

    <label>
      Numéro du dossier :
      <input name="dossier" value={patient.dossier} onChange={handleChange} />
    </label>

    <label>
      Date de naissance :
      <input name="date_naissance" value={patient.date_naissance} onChange={handleChange} />
    </label>

    <label>
      Pathologie :
      <input name="pathologie" value={patient.pathologie} onChange={handleChange} />
    </label>

    <label>
      Infos :
      <textarea name="infos" value={patient.infos} onChange={handleChange} />
    </label>

    <button onClick={handleUpdate}>Mettre à jour</button>
  </div>
);

}
