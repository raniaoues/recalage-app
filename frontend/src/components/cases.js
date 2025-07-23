// ✅ CASES.JS (frontend React)
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate, useParams } from 'react-router-dom';

export default function Cases() {
const { id } = useParams(); // patient_id
const [cases, setCases] = useState([]);
const navigate = useNavigate();

useEffect(() => {
axios.get(`http://localhost:5000/get-cases/${id}`)
    .then(response => setCases(response.data))
    .catch(error => console.error(error));
}, [id]);

const handleLoadCase = (c) => {
  localStorage.setItem('cas_fixed', c.fixed_image);
  localStorage.setItem('cas_moved', c.moved_image);
  localStorage.setItem('cas_matrix', JSON.stringify(c.transformation_matrix));
  localStorage.setItem('cas_mi_list', JSON.stringify(c.mi_list || []));  // <-- Bien stocker ici
  localStorage.setItem('cas_patient_id', id);
  localStorage.setItem('cas_from_history', 'true'); 
  navigate('/upload');
};



return (
<div className="case-container">
    <h2>Historique</h2>
    <table>
    <thead>
    <tr>
        <th>Fixed</th>
        <th>Moved</th>
        <th>Matrice</th>
        <th>Action</th>
    </tr>
    </thead>

    <tbody>
        {cases.map((c, i) => (
        <tr key={i}>
            <td><img src={`data:image/png;base64,${c.fixed_image}`} width="100" /></td>
            <td><img src={`data:image/png;base64,${c.moved_image}`} width="100" /></td>
            <td><pre>{JSON.stringify(c.transformation_matrix, null, 2)}</pre></td>
            <td><button onClick={() => handleLoadCase(c)}>Charger</button></td>
        </tr>
        ))}
    </tbody>
    </table>
</div>
);
}
