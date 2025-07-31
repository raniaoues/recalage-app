import React, { useEffect, useState } from "react";
import axios from "axios";
import "./Dashboard.css";
const BASE_URL = "http://localhost:5000";
function Dashboard() {
const [doctorName, setDoctorName] = useState('');
const [doctorId, setDoctorId] = useState('');

const [stats, setStats] = useState({
nb_patients: 0,
nb_exams: 0,
exams_per_day: {},
doctor_name: "",
exams_by_pathology: {},
});
useEffect(() => {
    const id = localStorage.getItem("doctor_id");
    if (id) {
      setDoctorId(id);
      axios.get(`${BASE_URL}/get-doctor/${id}`)
        .then(res => {
          const email = res.data.email;
          const nameWithoutDomain = email.split('@')[0];
          setDoctorName(nameWithoutDomain);
        })
        .catch(err => console.error("Erreur récupération médecin :", err));
    }
  },[]);
useEffect(() => {
axios
    .get("http://localhost:5000/api/stats")
    .then((res) => setStats(res.data))
    .catch((err) => console.error(err));
}, []);

return (
<div className="dashboard-container">
    <div className="dashboard-header">
    <h1 className="dashboard-title">📊 Dashboard</h1>
    <div className="doctor-name">Welcome {doctorName || "Doctor"}</div>
</div>


    <div className="cards">
    <div className="card">
        <h3>Patients</h3>
        <p>{stats.nb_patients}</p>
    </div>
    <div className="card">
        <h3>Examens</h3>
        <p>{stats.nb_exams}</p>
    </div>
    </div>

    
    <div className="pathology-section">
    <h3>🧬 Examens par pathologie</h3>
    <table>
        <thead>
        <tr>
            <th>Pathologie</th>
            <th>Nombre d'examens</th>
        </tr>
        </thead>
        <tbody>
       {stats.exams_by_pathology &&
        Object.entries(stats.exams_by_pathology).map(([patho, count]) => (
            <tr key={patho}>
            <td>{patho}</td>
            <td>{count}</td>
            </tr>
            ))}
        </tbody>
    </table>
    </div>
</div>
);
}

export default Dashboard;
