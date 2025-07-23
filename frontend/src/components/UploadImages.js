import React, { useState, useRef, useEffect } from 'react';
import './UploadImages.css';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement);

const BASE_URL = "http://localhost:5000";

export default function ImageRegistrationApp() {
  const [miList, setMiList] = useState([]);
  const [fixedImage, setFixedImage] = useState(null);
  const [movingImage, setMovingImage] = useState(null);
  const [alpha, setAlpha] = useState(0);
  const [neurons, setNeurons] = useState(100);
  const [epochs, setEpochs] = useState(100);
  const [matrix, setMatrix] = useState(null);
  const canvasRef = useRef(null);
  
  const [registeredImage, setRegisteredImage] = useState(null);
  const [differenceImage, setDifferenceImage] = useState(null);
  const [overlayImage, setOverlayImage] = useState(null);
  const [grayFixed, setGrayFixed] = useState(null);
  const [grayRegistered, setGrayRegistered] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [doctorId, setDoctorId] = useState('');
  const [progress, setProgress] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [metrics, setMetrics] = useState(null);

  // Variable pour stocker l'intervalle de mise à jour du progrès
  let progressInterval = null;

  useEffect(() => {
    const id = localStorage.getItem("cas_patient_id");
    if (id) {
      axios.get(`${BASE_URL}/get-patient/${id}`)
        .then(res => setPatientId(res.data.patient_id))
        .catch(err => console.error("Erreur récupération patient :", err));
    }
  }, []);

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
  }, []);

  useEffect(() => {
    const fromHistory = localStorage.getItem('cas_from_history') === 'true';
    if (fromHistory) {
      const fixedBase64 = localStorage.getItem('cas_fixed');
      const movedBase64 = localStorage.getItem('cas_moved');
      const matrixStr = localStorage.getItem('cas_matrix');
      const miListStr = localStorage.getItem('cas_mi_list');

      if (fixedBase64 && movedBase64 && matrixStr) {
        try {
          const parsedMatrix = JSON.parse(matrixStr);
          const parsedMiList = miListStr ? JSON.parse(miListStr) : [];

          setFixedImage(`data:image/png;base64,${fixedBase64}`);
          setMovingImage(`data:image/png;base64,${movedBase64}`);
          setMatrix(parsedMatrix);
          setMiList(parsedMiList);
        } catch (e) {
          console.error("Erreur de parsing JSON :", e);
        }
      }
    }
    return () => {
      localStorage.removeItem("cas_from_history");
    };
  }, []);

  // Nettoyer l'intervalle de mise à jour du progrès lors du démontage du composant
  useEffect(() => {
    return () => {
      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
    };
  }, []);

  const handleFixedUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFixedImage(reader.result);
        localStorage.setItem("cas_fixed", reader.result.split(',')[1]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleMovingUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setMovingImage(reader.result);
        localStorage.setItem("cas_moved", reader.result.split(',')[1]);
      };
      reader.readAsDataURL(file);
    }
  };

  useEffect(() => {
    if (!fixedImage || !movingImage) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const fixedImg = new Image();
    const movingImg = new Image();
    fixedImg.crossOrigin = "anonymous";
    movingImg.crossOrigin = "anonymous";

    let isCancelled = false;

    fixedImg.src = fixedImage;
    movingImg.src = movingImage;

    fixedImg.onload = () => {
      movingImg.onload = () => {
        if (isCancelled) return;
        const width = Math.max(fixedImg.width, movingImg.width);
        const height = Math.max(fixedImg.height, movingImg.height);
        canvas.width = width;
        canvas.height = height;

        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(fixedImg, 0, 0);
        const fixedData = ctx.getImageData(0, 0, width, height);

        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(movingImg, 0, 0);
        const movedData = ctx.getImageData(0, 0, width, height);

        const diffData = ctx.createImageData(width, height);
        const a = alpha / 100;

        for (let i = 0; i < diffData.data.length; i += 4) {
          const diffR = Math.abs(fixedData.data[i] - movedData.data[i]);
          const diffG = Math.abs(fixedData.data[i + 1] - movedData.data[i + 1]);
          const diffB = Math.abs(fixedData.data[i + 2] - movedData.data[i + 2]);

          const fixedR = fixedData.data[i];
          const fixedG = fixedData.data[i + 1];
          const fixedB = fixedData.data[i + 2];

          const movedR = movedData.data[i];
          const movedG = movedData.data[i + 1];
          const movedB = movedData.data[i + 2];

          if (a > 0) {
            diffData.data[i] = Math.min(255, diffR * (1 - a) + fixedR * a);
            diffData.data[i + 1] = Math.min(255, diffG * (1 - a) + fixedG * a);
            diffData.data[i + 2] = Math.min(255, diffB * (1 - a) + fixedB * a);
          } else if (a < 0) {
            const aa = Math.abs(a);
            diffData.data[i] = Math.min(255, diffR * (1 - aa) + movedR * aa);
            diffData.data[i + 1] = Math.min(255, diffG * (1 - aa) + movedG * aa);
            diffData.data[i + 2] = Math.min(255, diffB * (1 - aa) + movedB * aa);
          } else {
            diffData.data[i] = diffR;
            diffData.data[i + 1] = diffG;
            diffData.data[i + 2] = diffB;
          }
          diffData.data[i + 3] = 255;
        }

        ctx.putImageData(diffData, 0, 0);
      };
    };

    return () => { isCancelled = true; };
  }, [fixedImage, movingImage, alpha]);

  const handleRegister = async () => {
    const fromHistory = localStorage.getItem("cas_from_history") === "true";
    if (fromHistory) {
      alert("Ce cas provient de l'historique, pas besoin de le sauvegarder à nouveau.");
      return;
    }

    const fixed_image = localStorage.getItem("cas_fixed");
    const moved_image = localStorage.getItem("cas_moved");
    const matrixStr = localStorage.getItem("cas_matrix");
    const miListStr = localStorage.getItem("cas_mi_list");

    if (!fixed_image || !moved_image || !matrixStr) {
      alert("Données d'images ou matrice manquantes.");
      return;
    }

    let matrix, mi_list;
    try {
      matrix = JSON.parse(matrixStr);
      mi_list = miListStr ? JSON.parse(miListStr) : [];
    } catch (e) {
      alert("Erreur de parsing des données.");
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/register?patient_id=${patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fixed_image, moved_image, matrix, mi_list })
      });

      const data = await response.json();
      console.log("Réponse backend metrics:", data.metrics);
      if (response.ok && data.registered_image) {
        setRegisteredImage(`data:image/png;base64,${data.registered_image}`);

        if (data.difference_image) {
          setDifferenceImage(`data:image/png;base64,${data.difference_image}`);
        } else {
          setDifferenceImage(null);
        }

        if (data.metrics) {
          setMetrics(data.metrics);
        } else {
          setMetrics(null);
        }

        localStorage.removeItem("cas_fixed");
        localStorage.removeItem("cas_moved");
        localStorage.removeItem("cas_matrix");
        localStorage.removeItem("cas_mi_list");
        localStorage.removeItem("cas_from_history");
      } else {
        alert("Erreur lors du recalage : " + (data.error || "Inconnue"));
      }
    } catch (err) {
      alert("Erreur réseau ou serveur.");
    }
  };

  const handleGenerateMatrix = async () => {
  if (!fixedImage || !movingImage) {
    alert("Veuillez d'abord charger les deux images.");
    return;
  }

  try {
    const fixedBase64 = fixedImage.split(',')[1];
    const movedBase64 = movingImage.split(',')[1];

    setIsGenerating(true);
    setProgress(0);

    // Lancer la génération côté serveur
    const response = await fetch(`${BASE_URL}/generate-matrix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fixed_image: fixedBase64,
        moved_image: movedBase64,
        neurons,
        epochs,
        direction: "forward",
        patient_id: patientId,
        doctor_id: doctorId
      }),
    });

    if (!response.ok) {
      const data = await response.json();
      alert("Erreur génération matrice : " + (data.error || "inconnue"));
      setIsGenerating(false);
      return;
    }

    // Maintenant on interroge régulièrement /progress
    const intervalId = setInterval(async () => {
      try {
        const progressRes = await fetch(`${BASE_URL}/progress`);
        const progressData = await progressRes.json();
        const val = progressData.progress || 0; // dans ta route Flask, tu renvoies "progress"
        setProgress(val);

        if (val >= 100) {
          clearInterval(intervalId);
          setProgress(100);

          // Quand fini, récupérer le résultat
          const resultRes = await fetch(`${BASE_URL}/result`);
          if (resultRes.ok) {
            const resultData = await resultRes.json();
            setMatrix(resultData.transformation_matrix);
            localStorage.setItem("cas_matrix", JSON.stringify(resultData.transformation_matrix));

            setMiList(resultData.mi_list || []);
            localStorage.setItem("cas_mi_list", JSON.stringify(resultData.mi_list || []));

          } else {
            alert("Erreur lors de la récupération du résultat final.");
          }
          setIsGenerating(false);
        }
      } catch (err) {
        console.error("Erreur récupération progression :", err);
        clearInterval(intervalId);
        setIsGenerating(false);
      }
    }, 500);

  } catch (err) {
    alert("Erreur serveur ou réseau.");
    console.error(err);
    setIsGenerating(false);
  }
};


  return (
    <div className="app-container">
      <header className="header">
        <div className="patient-id">
          <strong>Patient ID :</strong>
          <input className="patient-input" value={patientId} readOnly />
        </div>

        <div className="doctor-name">Dr. {doctorName}</div>

        <button className="logout-button">Logout</button>
      </header>

      <div className="upload-row">
        {/* Fixed Image */}
        <div className="upload-field">
          <label className="image-label">Fixed Image</label>
          {fixedImage ? (
            <img
              src={fixedImage}
              alt="Fixed"
              className="preview-img"
              onClick={() => document.getElementById('fixed-upload').click()}
            />
          ) : (
            <div className="upload-box" onClick={() => document.getElementById('fixed-upload').click()}>
              <input id="fixed-upload" type="file" accept="image/*" onChange={handleFixedUpload} />
              Click or drop fixed image
            </div>
          )}
        </div>

        {/* Moving Image */}
        <div className="upload-field">
          <label className="image-label">Moving Image</label>
          {movingImage ? (
            <img
              src={movingImage}
              alt="Moving"
              className="preview-img"
              onClick={() => document.getElementById('moving-upload').click()}
            />
          ) : (
            <div className="upload-box" onClick={() => document.getElementById('moving-upload').click()}>
              <input id="moving-upload" type="file" accept="image/*" onChange={handleMovingUpload} />
              Click or drop moving image
            </div>
          )}
        </div>
      </div>

      {/* Slider */}
      <div className="slider-section">
        <label className="slider-label">Difference Image</label>
        <canvas ref={canvasRef} className="matrix-image" />
        <input
          type="range"
          min={-100}
          max={100}
          value={alpha}
          onChange={(e) => setAlpha(parseInt(e.target.value))}
          className="slider-range"
        />
      </div>

      {/* Config (params) */}
      <div className="config-section">
        <div>
          <label>Nombre de neurones :</label>
          <select value={neurons} onChange={(e) => setNeurons(parseInt(e.target.value))}>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={300}>300</option>
          </select>
        </div>
        <div>
          <label>Nombre d'epochs :</label>
          <input type="number" min={100} value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} />
        </div>
      </div>

      {/* Bouton Générer matrice */}
      <div className="matrix-section" style={{ textAlign: 'center', marginBottom: '30px' }}>
        <button className="load-button" onClick={handleGenerateMatrix} disabled={isGenerating}>
          Générer matrice de transformation
        </button>
      </div>

     {/* Barre de progression */}
{isGenerating && (
  <div
    style={{
      width: '80%',
      margin: '0 auto 20px',
      background: '#fff',
      borderRadius: '8px',
      overflow: 'hidden',
      border: '2px solid #000', // ✅ Cadre noir
      position: 'relative',
      height: '25px'
    }}
  >
    <div
      style={{
        height: '100%',
        width: `${progress}%`,
        backgroundColor: '#6B3E75',
        transition: 'width 0.3s ease-in-out',
      }}
    />
    <span
      style={{
        position: 'absolute',
        top: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 'bold',
        color: '#000', // ✅ Texte noir
        fontSize: '14px'
      }}
    >
      {Math.floor(progress)}%
    </span>
  </div>
)}

      {overlayImage && (
        <div className="overlay-section">
          <h3>Image Fusion Overlay</h3>
          <img src={overlayImage} alt="Overlay" className="preview-img" />
        </div>
      )}

      {grayFixed && (
        <div className="gray-fixed-section">
          <h3>Image Fixe Niveau de Gris</h3>
          <img src={grayFixed} alt="Gray Fixed" className="preview-img" />
        </div>
      )}

      {grayRegistered && (
        <div className="gray-registered-section">
          <h3>Image Recalée Niveau de Gris</h3>
          <img src={grayRegistered} alt="Gray Registered" className="preview-img" />
        </div>
      )}

      {/* Affichage matrice */}
      {matrix && (
        <div className="matrix-wrapper">
          <h3>Matrice de transformation :</h3>
          <div className="matrix-brackets">
            <div className="left-bracket">[</div>
            <table className="matrix-table">
              <tbody>
                {matrix.map((row, i) => (
                  <tr key={i}>
                    {row.map((val, j) => (
                      <td key={j}>{val.toFixed(4)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="right-bracket">]</div>
          </div>
        </div>
      )}

      {/* Graphique MI */}
      {miList.length > 0 && (
        <div style={{ width: "600px", margin: "20px auto" }}>
          <h3>Courbe d'évolution de la Mutual Information</h3>
          <Line
            data={{
              labels: miList.map((_, i) => i + 1),
              datasets: [
                {
                  label: 'MI',
                  data: miList,
                  borderColor: 'rgba(75,192,192,1)',
                  backgroundColor: 'rgba(75,192,192,0.2)',
                 
                                    tension: 0.2,
                },
              ],
            }}
            options={{
              responsive: true,
              scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: { title: { display: true, text: 'Mutual Information' } }
              }
            }}
          />
        </div>
      )}

      {/* Bouton Register */}
      <div className="register-section">
        <button className="register-button" onClick={handleRegister} disabled={isGenerating}>
          Register Images
        </button>
      </div>

      <div style={{ display: 'flex', gap: '150px', alignItems: 'flex-start' }}>
        {/* Registered Image Preview */}
        {registeredImage && (
          <div className="registered-section" style={{ flex: 1 }}>
            <h3>Image Recalée (Registered)</h3>
            <img src={registeredImage} alt="Registered" className="preview-img" style={{ maxWidth: '100%', marginBottom: '40px' }} />
          </div>
        )}

        {/* Difference Image After Registration */}
        {differenceImage && (
          <div className="difference-section" style={{ flex: 1 }}>
            <h3>Différence après Recalage</h3>
            <img src={differenceImage} alt="Difference After Registration" className="preview-img" style={{ maxWidth: '100%' , marginBottom: '40px'}} />
          </div>
        )}
      </div>

      {/* Affichage métriques */}
      {metrics && (
        <div className="metrics-section">
          <h3>Métriques du recalage</h3>
          <ul>
            <li>Dice Coefficient: {metrics.dice_coefficient.toFixed(4)}</li>
            <li>HD95 Distance: {metrics.hd95_distance.toFixed(4)}</li>
            <li>Hausdorff Distance: {metrics.hausdorff_distance.toFixed(4)}</li>
            <li>Mean Squared Error: {metrics.mean_squared_error.toFixed(4)}</li>
            <li>Mutual Information: {metrics.mutual_information.toFixed(4)}</li>
            <li>Normalized Cross Correlation: {metrics.normalized_cross_correlation.toFixed(4)}</li>
          </ul>
        </div>
      )}
    </div>
  );
}
