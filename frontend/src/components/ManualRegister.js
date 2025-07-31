import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './ManualRegister.css';
import axios from 'axios';
function ManualRegister() {
  const location = useLocation();
  const navigate = useNavigate();
  const BASE_URL = "http://localhost:5000";
  const [numPairs, setNumPairs] = useState(4);
  const [ctPoints, setCtPoints] = useState([]);
  const [mriPoints, setMriPoints] = useState([]);
  const [clickingOnCT, setClickingOnCT] = useState(true);
  const [totalClicks, setTotalClicks] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [serverResponse, setServerResponse] = useState(null);
  const [registeredImageUrl, setRegisteredImageUrl] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [doctorId, setDoctorId] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [differenceImage, setDifferenceImage] = useState(null);
  const [dynamicDifference, setDynamicDifference] = useState(null);
  const [differenceAlpha, setDifferenceAlpha] = useState(0.5);

  const containerRef = useRef(null);
  const { fixedImage, movingImage } = location.state || {};

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
    if (registeredImageUrl && fixedImage) {
      computeDynamicDifference(differenceAlpha);
    }
  }, [differenceAlpha, registeredImageUrl, fixedImage]);

  const computeDynamicDifference = async (alpha) => {
    if (!fixedImage || !registeredImageUrl) return;

    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      canvas.width = 512;
      canvas.height = 512;

      const loadImage = (src) => new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = src;
      });

      const [fixedImg, registeredImg] = await Promise.all([
        loadImage(fixedImage),
        loadImage(registeredImageUrl)
      ]);

      ctx.drawImage(fixedImg, 0, 0);
      const fixedData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(registeredImg, 0, 0);
      const registeredData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      const diffData = ctx.createImageData(canvas.width, canvas.height);
      
      for (let i = 0; i < fixedData.data.length; i += 4) {
        const weight = (alpha < 0.5) ? (0.5 - alpha) * 2 : (alpha - 0.5) * 2;
        
        for (let ch = 0; ch < 3; ch++) {
          const fixedVal = fixedData.data[i + ch];
          const registeredVal = registeredData.data[i + ch];
          
          diffData.data[i + ch] = Math.round(
            (alpha < 0.5 ? fixedVal : registeredVal) * weight + 
            Math.abs(fixedVal - registeredVal) * (1 - weight)
          );
        }
        diffData.data[i + 3] = 255;
      }

      ctx.putImageData(diffData, 0, 0);
      setDynamicDifference(canvas.toDataURL());
      
    } catch (error) {
      console.error("Erreur de calcul:", error);
    }
  };

  if (!fixedImage || !movingImage) {
    return (
      <div className="manual-register-container">
        <p>Aucune image reçue. Retour à l'upload.</p>
        <button className="btn" onClick={() => navigate('/upload')}>Retour</button>
      </div>
    );
  }

  const toBase64 = async (url) => {
    const response = await fetch(url);
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const handleClick = (event) => {
    if (totalClicks >= 2 * numPairs) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const imageWidth = rect.width / 2;

    if (clickingOnCT && x < imageWidth) {
      setCtPoints([...ctPoints, [x, y]]);
      setClickingOnCT(false);
      setTotalClicks(totalClicks + 1);
    } else if (!clickingOnCT && x >= imageWidth) {
      setMriPoints([...mriPoints, [x - imageWidth, y]]);
      setClickingOnCT(true);
      setTotalClicks(totalClicks + 1);
    }
  };

  const handleSubmitPoints = async () => {
    if (ctPoints.length !== numPairs || mriPoints.length !== numPairs) {
      alert(`Veuillez sélectionner exactement ${numPairs} paires de points.`);
      return;
    }

    setIsSubmitting(true);
    setServerResponse(null);
    setRegisteredImageUrl(null);
    setMetrics(null);

    try {
      const fixedImageBase64 = await toBase64(fixedImage);
      const movingImageBase64 = await toBase64(movingImage);

      const response = await fetch('http://localhost:5000/manual-register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_pairs: numPairs,
          ct_points: ctPoints,
          mri_points: mriPoints,
          fixed_image: fixedImageBase64,
          moving_image: movingImageBase64,
          patient_id: patientId,
          doctor_id: doctorId
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Erreur serveur');
      }

      const data = await response.json();
      setServerResponse(data);
      if (data.registeredImageUrl) {
        setRegisteredImageUrl(data.registeredImageUrl);
      }

      // Calculer les métriques
      const metricsResponse = await fetch('http://localhost:5000/calculate-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fixed_image: fixedImageBase64,
          registered_image: data.registeredImageUrl.split(',')[1],
          patient_id: patientId
        }),
      });

      if (metricsResponse.ok) {
        const metricsData = await metricsResponse.json();
        setMetrics(metricsData);
      }

      
    } catch (error) {
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="manual-register-container">
      <header className="header">
        <div className="patient-id">
          <strong>Patient ID :</strong>
          <input className="patient-input" value={patientId} readOnly />
        </div>

        <div className="doctor-name">Dr. {doctorName}</div>

        <button className="logout-button">Logout</button>
      </header>

      <h2>Recalage Manuel</h2>
      <p className="instructions">
      </p>

      <div className="input-group">
        <label>Nombre de paires de points :</label>
        <input
          type="number"
          min="1"
          max="20"
          value={numPairs}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            if (val > 0 && val <= 20) {
              setNumPairs(val);
              setCtPoints([]);
              setMriPoints([]);
              setClickingOnCT(true);
              setTotalClicks(0);
              setServerResponse(null);
              setRegisteredImageUrl(null);
              setMetrics(null);
            }
          }}
          disabled={totalClicks > 0}
        />
      </div>

      <div className="image-selection-area" ref={containerRef} onClick={handleClick}>
        <img src={fixedImage} alt="CT" className="half-image left" draggable={false} />
        <img src={movingImage} alt="MRI" className="half-image right" draggable={false} />
        <svg className="points-overlay">
          {ctPoints.map(([x, y], i) => (
            <circle key={`ct-${i}`} cx={x} cy={y} r={5} fill="blue" />
          ))}
          {mriPoints.map(([x, y], i) => (
            <circle
              key={`mri-${i}`}
              cx={x + (containerRef.current?.offsetWidth || 0) / 2}
              cy={y}
              r={5}
              fill="green"
            />
          ))}
        </svg>
      </div>

      <button
        className="btn submit-btn"
        onClick={handleSubmitPoints}
        disabled={isSubmitting || totalClicks < 2 * numPairs}
      >
        {isSubmitting ? 'Envoi...' : 'Register'}
      </button>

      {registeredImageUrl && (
        <div className="result-section">
          <h3>Résultats du recalage</h3>
          
          <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
            <div style={{ flex: 1 }}>
              <h4>Image Fixe</h4>
              <img src={fixedImage} alt="Fixed" style={{ width: '100%', maxHeight: '300px' }} />
            </div>
            
            <div style={{ flex: 1 }}>
              <h4>Image Recalée</h4>
              <img
                src={registeredImageUrl}
                alt="Registered"
                style={{ width: '100%', maxHeight: '300px' }}
              />
            </div>
            
            <div style={{ flex: 1 }}>
              <h4>Différence</h4>
              {dynamicDifference ? (
                <img
                  src={dynamicDifference}
                  alt="Difference"
                  style={{ width: '100%', maxHeight: '300px', maxWidth: '300px' }}
                />
              ) : (
                <div style={{ height: '300px', background: '#f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  Calcul en cours...
                </div>
              )}
              
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={differenceAlpha}
                onChange={(e) => setDifferenceAlpha(parseFloat(e.target.value))}
                style={{ width: '100%', marginTop: '10px' }}
              />
            </div>
          </div>

          {metrics && (
            <div className="metrics-section">
              <h4>Métriques du recalage</h4>
              <ul>
                <li>Dice Coefficient: {metrics.dice_coefficient?.toFixed(4) || 'N/A'}</li>
                <li>HD95 Distance: {metrics.hd95_distance?.toFixed(4) || 'N/A'}</li>
                <li>Hausdorff Distance: {metrics.hausdorff_distance?.toFixed(4) || 'N/A'}</li>
                <li>Mean Squared Error: {metrics.mean_squared_error?.toFixed(4) || 'N/A'}</li>
                <li>Mutual Information: {metrics.mutual_information?.toFixed(4) || 'N/A'}</li>
                <li>Normalized Cross Correlation: {metrics.normalized_cross_correlation?.toFixed(4) || 'N/A'}</li>
              </ul>
            </div>
          )}
        </div>
      )}

      <button className="btn return-btn" onClick={() => navigate('/upload')}>
        Retour à l'upload
      </button>
    </div>
  );
}

export default ManualRegister;