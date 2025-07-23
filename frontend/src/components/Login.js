import React, { useState } from 'react';
import './Login.css';
import { Eye, EyeOff } from 'lucide-react';

function LoginForm({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
    setError('');
    try {
      const res = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (data.success && data.doctor_id) {
        localStorage.setItem('doctor_id', data.doctor_id);
        localStorage.setItem('authenticated', 'true'); // si tu veux garder un flag
        onLogin();
      } else {
        setError('Email ou mot de passe incorrect.');
      }
    } catch (err) {
      setError('Erreur de connexion au serveur.');
    }
  };

  return (
    <div className="login-box">
      <h2>Connexion</h2>

      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
      />

      <div className="password-container">
        <input
          type={showPassword ? 'text' : 'password'}
          placeholder="Mot de passe"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />
        <span
          className="eye-icon"
          onClick={() => setShowPassword(!showPassword)}
          style={{ cursor: 'pointer' }}
        >
          {showPassword ? <Eye size={20} /> : <EyeOff size={20} />}
        </span>
      </div>

      <button onClick={handleLogin}>Se connecter</button>

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default LoginForm;
