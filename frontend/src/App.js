import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LoginForm from './components/Login';
import Sidebar from './components/Sidebar'; // le sidebar ci-dessus
import ExportResults from './components/ExportResults';
import PatientForm from './components/Patient';
import PatientList from './components/PatientList';
import PatientDetail from './components/PatientDetail';
import UploadImages from './components/UploadImages';
import Cases from './components/cases';
function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);

  useEffect(() => {
    const loggedIn = localStorage.getItem('authenticated');
    setIsAuthenticated(loggedIn === 'true');
    setCheckingAuth(false);
  }, []);

  const handleLogin = () => {
    localStorage.setItem('authenticated', 'true');
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('authenticated');
    setIsAuthenticated(false);
  };

  const ProtectedRoute = ({ children }) => {
    if (checkingAuth) return null;
    return isAuthenticated ? children : <Navigate to="/login" replace />;
  };

  if (checkingAuth) return null;

  return (
    <Router>
      <Sidebar isAuthenticated={isAuthenticated} onLogout={handleLogout} />
      <main className={`content ${false ? 'collapsed' : ''}`}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route
            path="/login"
            element={
              isAuthenticated ? <Navigate to="/" replace /> : <LoginForm onLogin={handleLogin} />
            }
          />
          <Route
            path="/patients"
            element={
              <ProtectedRoute>
                <PatientList />
              </ProtectedRoute>
            }
          />

          <Route
            path="/upload"
            element={
              <ProtectedRoute>
                <UploadImages />
              </ProtectedRoute>
            }
          />
          <Route
            path="/export"
            element={
              <ProtectedRoute>
                <ExportResults />
              </ProtectedRoute>
            }
          />
          <Route
            path="/patient-form"
            element={
              <ProtectedRoute>
                <PatientForm />
              </ProtectedRoute>
            }
          />
          <Route path="/UploadImages" element={
            <ProtectedRoute>
            <UploadImages />
            </ProtectedRoute>} />
          <Route path="/patients/:id" element={<PatientDetail />} />
          <Route path="/cases/:id" element={
            <ProtectedRoute>
              <Cases />
            </ProtectedRoute>
          } />

          <Route path="*" element={<Navigate to="/" />} />
        </Routes>

      </main>
    </Router>
  );
}

function Dashboard() {
  return (
    <div>
      <h2>Bienvenue Docteur 👩‍⚕️</h2>
      <p>Tableau de bord accessible même sans connexion.</p>
    </div>
  );
}



export default App;
