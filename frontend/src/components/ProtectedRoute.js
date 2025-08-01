import React from 'react';
import { Navigate } from 'react-router-dom';

function ProtectedRoute({ children }) {
const isAuthenticated = localStorage.getItem('authenticated') === 'true';

return isAuthenticated ? children : <Navigate to="/login" />;
}

export default ProtectedRoute;
