import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import Home from './pages/Home';
import ProtectedRoute from './components/ProtectedRoute';

function AppRouter() {
return (
<Router>
    <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route
        path="/"
        element={
        <ProtectedRoute>
            <Home />
        </ProtectedRoute>
        }
    />
    </Routes>
</Router>
);
}

export default AppRouter;
