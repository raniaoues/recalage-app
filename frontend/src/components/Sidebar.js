import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, User, Upload, FileText, LogOut, LogIn,UserPlus  } from 'lucide-react';

function Sidebar({ isAuthenticated, onLogout }) {
const [collapsed, setCollapsed] = useState(false);
const location = useLocation();

const menuItems = [
{ label: 'Tableau de bord', icon: <Home size={20} />, to: '/' },
{ label: 'Liste Patients', icon: <User size={20} />, to: '/patients' },
{ label: 'Créer un patient', icon: <UserPlus size={20} />, to: '/patient-form', protected: true },
{ label: 'Importer des images', icon: <Upload size={20} />, to: '/upload', protected: true },
//{ label: 'Exporter les résultats', icon: <FileText size={20} />, to: '/export', protected: true },
];

return (
<div
    style={{
    position: 'fixed',
    top: 0,
    left: 0,
    height: '100vh',
    width: collapsed ? '60px' : '230px',
    background: '#ffffff',
    borderRight: '1px solid #e0e0e0',
    transition: 'width 0.3s ease',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    zIndex: 1000,
    }}
>
    {/* Sidebar Header */}
    <div>
    <button
        onClick={() => setCollapsed(!collapsed)}
        style={{
        background: 'transparent',
        border: 'none',
        color: '#4F68C5',
        padding: '18px',
        fontSize: '20px',
        cursor: 'pointer',
        textAlign: 'left',
        width: '100%',
        }}
    >
        {collapsed ? '☰' : '✖'}
    </button>

    {/* Menu */}
    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {menuItems.map(
        (item, index) =>
            (!item.protected || isAuthenticated) && (
            <li key={index}>
                <Link
                to={item.to}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '12px 20px',
                    color:
                    location.pathname === item.to ? '#4F68C5' : '#4F68C5',
                    textDecoration: 'none',
                    fontWeight: '500',
                    background:
                    location.pathname === item.to
                        ? '#f0f4ff'
                        : 'transparent',
                    transition: 'all 0.2s ease',
                }}
                >
                <span style={{ marginRight: collapsed ? 0 : 12 }}>
                    {item.icon}
                </span>
                {!collapsed && item.label}
                </Link>
            </li>
            )
        )}
    </ul>
    </div>

    {/* Footer */}
    <div style={{ padding: '16px' }}>
    {isAuthenticated ? (
        <button
        onClick={onLogout}
        style={{
            display: 'flex',
            alignItems: 'center',
            background: 'transparent',
            border: 'none',
            color: '#4F68C5',
            fontSize: '1rem',
            cursor: 'pointer',
            padding: '8px 0',
            fontWeight: '500',
            width: '100%',
        }}
        >
        <LogOut size={20} style={{ marginRight: collapsed ? 0 : 12 }} />
        {!collapsed && 'Déconnexion'}
        </button>
    ) : (
        <Link
        to="/login"
        style={{
            display: 'flex',
            alignItems: 'center',
            color: '#4F68C5',
            textDecoration: 'none',
            fontWeight: '500',
            padding: '8px 0',
            width: '100%',
        }}
        >
        <LogIn size={20} style={{ marginRight: collapsed ? 0 : 12 }} />
        {!collapsed && 'Connexion'}
        </Link>
    )}
    </div>
</div>
);
}

export default Sidebar;
