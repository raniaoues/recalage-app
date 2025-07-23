import React, { useState } from 'react';
import Sidebar from './Sidebar';
import { Outlet } from 'react-router-dom';

export default function Layout({ isAuthenticated, onLogout }) {
const [collapsed, setCollapsed] = useState(false);

return (
<div>
    <Sidebar
    isAuthenticated={isAuthenticated}
    onLogout={onLogout}
    collapsed={collapsed}
    setCollapsed={setCollapsed}
    />
    <main
    style={{
        marginLeft: collapsed ? '60px' : '230px',
        padding: '20px',
        transition: 'margin-left 0.3s ease',
    }}
    >
    <Outlet />
    </main>
</div>
);
}
