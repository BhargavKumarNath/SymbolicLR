import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import AmbientBg from './AmbientBg';

const MainLayout = () => {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    return (
        <>
            <AmbientBg />
            <Sidebar collapsed={sidebarCollapsed} setCollapsed={setSidebarCollapsed} />
            <div style={{
                marginLeft: sidebarCollapsed ? 72 : 220,
                minHeight: "100vh",
                position: "relative", zIndex: 1,
                transition: "margin-left 0.3s cubic-bezier(0.25,0.46,0.45,0.94)"
            }}>
                <div className="fade-in">
                    <Outlet />
                </div>
            </div>
        </>
    );
};

export default MainLayout;
