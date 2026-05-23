import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';

// Lazy load all pages for code-splitting
const HomePage = lazy(() => import('./pages/HomePage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));
const SystemPage = lazy(() => import('./pages/SystemPage'));
const EvolutionPage = lazy(() => import('./pages/EvolutionPage'));
const MapElitesPage = lazy(() => import('./pages/MapElitesPage'));
const PlaygroundPage = lazy(() => import('./pages/PlaygroundPage'));
const BaselinesPage = lazy(() => import('./pages/BaselinesPage'));
const DiagnosticsPage = lazy(() => import('./pages/DiagnosticsPage'));
const RustPage = lazy(() => import('./pages/RustPage'));

// Premium loading skeleton
const LoadingSkeleton = () => (
    <div style={{ padding: "60px", animation: "pulse 1.5s ease-in-out infinite" }}>
        <div style={{ width: 100, height: 24, borderRadius: 12, background: "rgba(255,255,255,0.05)", marginBottom: 16 }} />
        <div style={{ width: 400, height: 48, borderRadius: 12, background: "rgba(255,255,255,0.1)", marginBottom: 24 }} />
        <div style={{ width: 600, height: 20, borderRadius: 10, background: "rgba(255,255,255,0.05)", marginBottom: 40 }} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <div style={{ height: 300, borderRadius: 16, background: "rgba(255,255,255,0.03)" }} />
            <div style={{ height: 300, borderRadius: 16, background: "rgba(255,255,255,0.03)" }} />
        </div>
    </div>
);

export default function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainLayout />}>
                    <Route index element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <HomePage />
                        </Suspense>
                    } />
                    <Route path="about" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <AboutPage />
                        </Suspense>
                    } />
                    <Route path="system" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <SystemPage />
                        </Suspense>
                    } />
                    <Route path="evolution" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <EvolutionPage />
                        </Suspense>
                    } />
                    <Route path="map" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <MapElitesPage />
                        </Suspense>
                    } />
                    <Route path="playground" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <PlaygroundPage />
                        </Suspense>
                    } />
                    <Route path="schedules" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <BaselinesPage />
                        </Suspense>
                    } />
                    <Route path="diagnostics" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <DiagnosticsPage />
                        </Suspense>
                    } />
                    <Route path="rust" element={
                        <Suspense fallback={<LoadingSkeleton />}>
                            <RustPage />
                        </Suspense>
                    } />
                    <Route path="*" element={
                        <div style={{ padding: 60, color: "var(--text-2)" }}>404 Not Found</div>
                    } />
                </Route>
            </Routes>
        </Router>
    );
}