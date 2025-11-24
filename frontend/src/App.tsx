import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { ImportPage } from './pages/ImportPage';
import { RequestsPage } from './pages/RequestsPage';
import { UrgentPage } from './pages/UrgentPage';
import { StatsPage } from './pages/StatsPage';
import { ChatWidget } from './components/chat/ChatWidget';
import { DataProvider } from './context/DataContext';
import { EXTERNAL_ROUTES } from './config/links';
import { ExternalRedirectPage } from './components/ExternalRedirectPage';

function App() {
  return (
    <DataProvider>
      <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <div className="flex min-h-screen bg-background font-sans text-gray-900">
          <Sidebar />
          
          {/* Main Content */}
          <main className="flex-1 ml-64 p-8">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/requests" element={<RequestsPage />} />
              <Route path="/urgent" element={<UrgentPage />} />
              <Route path="/stats" element={<StatsPage />} />
              <Route path="/import" element={<ImportPage />} />
              {EXTERNAL_ROUTES.map(({ path, label }) => (
                <Route
                  key={path}
                  path={path}
                  element={<ExternalRedirectPage sourceLabel={label} />}
                />
              ))}
              <Route path="*" element={<div className="p-10">Page en construction...</div>} />
            </Routes>
          </main>

          {/* Widget Chat Global (pour démo) */}
          <ChatWidget />
        </div>
      </Router>
    </DataProvider>
  );
}

export default App;
