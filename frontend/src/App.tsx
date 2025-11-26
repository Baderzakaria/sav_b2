import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { ImportPage } from './pages/ImportPage';
import { RequestsPage } from './pages/RequestsPage';
import { UrgentPage } from './pages/UrgentPage';
import { StatsPage } from './pages/StatsPage';
import { InterfacePage } from './pages/InterfacePage';
import { ChatWidget } from './components/chat/ChatWidget';
import { DataProvider } from './context/DataContext';
import { EXTERNAL_ROUTES } from './config/links';
import { ExternalEmbedPage, ExternalRedirectPage } from './components/ExternalRedirectPage';

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
              <Route path="/interface" element={<InterfacePage />} />
              {EXTERNAL_ROUTES.map(({ path, label, url, mode = 'embed' }) => (
                <Route
                  key={path}
                  path={path}
                  element={
                    mode === 'redirect' ? (
                      <ExternalRedirectPage sourceLabel={label} targetUrl={url} />
                    ) : (
                      <ExternalEmbedPage sourceLabel={label} targetUrl={url} />
                    )
                  }
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
