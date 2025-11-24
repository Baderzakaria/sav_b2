import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, MessageSquare, Users, BarChart3, LogOut, FileUp, Monitor, Activity, Globe } from 'lucide-react';
import { clsx } from 'clsx';

const menuItems = [
  { icon: LayoutDashboard, label: 'Tableau de bord', path: '/' },
  { icon: MessageSquare, label: 'Requêtes clients', path: '/requests' },
  { icon: BarChart3, label: 'Statistiques', path: '/stats' },
  { icon: Users, label: 'Gestion Agents', path: '/agents' },
  { icon: FileUp, label: 'Import Données', path: '/import' },
  { icon: Monitor, label: 'Streamlit', path: '/streamlit' },
  { icon: Activity, label: 'MLflow', path: '/mlflow' },
  { icon: Globe, label: 'Interface', path: '/interface' },
];

export const Sidebar = () => {
  const location = useLocation();

  return (
    <aside className="w-64 h-screen bg-surface border-r border-gray-100 flex flex-col shadow-soft fixed left-0 top-0 z-10">
      <div className="p-6">
        <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          <span className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center text-white text-lg">F</span>
          FreeMind
        </h1>
      </div>

      <nav className="flex-1 px-4 space-y-2 overflow-y-auto">
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200",
                isActive 
                  ? "bg-primary text-white shadow-md shadow-red-200" 
                  : "text-gray-500 hover:bg-gray-50 hover:text-gray-900"
              )}
            >
              <item.icon size={20} />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-100">
        <button className="flex items-center gap-3 px-4 py-3 text-gray-500 hover:text-red-500 transition-colors w-full">
          <LogOut size={20} />
          <span className="font-medium">Déconnexion</span>
        </button>
      </div>
    </aside>
  );
};
