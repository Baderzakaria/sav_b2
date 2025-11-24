const DEFAULT_PORTAL_URL = 'http://localhost:8501';

const envPortalUrl = (import.meta.env.VITE_SAV_PORTAL_URL as string | undefined)?.trim();

export const SHARED_PORTAL_URL = envPortalUrl && envPortalUrl.length > 0 ? envPortalUrl : DEFAULT_PORTAL_URL;

export const EXTERNAL_ROUTES = [
  { path: '/streamlit', label: 'Streamlit' },
  { path: '/mlflow', label: 'MLflow' },
  { path: '/interface', label: 'Interface' },
];

