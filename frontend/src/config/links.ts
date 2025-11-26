import { withDefault } from './env';

export const STREAMLIT_URL = withDefault(
  import.meta.env.VITE_STREAMLIT_URL as string | undefined,
  'http://localhost:8501'
);

export const MLFLOW_URL = withDefault(
  import.meta.env.VITE_MLFLOW_URL as string | undefined,
  'http://localhost:5000'
);

export const INTERFACE_URL = withDefault(
  import.meta.env.VITE_INTERFACE_URL as string | undefined,
  'http://localhost:8501'
);

type ExternalRouteConfig = {
  path: string;
  label: string;
  url?: string;
  mode?: 'embed' | 'redirect';
};

export const EXTERNAL_ROUTES: ExternalRouteConfig[] = [
  { path: '/streamlit', label: 'Streamlit', url: STREAMLIT_URL, mode: 'embed' },
  { path: '/mlflow', label: 'MLflow', url: MLFLOW_URL, mode: 'embed' },
];

