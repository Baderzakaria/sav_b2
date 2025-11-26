const normalize = (value?: string) => {
  if (!value) return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const RAW_BACKEND_URL = normalize(import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://localhost:8000';
const BASE = RAW_BACKEND_URL.replace(/\/$/, '');

export const apiPath = (path: string) => {
  if (!path.startsWith('/')) {
    return `${BASE}/${path}`;
  }
  return `${BASE}${path}`;
};

export const BACKEND_URL = BASE;
import { withDefault } from './env';

export const BACKEND_URL = withDefault(
  import.meta.env.VITE_BACKEND_URL as string | undefined,
  'http://localhost:8000'
);


