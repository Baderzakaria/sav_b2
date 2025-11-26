import { withDefault } from './env';

const RAW_BACKEND_URL = withDefault(
  import.meta.env.VITE_BACKEND_URL as string | undefined,
  'http://localhost:8000'
);

export const BACKEND_URL = RAW_BACKEND_URL.replace(/\/$/, '');

export const apiPath = (path: string) => {
  if (!path.startsWith('/')) {
    return `${BACKEND_URL}/${path}`;
  }
  return `${BACKEND_URL}${path}`;
};
