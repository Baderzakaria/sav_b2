import { useEffect } from 'react';

import { SHARED_PORTAL_URL } from '../config/links';

type ExternalRedirectProps = {
  sourceLabel: string;
};

export const ExternalRedirectPage = ({ sourceLabel }: ExternalRedirectProps) => {
  useEffect(() => {
    if (SHARED_PORTAL_URL) {
      window.location.replace(SHARED_PORTAL_URL);
    }
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center gap-4">
      <h2 className="text-2xl font-semibold text-gray-800">Redirection en cours…</h2>
      <p className="text-gray-500 max-w-xl">
        La page <strong>{sourceLabel}</strong> redirige automatiquement vers&nbsp;
        <a
          href={SHARED_PORTAL_URL}
          className="text-primary underline"
          target="_blank"
          rel="noreferrer"
        >
          {SHARED_PORTAL_URL}
        </a>
        . Cliquez sur le lien si la redirection ne se lance pas.
      </p>
    </div>
  );
};

