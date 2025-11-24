import { useEffect, useState } from 'react';

type ExternalEmbedPageProps = {
  sourceLabel: string;
  targetUrl?: string;
};

export const ExternalEmbedPage = ({ sourceLabel, targetUrl }: ExternalEmbedPageProps) => {
  const canEmbed = Boolean(targetUrl);

  return (
    <div className="flex flex-col gap-6 min-h-[80vh]">
      <header className="space-y-2">
        <h2 className="text-2xl font-bold text-gray-800">
          {sourceLabel} · Embedded preview
        </h2>
        {!canEmbed && (
          <p className="text-sm text-gray-500">
            No endpoint configured yet. Provide an environment variable for this page (for example
            <code> VITE_STREAMLIT_URL </code> or <code> VITE_MLFLOW_URL </code>).
          </p>
        )}
        {canEmbed && (
          <p className="text-sm text-gray-500">
            Loaded from&nbsp;
            <a className="text-primary underline" href={targetUrl} target="_blank" rel="noreferrer">
              {targetUrl}
            </a>
          </p>
        )}
      </header>

      {canEmbed ? (
        <iframe
          title={`Embed ${sourceLabel}`}
          src={targetUrl}
          className="flex-1 min-h-[70vh] rounded-2xl border border-gray-100 shadow-inner bg-white"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-downloads"
        />
      ) : (
        <div className="flex flex-col items-center justify-center flex-1 bg-gray-50 rounded-2xl border border-dashed border-gray-200 text-gray-500">
          <p>Configure the target URL to show {sourceLabel} here.</p>
        </div>
      )}
    </div>
  );
};

export const ExternalRedirectPage = ({ sourceLabel, targetUrl }: ExternalEmbedPageProps) => {
  const [redirected, setRedirected] = useState(false);
  const [blocked, setBlocked] = useState(false);

  useEffect(() => {
    if (targetUrl) {
      const newWindow = window.open(targetUrl, '_blank', 'noopener,noreferrer');
      if (newWindow) {
        setRedirected(true);
      } else {
        setBlocked(true);
      }
    }
  }, [targetUrl]);

  if (!targetUrl) {
    return (
      <div className="flex flex-col gap-4">
        <p className="text-lg font-semibold text-gray-700">Aucun lien configuré pour {sourceLabel}.</p>
        <p className="text-sm text-gray-500">Ajoutez une variable d&apos;environnement pour activer cette redirection.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {redirected && <p className="text-lg font-semibold text-gray-700">{sourceLabel} est ouvert dans un nouvel onglet.</p>}
      {blocked && (
        <p className="text-lg font-semibold text-red-600">
          Le navigateur a bloqué l&apos;ouverture automatique de {sourceLabel}.
        </p>
      )}
      <p className="text-sm text-gray-500">Vous pouvez continuer à travailler ici ou rouvrir la page si nécessaire.</p>
      <a className="inline-flex items-center justify-center px-4 py-2 text-sm font-semibold text-white bg-primary rounded-lg w-fit" href={targetUrl} target="_blank" rel="noreferrer">
        Ouvrir {sourceLabel}
      </a>
    </div>
  );
};

