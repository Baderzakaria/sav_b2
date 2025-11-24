import { useMemo } from 'react';
import { AlertTriangle, MessageCircleOff } from 'lucide-react';
import { useData } from '../context/DataContext';

const normalize = (value?: string) =>
  value
    ? value
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
    : '';

export const UrgentPage = () => {
  const { tickets } = useData();

  const urgentTickets = useMemo(() => {
    return tickets
      .filter((ticket) => {
        const gravity = ticket.gravity ?? 0;
        const isSevere = Math.abs(gravity) >= 10;
        const typeNorm = normalize(ticket.type);
        const categoryNorm = normalize(ticket.categorie);
        const isAlert = typeNorm.includes('alert') || categoryNorm.includes('alert');
        const isPanne = typeNorm.includes('panne') || categoryNorm.includes('panne');
        return isSevere && (isAlert || isPanne);
      })
      .sort((a, b) => Math.abs((b.gravity ?? 0)) - Math.abs((a.gravity ?? 0)));
  }, [tickets]);

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <AlertTriangle className="text-primary" size={24} />
            Urgences
          </h2>
          <p className="text-gray-500">
            Tickets alertes & pannes dont la gravité est évaluée à 10 (seuil critique).
          </p>
        </div>
        <div className="text-sm text-gray-400">
          {urgentTickets.length} ticket{urgentTickets.length > 1 ? 's' : ''} critiques détectés
        </div>
      </div>

      {urgentTickets.length === 0 ? (
        <div className="bg-white border border-dashed border-gray-200 rounded-2xl p-10 text-center text-gray-400">
          Aucun ticket alerte/panne avec gravité 10 pour le moment.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {urgentTickets.map((ticket) => (
            <div key={ticket.id} className="bg-white rounded-2xl shadow-soft p-6 space-y-4 border border-red-100">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs uppercase text-red-500 font-semibold tracking-wide">Panne / Alerte</p>
                  <h3 className="text-lg font-bold text-gray-900 mt-1">{ticket.subject || 'Ticket sans sujet'}</h3>
                  <p className="text-sm text-gray-500">Client : {ticket.customer || 'Inconnu'}</p>
                </div>
                <div className="text-right">
                  <span className="inline-flex items-center text-sm font-bold text-red-600">
                    Gravité&nbsp;
                    <span className="text-2xl leading-none">{ticket.gravity ?? '—'}</span>
                  </span>
                  <p className="text-xs text-gray-400">Type IA : {ticket.type || '—'}</p>
                </div>
              </div>

              <p className="text-gray-700 text-sm leading-relaxed line-clamp-3">{ticket.description}</p>

              <div className="flex flex-wrap gap-3">
                {ticket.sentiment && (
                  <span className="px-3 py-1 text-xs rounded-full bg-gray-100 text-gray-700">
                    Sentiment : {ticket.sentiment}
                  </span>
                )}
                {ticket.categorie && (
                  <span className="px-3 py-1 text-xs rounded-full bg-gray-100 text-gray-700">
                    Catégorie : {ticket.categorie}
                  </span>
                )}
              </div>

              <button
                type="button"
                disabled
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-400 rounded-xl text-sm font-semibold cursor-not-allowed"
                title="Simulation de réponse agent Free (en cours de développement)"
              >
                <MessageCircleOff size={16} />
                Réponse Free (indispo)
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

