import { useMemo, useState } from 'react';
import { MessageSquare, Users, Activity, ListChecks } from 'lucide-react';
import { useData, Ticket } from '../context/DataContext';
import { KpiCard } from '../components/ui/KpiCard';

const formatBoolLabel = (value?: string) => {
  if (!value) return '—';
  const normalized = value.trim().toLowerCase();
  if (['true', 'vrai', 'yes', '1'].includes(normalized)) return 'Oui';
  if (['false', 'faux', 'no', '0'].includes(normalized)) return 'Non';
  return value;
};

const formatDate = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric' });
};

export const RequestsPage = () => {
  const { tickets } = useData();
  const [searchTerm, setSearchTerm] = useState('');
  const [utileFilter, setUtileFilter] = useState<'all' | 'oui' | 'non'>('all');
  const [sortConfig, setSortConfig] = useState<{ key: keyof Ticket | 'agentCategory' | 'agentSentiment' | 'agentType' | 'agentGravity' | 'agentUtile'; direction: 'asc' | 'desc' }>({
    key: 'created_at',
    direction: 'desc',
  });

  const agentStats = useMemo(() => {
    if (tickets.length === 0) {
      return {
        total: 0,
        utileCount: 0,
        utileRate: 0,
        categoryCount: 0,
        avgGravity: 0,
      };
    }

    const utileCount = tickets.filter((ticket) => ticket.agentResponses?.utile?.toLowerCase() === 'true').length;
    const utileRate = Math.round((utileCount / tickets.length) * 100);
    const categories = new Set(
      tickets
        .map((ticket) => ticket.agentResponses?.categorie?.trim())
        .filter((value): value is string => Boolean(value))
    );
    const gravityValues = tickets
      .map((ticket) => Number(ticket.agentResponses?.gravity))
      .filter((value) => !Number.isNaN(value));
    const avgGravity = gravityValues.length
      ? Math.round((gravityValues.reduce((acc, value) => acc + value, 0) / gravityValues.length) * 10) / 10
      : 0;

    return {
      total: tickets.length,
      utileCount,
      utileRate,
      categoryCount: categories.size,
      avgGravity,
    };
  }, [tickets]);

  const recentTickets = useMemo(
    () =>
      [...tickets]
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, 50),
    [tickets]
  );

  const filteredTickets = useMemo(() => {
    return recentTickets.filter((ticket) => {
      const haystack = `${ticket.customer || ''} ${ticket.subject || ''} ${ticket.agentResponses?.categorie || ''} ${ticket.agentResponses?.sentiment || ''}`.toLowerCase();
      const matchesSearch = haystack.includes(searchTerm.toLowerCase().trim());
      if (!matchesSearch) return false;
      if (utileFilter === 'all') return true;
      const raw = ticket.agentResponses?.utile?.toLowerCase();
      if (utileFilter === 'oui') return raw === 'true' || raw === 'vrai' || raw === 'yes' || raw === '1';
      if (utileFilter === 'non') return raw === 'false' || raw === 'faux' || raw === 'no' || raw === '0';
      return true;
    });
  }, [recentTickets, searchTerm, utileFilter]);

  const sortedTickets = useMemo(() => {
    const sorted = [...filteredTickets];
    sorted.sort((a, b) => {
      const direction = sortConfig.direction === 'asc' ? 1 : -1;
      const getValue = (ticket: typeof a) => {
        switch (sortConfig.key) {
          case 'created_at':
            return new Date(ticket.created_at).getTime();
          case 'customer':
            return ticket.customer || '';
          case 'subject':
            return ticket.subject || '';
          case 'agentCategory':
            return ticket.agentResponses?.categorie || '';
          case 'agentSentiment':
            return ticket.agentResponses?.sentiment || '';
          case 'agentType':
            return ticket.agentResponses?.type || '';
          case 'agentGravity':
            return ticket.agentResponses?.gravity || '';
          case 'agentUtile': {
            const raw = ticket.agentResponses?.utile?.toLowerCase();
            if (raw === 'true' || raw === 'vrai' || raw === 'yes' || raw === '1') return 1;
            if (raw === 'false' || raw === 'faux' || raw === 'no' || raw === '0') return 0;
            return -1;
          }
          default:
            return '';
        }
      };

      const valueA = getValue(a);
      const valueB = getValue(b);

      if (typeof valueA === 'number' && typeof valueB === 'number') {
        return (valueA - valueB) * direction;
      }

      return String(valueA).localeCompare(String(valueB)) * direction;
    });
    return sorted;
  }, [filteredTickets, sortConfig]);

  const handleSort = (key: typeof sortConfig.key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key, direction: 'asc' };
    });
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <MessageSquare className="text-primary" size={24} />
            Requêtes clients
          </h2>
          <p className="text-gray-500">
            Suivi des réponses agents A1 → A5 pour les requêtes importées
          </p>
        </div>
        {tickets.length > 0 && (
          <span className="text-sm text-gray-400">
            {tickets.length} requêtes au total — affichage des {recentTickets.length} plus récentes
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KpiCard
          title="Requêtes agentées"
          value={agentStats.total}
          trend="Agents IA"
          trendUp={true}
          icon={<Users size={24} />}
          color="blue"
        />
        <KpiCard
          title="Utile (Agent A1)"
          value={agentStats.utileCount}
          trend={`${agentStats.utileRate}%`}
          trendUp={agentStats.utileRate >= 50}
          icon={<ListChecks size={24} />}
          color="green"
        />
        <KpiCard
          title="Catégories (Agent A2)"
          value={agentStats.categoryCount}
          trend="Détection"
          trendUp
          icon={<MessageSquare size={24} />}
          color="orange"
        />
        <KpiCard
          title="Gravité moyenne (A5)"
          value={tickets.length > 0 ? agentStats.avgGravity.toFixed(1) : 'N/A'}
          trend={agentStats.avgGravity >= 0 ? 'Stable' : 'Alerte'}
          trendUp={agentStats.avgGravity >= 0}
          icon={<Activity size={24} />}
          color="primary"
        />
      </div>

      <div className="bg-white rounded-2xl shadow-soft p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-gray-800">Réponses agents par requête</h3>
            <p className="text-sm text-gray-500">Les colonnes A1 → A5 reflètent la sortie brute de chaque agent.</p>
          </div>
          <div className="flex flex-col md:flex-row gap-3 w-full md:w-auto">
            <input
              type="text"
              placeholder="Filtrer par mot-clé (client, sujet, sentiment...)"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full md:w-72 px-3 py-2 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
            <select
              value={utileFilter}
              onChange={(e) => setUtileFilter(e.target.value as typeof utileFilter)}
              className="px-3 py-2 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="all">Tous les résultats A1</option>
              <option value="oui">A1 = Oui</option>
              <option value="non">A1 = Non</option>
            </select>
          </div>
        </div>

        {sortedTickets.length === 0 ? (
          <p className="text-gray-400 text-center py-10">Aucune requête importée pour l'instant.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-100">
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('created_at')}>
                    Date {sortConfig.key === 'created_at' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('customer')}>
                    Client {sortConfig.key === 'customer' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('subject')}>
                    Sujet {sortConfig.key === 'subject' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('agentUtile')}>
                    A1 · Utile {sortConfig.key === 'agentUtile' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('agentCategory')}>
                    A2 · Catégorie {sortConfig.key === 'agentCategory' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('agentSentiment')}>
                    A3 · Sentiment {sortConfig.key === 'agentSentiment' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('agentType')}>
                    A4 · Type {sortConfig.key === 'agentType' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="py-3 pr-4 font-medium cursor-pointer select-none" onClick={() => handleSort('agentGravity')}>
                    A5 · Gravité {sortConfig.key === 'agentGravity' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedTickets.map((ticket) => (
                  <tr key={ticket.id} className="border-b border-gray-50 hover:bg-gray-50/70 transition-colors">
                    <td className="py-3 pr-4 text-gray-700">{formatDate(ticket.created_at)}</td>
                    <td className="py-3 pr-4 text-gray-700">{ticket.customer || '—'}</td>
                    <td className="py-3 pr-4 text-gray-500 max-w-[220px] truncate" title={ticket.subject}>
                      {ticket.subject}
                    </td>
                    <td className="py-3 pr-4 font-semibold text-gray-800">
                      {formatBoolLabel(ticket.agentResponses?.utile)}
                    </td>
                    <td className="py-3 pr-4 text-gray-800">{ticket.agentResponses?.categorie || '—'}</td>
                    <td className="py-3 pr-4 text-gray-800">{ticket.agentResponses?.sentiment || '—'}</td>
                    <td className="py-3 pr-4 text-gray-800">{ticket.agentResponses?.type || '—'}</td>
                    <td className="py-3 pr-4 text-gray-800">{ticket.agentResponses?.gravity || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

