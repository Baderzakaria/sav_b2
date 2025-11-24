import { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
  LineChart,
  Line,
} from 'recharts';
import { useData } from '../context/DataContext';
import { MessageSquare } from 'lucide-react';

const COLORS = ['#e11d2d', '#fde68a', '#c084fc', '#38bdf8', '#f97316', '#22c55e', '#f43f5e', '#0ea5e9'];

const formatDateLabel = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString('fr-FR', { month: 'short', year: '2-digit' });
};

export const StatsPage = () => {
  const { tickets } = useData();

  const {
    timelineData,
    sentimentData,
    categoryData,
    typeData,
    severityData,
    utileData,
    channelData,
    gravityBuckets,
    engagementData,
    sentimentGravityData,
    nlpSummary,
  } = useMemo(() => {
    if (tickets.length === 0) {
      return {
        timelineData: [],
        sentimentData: [],
        categoryData: [],
        typeData: [],
        severityData: [],
        utileData: [],
        channelData: [],
        gravityBuckets: [],
        engagementData: [],
        sentimentGravityData: [],
        nlpSummary: "Aucune donnée importée pour l'instant. Utilisez l'import CSV pour lancer l'analyse.",
      };
    }

    const monthMap: Record<string, number> = {};
    tickets.forEach((ticket) => {
      const date = new Date(ticket.created_at);
      if (Number.isNaN(date.getTime())) return;
      const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      monthMap[key] = (monthMap[key] || 0) + 1;
    });
    const timelineData = Object.entries(monthMap)
      .map(([key, value]) => ({ name: key, tickets: value }))
      .sort((a, b) => a.name.localeCompare(b.name));

    const countBy = (items: string[]) =>
      items.reduce<Record<string, number>>((acc, item) => {
        if (!item) return acc;
        const normalized = item.trim().toLowerCase();
        acc[normalized] = (acc[normalized] || 0) + 1;
        return acc;
      }, {});

    const sentimentCounts = countBy(tickets.map((t) => t.sentiment || 'inconnu'));
    const sentimentData = Object.entries(sentimentCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);

    const categoryCounts = countBy(tickets.map((t) => t.categorie || 'autre'));
    const categoryData = Object.entries(categoryCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);

    const typeCounts = countBy(tickets.map((t) => t.type || 'autre'));
    const typeData = Object.entries(typeCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);

    const severityCounts = countBy(tickets.map((t) => t.severity || 'inconnue'));
    const severityData = Object.entries(severityCounts).map(([name, value]) => ({ name, value }));

    const utileCounts = tickets.reduce(
      (acc, ticket) => {
        if (ticket.utile) acc.oui += 1;
        else acc.non += 1;
        return acc;
      },
      { oui: 0, non: 0 }
    );
    const utileData = [
      { name: 'Utile', value: utileCounts.oui },
      { name: 'Non utile', value: utileCounts.non },
    ];

    const channelCounts = countBy(tickets.map((t) => t.channel || 'N/A'));
    const channelData = Object.entries(channelCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);

    const gravityBucketsConfig = [
      { label: '≤ -10', test: (v: number) => v <= -10 },
      { label: '-9 à -5', test: (v: number) => v >= -9 && v <= -5 },
      { label: '-4 à -1', test: (v: number) => v >= -4 && v <= -1 },
      { label: '0', test: (v: number) => v === 0 },
      { label: '1 à 4', test: (v: number) => v >= 1 && v <= 4 },
      { label: '5 à 9', test: (v: number) => v >= 5 && v <= 9 },
      { label: '≥ 10', test: (v: number) => v >= 10 },
    ];
    const gravityBuckets = gravityBucketsConfig.map(({ label, test }) => ({
      label,
      value: tickets.filter((ticket) => typeof ticket.gravity === 'number' && test(ticket.gravity)).length,
    }));

    const engagementData = tickets
      .filter((ticket) => typeof ticket.gravity === 'number')
      .slice(0, 300)
      .map((ticket) => ({
        gravity: ticket.gravity || 0,
        favorites: ticket.favorite_count || 0,
        replies: ticket.reply_count || 0,
      }));

    const sentimentGravityMap: Record<string, { sum: number; count: number }> = {};
    tickets.forEach((ticket) => {
      if (typeof ticket.gravity !== 'number') return;
      const key = ticket.sentiment || 'inconnu';
      if (!sentimentGravityMap[key]) sentimentGravityMap[key] = { sum: 0, count: 0 };
      sentimentGravityMap[key].sum += ticket.gravity;
      sentimentGravityMap[key].count += 1;
    });
    const sentimentGravityData = Object.entries(sentimentGravityMap)
      .map(([name, stats]) => ({
        name,
        value: stats.count ? Math.round((stats.sum / stats.count) * 10) / 10 : 0,
      }))
      .sort((a, b) => b.value - a.value);

    const topSentiment = sentimentData[0]?.name || 'N/A';
    const topCategory = categoryData[0]?.name || 'N/A';
    const criticalCount = tickets.filter((ticket) => Math.abs(ticket.gravity ?? 0) >= 10).length;
    const biggestChannel = channelData[0]?.name || 'N/A';
    const nlpSummary = `Analyse rapide : ${topSentiment} domine les sentiments détectés, avec ${topCategory} comme catégorie la plus fréquente. ${criticalCount} requêtes présentent une gravité critique, principalement sur le canal ${biggestChannel}.`;

    return {
      timelineData,
      sentimentData,
      categoryData,
      typeData,
      severityData,
      utileData,
      channelData,
      gravityBuckets,
      engagementData,
      sentimentGravityData,
      nlpSummary,
    };
  }, [tickets]);

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <MessageSquare className="text-primary" size={24} />
            Statistiques détaillées
          </h2>
          <p className="text-gray-500">10 visualisations pour explorer l’ensemble des colonnes disponibles.</p>
        </div>
        <div className="bg-white rounded-2xl shadow-soft p-4 max-w-xl">
          <p className="text-sm text-gray-600 leading-relaxed">{nlpSummary}</p>
        </div>
      </div>

      {tickets.length === 0 ? (
        <div className="bg-white border border-dashed border-gray-200 rounded-2xl p-10 text-center text-gray-400">
          Importez un CSV pour afficher les statistiques détaillées.
        </div>
      ) : (
        <div className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Volume mensuel</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={timelineData}>
                    <defs>
                      <linearGradient id="volume" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#e11d2d" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#e11d2d" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="name" tickFormatter={formatDateLabel} />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="tickets" stroke="#e11d2d" fillOpacity={1} fill="url(#volume)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Répartition des sentiments</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={sentimentData} dataKey="value" nameKey="name" outerRadius={100} label>
                      {sentimentData.map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Top catégories (IA)</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={categoryData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Top types détectés</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={typeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#f97316" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Sévérité déclarée</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={severityData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90} label>
                      {severityData.map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Tickets utiles vs non utiles</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={utileData} dataKey="value" cx="50%" cy="50%" innerRadius={40} outerRadius={80} label>
                      {utileData.map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Canaux dominants</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={channelData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#22c55e" radius={[0, 8, 8, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Distribution des gravités</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={gravityBuckets}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="label" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#e11d2d" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Gravité vs Engagement</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis type="number" dataKey="gravity" name="Gravité" />
                    <YAxis yAxisId="left" type="number" dataKey="favorites" name="Likes" stroke="#0ea5e9" />
                    <YAxis yAxisId="right" orientation="right" type="number" dataKey="replies" name="Commentaires" stroke="#f97316" />
                    <ZAxis type="number" range={[20, 20]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter yAxisId="left" data={engagementData} fill="#0ea5e9" name="Likes" shape="circle" />
                    <Scatter yAxisId="right" data={engagementData} fill="#f97316" name="Commentaires" shape="triangle" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-soft p-6">
              <h3 className="font-semibold mb-4 text-gray-800">Gravité moyenne par sentiment</h3>
              <div className="h-64">
                <ResponsiveContainer>
                  <LineChart data={sentimentGravityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#f43f5e" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

