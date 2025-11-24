import { useMemo } from 'react';
import { KpiCard } from '../components/ui/KpiCard';
import { Users, Clock, AlertOctagon, CheckCircle, TrendingDown, Smile } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useData } from '../context/DataContext';

export const DashboardPage = () => {
  const { tickets, stats } = useData();

  // Calculate Chart Data (Tickets over time)
  const chartData = useMemo(() => {
    if (tickets.length === 0) return [];
    
    // Group by Date (YYYY-MM-DD) or Month depending on range
    // For this demo with the specific CSV, let's group by Month-Year
    const grouped = tickets.reduce((acc, ticket) => {
      const date = new Date(ticket.created_at);
      const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`; // YYYY-MM
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(grouped)
      .map(([name, tickets]) => ({ name, tickets }))
      .sort((a, b) => a.name.localeCompare(b.name))
      .slice(-12); // Last 12 months/periods for readability
  }, [tickets]);

  // Calculate Channels Distribution
  const channelsData = useMemo(() => {
    const total = tickets.length;
    if (total === 0) return [];

    const counts = tickets.reduce((acc, ticket) => {
      acc[ticket.channel] = (acc[ticket.channel] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(counts)
      .map(([name, count]) => ({
        name,
        count,
        percent: Math.round((count / total) * 100)
      }))
      .sort((a, b) => b.count - a.count);
  }, [tickets]);

  // Calculate Sentiment Distribution
  const sentimentData = useMemo(() => {
    if (tickets.length === 0) return [];
    const counts = tickets.reduce((acc, ticket) => {
      if (ticket.sentiment) {
        const sentiments = ticket.sentiment.split('|');
        sentiments.forEach(s => {
          acc[s] = (acc[s] || 0) + 1;
        });
      }
      return acc;
    }, {} as Record<string, number>);

    const colors = ['#ef4444', '#f97316', '#eab308', '#64748b', '#22c55e', '#3b82f6', '#a855f7'];
    return Object.entries(counts)
      .map(([name, value], idx) => ({ name, value, color: colors[idx % colors.length] }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 7);
  }, [tickets]);

  // Calculate Type Distribution
  const typeData = useMemo(() => {
    if (tickets.length === 0) return [];
    const counts = tickets.reduce((acc, ticket) => {
      if (ticket.type) {
        acc[ticket.type] = (acc[ticket.type] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [tickets]);

  // Calculate Category Distribution
  const categoryData = useMemo(() => {
    if (tickets.length === 0) return [];
    const counts = tickets.reduce((acc, ticket) => {
      if (ticket.categorie) {
        acc[ticket.categorie] = (acc[ticket.categorie] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [tickets]);

  // Calculate Average Gravity
  const avgGravity = useMemo(() => {
    const ticketsWithGravity = tickets.filter(t => t.gravity !== undefined);
    if (ticketsWithGravity.length === 0) return 0;
    const sum = ticketsWithGravity.reduce((acc, t) => acc + (t.gravity || 0), 0);
    return Math.round((sum / ticketsWithGravity.length) * 10) / 10;
  }, [tickets]);

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h2 className="text-2xl font-bold text-gray-800">Tableau de Bord SAV</h2>
        <p className="text-gray-500">
          Vue d'ensemble des performances {tickets.length > 0 ? `(${tickets.length} tickets analysés)` : ''}
        </p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KpiCard 
          title="Total Requêtes" 
          value={stats.total} 
          trend={tickets.length > 0 ? "+100%" : "0%"} 
          trendUp={true} 
          icon={<Users size={24} />} 
          color="blue"
        />
        <KpiCard 
          title="Urgences" 
          value={stats.urgent} 
          trend={stats.urgent > 0 ? "Attention" : "Stable"} 
          trendUp={stats.urgent === 0} 
          icon={<AlertOctagon size={24} />} 
          color="primary"
        />
        <KpiCard 
          title="Temps de réponse Moy." 
          value="4h 12m" 
          trend="-30m" 
          trendUp={true} 
          icon={<Clock size={24} />} 
          color="orange"
        />
        <KpiCard 
          title="Score Satisfaction" 
          value={tickets.some(t => t.sentiment) ? `${stats.satisfaction.toFixed(1)}/5` : 'N/A'} 
          trend={stats.satisfactionTrend.toFixed(1)} 
          trendUp={stats.satisfactionTrend >= 0} 
          icon={<CheckCircle size={24} />} 
          color="green"
        />
      </div>

      {/* AI Analysis KPIs */}
      {tickets.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <KpiCard 
            title="Gravité Moyenne (IA)" 
            value={avgGravity > 0 ? `+${avgGravity}` : `${avgGravity}`} 
            trend={avgGravity < 0 ? "Problèmes détectés" : "Stable"} 
            trendUp={avgGravity >= 0} 
            icon={<TrendingDown size={24} />} 
            color={avgGravity < 0 ? "primary" : "green"}
          />
          <KpiCard 
            title="Tickets Utiles (IA)" 
            value={tickets.filter(t => t.utile).length} 
            trend={`${Math.round((tickets.filter(t => t.utile).length / tickets.length) * 100)}%`} 
            trendUp={true} 
            icon={<Smile size={24} />} 
            color="blue"
          />
          <KpiCard 
            title="Types Détectés" 
            value={typeData.length} 
            trend="Par IA" 
            trendUp={true} 
            icon={<CheckCircle size={24} />} 
            color="orange"
          />
        </div>
      )}

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-soft">
          <h3 className="text-lg font-bold text-gray-800 mb-6">Évolution des requêtes (12 derniers mois)</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData.length > 0 ? chartData : [{name: 'No Data', tickets: 0}]}>
                <defs>
                  <linearGradient id="colorTickets" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#e11d2d" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#e11d2d" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#9ca3af'}} />
                <Tooltip 
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'}}
                />
                <Area 
                  type="monotone" 
                  dataKey="tickets" 
                  stroke="#e11d2d" 
                  strokeWidth={3}
                  fillOpacity={1} 
                  fill="url(#colorTickets)" 
                  animationDuration={1500}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-soft">
          <h3 className="text-lg font-bold text-gray-800 mb-6">Canaux de contact</h3>
          <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
             {channelsData.length > 0 ? channelsData.map((channel) => (
               <div key={channel.name} className="flex items-center justify-between group">
                 <span className="text-gray-600 text-sm font-medium w-24 truncate" title={channel.name}>
                   {channel.name}
                 </span>
                 <div className="flex-1 mx-4 h-2 bg-gray-100 rounded-full overflow-hidden">
                   <div 
                     className="h-full bg-gray-800 rounded-full transition-all duration-1000 group-hover:bg-primary" 
                     style={{ width: `${channel.percent}%` }}
                   />
                 </div>
                 <span className="text-gray-800 font-bold text-sm min-w-[3rem] text-right">
                   {channel.percent}%
                 </span>
               </div>
             )) : (
               <p className="text-gray-400 text-sm text-center py-10">Aucune donnée canal disponible</p>
             )}
          </div>
        </div>
      </div>

      {/* AI Analysis Charts */}
      {tickets.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sentiment Distribution */}
          <div className="bg-white p-6 rounded-2xl shadow-soft">
            <h3 className="text-lg font-bold text-gray-800 mb-6">Sentiments (IA)</h3>
            {sentimentData.length > 0 ? (
              <div className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sentimentData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {sentimentData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-gray-400 text-sm text-center py-10">Aucun sentiment détecté</p>
            )}
          </div>

          {/* Type Distribution */}
          <div className="bg-white p-6 rounded-2xl shadow-soft">
            <h3 className="text-lg font-bold text-gray-800 mb-6">Types de Tickets (IA)</h3>
            {typeData.length > 0 ? (
              <div className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={typeData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 11}} angle={-45} textAnchor="end" height={80} />
                    <YAxis axisLine={false} tickLine={false} tick={{fill: '#9ca3af'}} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#e11d2d" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-gray-400 text-sm text-center py-10">Aucun type détecté</p>
            )}
          </div>

          {/* Category Distribution */}
          <div className="bg-white p-6 rounded-2xl shadow-soft">
            <h3 className="text-lg font-bold text-gray-800 mb-6">Catégories (IA)</h3>
            <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2">
              {categoryData.length > 0 ? categoryData.map((cat) => {
                const percent = Math.round((cat.value / tickets.length) * 100);
                return (
                  <div key={cat.name} className="flex items-center justify-between group">
                    <span className="text-gray-600 text-sm font-medium flex-1 truncate" title={cat.name}>
                      {cat.name}
                    </span>
                    <div className="flex-1 mx-3 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-purple-500 rounded-full transition-all duration-1000 group-hover:bg-purple-600" 
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                    <span className="text-gray-800 font-bold text-sm min-w-[3rem] text-right">
                      {cat.value}
                    </span>
                  </div>
                );
              }) : (
                <p className="text-gray-400 text-sm text-center py-10">Aucune catégorie détectée</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
