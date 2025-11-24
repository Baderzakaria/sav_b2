import { createContext, useContext, useMemo, useState, ReactNode } from 'react';

export interface AgentResponses {
  utile?: string;
  categorie?: string;
  sentiment?: string;
  type?: string;
  gravity?: string;
}

export interface Ticket {
  id: string;
  source: string;
  subject: string;
  description: string;
  clean_text?: string;
  customer: string;
  created_at: string; // ISO string
  status: 'nouveau' | 'en_cours' | 'résolu' | 'clos';
  severity: 'faible' | 'moyenne' | 'élevée' | 'critique';
  channel: string;
  // AI Analysis Fields
  type?: string; // A4_type / panne, information, autre, etc.
  sentiment?: string; // Final_sentiment / colere, neutre, satisfaction, etc.
  categorie?: string; // Final_categorie / probleme, retour_client, etc.
  gravity?: number; // Final_gravity score
  favorite_count?: number;
  reply_count?: number;
  utile?: boolean; // Final_utile
  agentResponses?: AgentResponses;
}


interface DataContextType {
  tickets: Ticket[];
  addTickets: (newTickets: Ticket[]) => void;
  stats: {
    total: number;
    urgent: number;
    resolved: number;
    satisfaction: number;
    satisfactionTrend: number;
  };
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const DataProvider = ({ children }: { children: ReactNode }) => {
  const [tickets, setTickets] = useState<Ticket[]>([]);

  const addTickets = (newTickets: Ticket[]) => {
    setTickets(prev => [...newTickets, ...prev]);
  };

  const satisfactionMetrics = useMemo(() => computeSatisfactionMetrics(tickets), [tickets]);

  const stats = useMemo(() => ({
    total: tickets.length,
    urgent: tickets.filter(t => t.severity === 'critique').length, // Only count 'critique' as urgent for the red KPI
    resolved: tickets.filter(t => t.status === 'résolu' || t.status === 'clos').length,
    satisfaction: satisfactionMetrics.score,
    satisfactionTrend: satisfactionMetrics.trend,
  }), [tickets, satisfactionMetrics]);

  return (
    <DataContext.Provider value={{ tickets, addTickets, stats }}>
      {children}
    </DataContext.Provider>
  );
};

const sentimentScoreMap: Record<string, number> = {
  satisfaction: 5,
  joie: 4.5,
  positif: 4.5,
  positive: 4.5,
  enthousiasme: 4,
  neutre: 3,
  question: 3,
  information: 3,
  inquietude: 2,
  frustration: 2,
  deception: 2,
  desappointement: 2,
  mecontentement: 1.5,
  colere: 1,
  mecontent: 1.5,
};

const normalizeSentiment = (value?: string) => {
  if (!value) return '';
  return value
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
};

const getSentimentScore = (sentiment?: string) => {
  const normalized = normalizeSentiment(sentiment);
  return sentimentScoreMap[normalized] ?? 2.5;
};

const computeSatisfactionMetrics = (tickets: Ticket[]) => {
  const ticketsWithSentiment = tickets.filter(t => t.sentiment);
  if (ticketsWithSentiment.length === 0) {
    return { score: 0, trend: 0 };
  }

  const totalScore = ticketsWithSentiment.reduce((acc, ticket) => acc + getSentimentScore(ticket.sentiment), 0);
  const averageScore = totalScore / ticketsWithSentiment.length;
  const roundedScore = Math.round(averageScore * 10) / 10;

  const monthlyBuckets = ticketsWithSentiment.reduce((acc, ticket) => {
    const date = new Date(ticket.created_at);
    if (isNaN(date.getTime())) {
      return acc;
    }
    const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    if (!acc[key]) {
      acc[key] = { sum: 0, count: 0 };
    }
    acc[key].sum += getSentimentScore(ticket.sentiment);
    acc[key].count += 1;
    return acc;
  }, {} as Record<string, { sum: number; count: number }>);

  const sortedMonths = Object.entries(monthlyBuckets).sort((a, b) => a[0].localeCompare(b[0]));
  if (sortedMonths.length < 2) {
    return { score: roundedScore, trend: 0 };
  }

  const last = sortedMonths[sortedMonths.length - 1][1];
  const previous = sortedMonths[sortedMonths.length - 2][1];
  const lastAvg = last.sum / last.count;
  const previousAvg = previous.sum / previous.count;
  const trend = Math.round((lastAvg - previousAvg) * 10) / 10;

  return { score: roundedScore, trend };
};

export const useData = () => {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};

