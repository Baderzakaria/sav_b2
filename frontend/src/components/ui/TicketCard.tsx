import { Ticket } from '../../context/DataContext';
import { Heart, MessageSquare } from 'lucide-react';
import { clsx } from 'clsx';

interface TicketCardProps {
  ticket: Ticket;
}

const sentimentColors: Record<string, string> = {
  'colere': 'bg-red-100 text-red-700 border-red-200',
  'frustration': 'bg-orange-100 text-orange-700 border-orange-200',
  'deception': 'bg-yellow-100 text-yellow-700 border-yellow-200',
  'neutre': 'bg-gray-100 text-gray-700 border-gray-200',
  'satisfaction': 'bg-green-100 text-green-700 border-green-200',
  'enthousiasme': 'bg-blue-100 text-blue-700 border-blue-200',
  'inquietude': 'bg-purple-100 text-purple-700 border-purple-200',
};

const typeColors: Record<string, string> = {
  'panne': 'bg-red-50 text-red-600',
  'information': 'bg-blue-50 text-blue-600',
  'autre': 'bg-gray-50 text-gray-600',
  'retour_client': 'bg-green-50 text-green-600',
};

const severityColors: Record<string, string> = {
  'critique': 'bg-red-500 text-white',
  'élevée': 'bg-orange-500 text-white',
  'moyenne': 'bg-yellow-500 text-white',
  'faible': 'bg-green-500 text-white',
};

export const TicketCard = ({ ticket }: TicketCardProps) => {
  const sentimentClass = ticket.sentiment 
    ? sentimentColors[ticket.sentiment.split('|')[0]] || sentimentColors['neutre']
    : sentimentColors['neutre'];
  
  const typeClass = ticket.type 
    ? typeColors[ticket.type] || typeColors['autre']
    : typeColors['autre'];

  return (
    <div className="bg-white rounded-xl shadow-soft p-5 hover:shadow-soft-xl transition-all duration-200 border border-gray-100">
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <h4 className="font-bold text-gray-800 text-sm mb-1 line-clamp-1">{ticket.subject}</h4>
          <p className="text-xs text-gray-500 mb-2 line-clamp-2">{ticket.description}</p>
        </div>
        <span className={clsx("px-2 py-1 rounded-lg text-xs font-bold shrink-0 ml-2", severityColors[ticket.severity])}>
          {ticket.severity}
        </span>
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        {ticket.type && (
          <span className={clsx("px-2 py-1 rounded-md text-xs font-medium", typeClass)}>
            {ticket.type}
          </span>
        )}
        {ticket.sentiment && (
          <span className={clsx("px-2 py-1 rounded-md text-xs font-medium border", sentimentClass)}>
            {ticket.sentiment.split('|')[0]}
          </span>
        )}
        {ticket.categorie && (
          <span className="px-2 py-1 rounded-md text-xs font-medium bg-purple-50 text-purple-600">
            {ticket.categorie}
          </span>
        )}
      </div>

      <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-gray-100">
        <div className="flex items-center gap-4">
          <span className="font-medium text-gray-700">{ticket.customer}</span>
          <span>{new Date(ticket.created_at).toLocaleDateString('fr-FR')}</span>
        </div>
        <div className="flex items-center gap-3">
          {ticket.favorite_count !== undefined && (
            <span className="flex items-center gap-1">
              <Heart size={14} />
              {ticket.favorite_count}
            </span>
          )}
          {ticket.reply_count !== undefined && (
            <span className="flex items-center gap-1">
              <MessageSquare size={14} />
              {ticket.reply_count}
            </span>
          )}
          {ticket.gravity !== undefined && (
            <span className={clsx(
              "px-2 py-0.5 rounded text-xs font-bold",
              ticket.gravity < 0 ? "bg-red-100 text-red-600" : "bg-green-100 text-green-600"
            )}>
              {ticket.gravity > 0 ? '+' : ''}{ticket.gravity}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

