import { ReactNode } from 'react';

interface KpiCardProps {
  title: string;
  value: string | number;
  trend?: string;
  trendUp?: boolean;
  icon: ReactNode;
  color?: 'primary' | 'blue' | 'green' | 'orange';
}

const colorMap = {
  primary: 'bg-red-50 text-red-500',
  blue: 'bg-blue-50 text-blue-500',
  green: 'bg-emerald-50 text-emerald-500',
  orange: 'bg-orange-50 text-orange-500',
};

export const KpiCard = ({ title, value, trend, trendUp, icon, color = 'primary' }: KpiCardProps) => {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-soft hover:shadow-soft-xl transition-shadow duration-300">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
          <h3 className="text-2xl font-bold text-gray-800">{value}</h3>
          {trend && (
            <p className={`text-xs mt-2 font-semibold ${trendUp ? 'text-green-500' : 'text-red-500'}`}>
              {trendUp ? '+' : ''}{trend}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-xl ${colorMap[color]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

