import { useEffect, useMemo, useState } from 'react';
import { Play, RefreshCw, AlertCircle, Star } from 'lucide-react';
import { apiPath } from '../config/api';

const MAX_INTERFACE_SAMPLES = Number(import.meta.env.VITE_INTERFACE_TEST_MAX ?? '20');

type Example = {
  id: string;
  title: string;
  text: string;
  category: string;
};

type AgentOutputs = {
  A1?: string;
  A2?: string;
  A3?: string;
  A4?: string;
  A5?: string;
};

type InterfaceResult = {
  id?: string;
  text: string;
  latency_ms?: number;
  final?: Record<string, unknown>;
  agents?: AgentOutputs;
  error?: string;
};

type Ticket = {
  id: string;
  subject: string;
  categorie: string;
  sentiment: string;
  severity: string;
  gravity: number;
  description: string;
  created_at: string;
};

const DEFAULT_EXAMPLES: Example[] = [
  {
    id: 'tv-blackout',
    title: 'Coupure totale TV',
    category: 'panne',
    text: 'Ma Freebox Delta est bloquée sur l’étape 2 depuis hier, impossible d’avoir la TV ni Internet. C’est inacceptable.',
  },
  {
    id: 'billing-confusion',
    title: 'Facturation incompréhensible',
    category: 'facturation',
    text: 'Pourquoi je paye deux fois le même mois ? Mon espace abonné affiche une ligne “ajustement” en plus, ça sort d’où ?',
  },
  {
    id: 'fiber-delay',
    title: 'Raccordement fibre',
    category: 'information',
    text: 'Le technicien devait passer aujourd’hui pour raccorder la fibre mais personne n’est venu. Pouvez-vous reprogrammer rapidement ?',
  },
  {
    id: 'wifi-slow',
    title: 'Wi-Fi très lent',
    category: 'probleme',
    text: 'Depuis la dernière mise à jour le Wi-Fi Freebox Pop est inutilisable, à peine 2 Mbps. Vous avez une solution ?',
  },
  {
    id: 'mobile-activation',
    title: 'Activation carte SIM',
    category: 'question',
    text: 'Je viens de recevoir ma carte SIM et elle reste inactive même après 24h. Quelle manipulation faut-il faire ?',
  },
  {
    id: 'move-request',
    title: 'Déménagement Freebox',
    category: 'information',
    text: 'Je déménage à la fin du mois, comment transférer mon abonnement Freebox Révolution sans coupure ?',
  },
  {
    id: 'streaming-issue',
    title: 'Netflix bugué',
    category: 'retour_client',
    text: "Netflix plante toutes les 5 minutes sur l'appli Freebox. Même problème chez d'autres ?",
  },
  {
    id: 'angry-customer',
    title: 'Colère client',
    category: 'probleme',
    text: "Encore des coupures réseau ce matin ! J'en ai marre de Free, vous ne respectez pas les clients pro.",
  },
  {
    id: 'positive-feedback',
    title: 'Retour positif',
    category: 'retour_client',
    text: "Bravo Free pour le nouveau portail SAV, super clair et rapide d'utilisation 👏.",
  },
  {
    id: 'subscription-question',
    title: 'Offre multi-lignes',
    category: 'question',
    text: "Peut-on associer deux lignes mobiles sur la même Freebox Pop ? J'ai besoin d'une info avant de commander.",
  },
];

const historySeverityClasses: Record<string, string> = {
  critique: 'bg-red-100 text-red-800',
  élevée: 'bg-orange-100 text-orange-800',
  moyenne: 'bg-amber-100 text-amber-900',
  faible: 'bg-emerald-100 text-emerald-700',
};

const ResultBadge = ({ label }: { label: string }) => (
  <span className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600 uppercase tracking-wide">{label}</span>
);

export const InterfacePage = () => {
  const [selectedExamples, setSelectedExamples] = useState<string[]>([DEFAULT_EXAMPLES[0].id]);
  const [customText, setCustomText] = useState('');
  const [results, setResults] = useState<InterfaceResult[]>([]);
  const [history, setHistory] = useState<Ticket[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedCount = selectedExamples.length + (customText.trim() ? 1 : 0);

  const activeExamples = useMemo(() => {
    return DEFAULT_EXAMPLES.filter((example) => selectedExamples.includes(example.id));
  }, [selectedExamples]);

  const loadHistory = async () => {
    try {
      const resp = await fetch(apiPath('/tickets/latest?limit=15'));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setHistory(data.tickets ?? []);
    } catch (err) {
      console.error('Failed to load history', err);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleToggleExample = (id: string) => {
    setSelectedExamples((prev) =>
      prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id],
    );
  };

  const handleRun = async () => {
    const samples = [
      ...activeExamples.map((example) => ({ id: example.id, text: example.text })),
      ...(customText.trim()
        ? [
            {
              id: 'custom',
              text: customText.trim(),
            },
          ]
        : []),
    ];

    if (samples.length === 0) {
      setError('Sélectionnez au moins un exemple ou saisissez un texte.');
      return;
    }

    setIsRunning(true);
    setError(null);
    try {
      const resp = await fetch(apiPath('/interface-tests'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples }),
      });
      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error(msg || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      setResults(data.results ?? []);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Impossible de lancer les tests');
    } finally {
      setIsRunning(false);
    }
  };

  const renderAgentOutputs = (agents?: AgentOutputs) => {
    if (!agents) return null;
    const entries = Object.entries(agents).filter(([, value]) => Boolean(value));
    if (entries.length === 0) return null;
    return (
      <dl className="grid grid-cols-2 gap-3 text-xs text-gray-600">
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt className="font-semibold text-gray-400 uppercase">{key}</dt>
            <dd className="text-gray-800 truncate">{value as string}</dd>
          </div>
        ))}
      </dl>
    );
  };

  return (
    <div className="space-y-8">
      <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm uppercase tracking-wide text-primary font-semibold mb-1">Interface</p>
          <h1 className="text-3xl font-bold text-gray-900">Tester l&apos;orchestrateur en direct</h1>
          <p className="text-gray-500 max-w-2xl">
            Sélectionnez un ou plusieurs scénarios clients, lancez l&apos;analyse via l&apos;API et
            consultez les sorties des agents A1–A5 ainsi que la décision finale.
          </p>
        </div>
        <button
          className="inline-flex items-center gap-2 rounded-xl border border-gray-200 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50"
          onClick={loadHistory}
        >
          <RefreshCw size={16} />
          Rafraîchir l&apos;historique
        </button>
      </header>

      <section className="bg-white border border-gray-100 rounded-3xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 gap-4">
          <h2 className="text-xl font-semibold text-gray-900">Bibliothèque d&apos;exemples</h2>
          <div className="text-sm text-gray-500">
            {selectedCount} scénario{selectedCount > 1 ? 's' : ''} prêts à être exécutés.
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {DEFAULT_EXAMPLES.map((example) => {
            const isSelected = selectedExamples.includes(example.id);
            return (
              <button
                key={example.id}
                onClick={() => handleToggleExample(example.id)}
                className={`text-left rounded-2xl border p-4 transition-all ${
                  isSelected ? 'border-primary bg-red-50 shadow-sm' : 'border-gray-100 hover:border-gray-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm uppercase tracking-wide text-gray-400">
                    <Star size={14} />
                    {example.category}
                  </div>
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isSelected ? 'bg-primary shadow-primary' : 'bg-gray-300'
                    }`}
                  />
                </div>
                <h3 className="mt-2 text-lg font-semibold text-gray-900">{example.title}</h3>
                <p className="text-sm text-gray-600 line-clamp-2">{example.text}</p>
              </button>
            );
          })}
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="md:col-span-2">
            <label className="text-sm font-medium text-gray-700">Message personnalisé</label>
            <textarea
              className="mt-2 w-full rounded-2xl border border-gray-200 bg-gray-50 p-3 text-sm focus:border-primary focus:ring-primary"
              rows={4}
              value={customText}
              onChange={(event) => setCustomText(event.target.value)}
              placeholder="Collez un message client ici pour le tester..."
            />
          </div>
          <div className="flex flex-col gap-4 rounded-2xl border border-dashed border-gray-200 p-4 text-sm text-gray-600">
            <p>
              ⏱️ Les tests utilisent le même pipeline que la production (prompts, OpenRouter,
              LangGraph) et renvoient toutes les sorties agents.
            </p>
            <p>🔐 Maximum {MAX_INTERFACE_SAMPLES} textes par requête pour préserver la latence.</p>
            <button
              disabled={isRunning}
              onClick={handleRun}
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-semibold text-white shadow-lg shadow-red-200 transition hover:bg-red-600 disabled:opacity-70"
            >
              <Play size={16} />
              Lancer les tests
            </button>
            {error && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle size={16} />
                {error}
              </div>
            )}
          </div>
        </div>
      </section>

      {results.length > 0 && (
        <section className="bg-white border border-gray-100 rounded-3xl p-6 shadow-sm">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 gap-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Résultats des agents</h2>
              <p className="text-gray-500 text-sm">Latence moyenne et classification finale.</p>
            </div>
            <div className="text-sm text-gray-500">{results.length} sortie(s)</div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {results.map((result) => (
              <div key={result.id ?? result.text} className="rounded-2xl border border-gray-100 p-4 shadow-sm bg-gray-50">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-gray-400">
                      {result.id ?? 'échantillon'}
                    </p>
                    <p className="font-semibold text-gray-900">{result.text.slice(0, 80)}...</p>
                  </div>
                  <ResultBadge label={`${result.latency_ms ?? 0} ms`} />
                </div>
                {result.error ? (
                  <p className="mt-3 text-sm text-red-600">{result.error}</p>
                ) : (
                  <>
                    {result.final && (
                      <pre className="mt-3 rounded-xl bg-white p-3 text-xs text-gray-700 border border-gray-100 overflow-x-auto">
                        {JSON.stringify(result.final, null, 2)}
                      </pre>
                    )}
                    {renderAgentOutputs(result.agents)}
                  </>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="bg-white border border-gray-100 rounded-3xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Historique récent</h2>
          <p className="text-sm text-gray-500">Derniers tickets du fichier CSV</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 uppercase text-xs tracking-wide">
                <th className="py-2">Sujet</th>
                <th className="py-2">Catégorie</th>
                <th className="py-2">Sentiment</th>
                <th className="py-2">Gravité</th>
                <th className="py-2">Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((ticket) => (
                <tr key={ticket.id} className="border-t border-gray-100">
                  <td className="py-3">
                    <p className="font-medium text-gray-900">{ticket.subject}</p>
                    <p className="text-gray-500 text-xs line-clamp-1">{ticket.description}</p>
                  </td>
                  <td className="py-3 text-gray-600">{ticket.categorie}</td>
                  <td className="py-3 text-gray-600">{ticket.sentiment}</td>
                  <td className="py-3">
                    <span
                      className={`text-xs font-semibold px-2 py-1 rounded-full ${
                        historySeverityClasses[ticket.severity] ?? 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {ticket.severity} ({ticket.gravity})
                    </span>
                  </td>
                  <td className="py-3 text-gray-500">{ticket.created_at}</td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-gray-500">
                    Aucun ticket trouvé. Lancez un run ou importez un CSV.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

