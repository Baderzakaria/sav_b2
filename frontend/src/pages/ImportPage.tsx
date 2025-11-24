import { useState } from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle2, Loader2, AlertCircle, ArrowRight, LayoutDashboard } from 'lucide-react';
import { clsx } from 'clsx';
import Papa from 'papaparse';
import { Link } from 'react-router-dom';
import { useData, Ticket } from '../context/DataContext';

type StepStatus = 'pending' | 'processing' | 'completed' | 'error';

interface ProcessingStep {
  id: number;
  label: string;
  description: string;
  status: StepStatus;
}

interface CsvRow {
  timestamp: string;
  row_index: string;
  tweet_id: string;
  screen_name: string;
  full_text: string;
  date_iso: string; // The real date
  clean_text: string;
  favorite_count: string;
  reply_count: string;
  elapsed_sec: string;
  A1_utile: string;
  A2_categorie: string;
  A3_sentiment: string;
  A4_type: string;
  A5_gravity: string; // The AI gravity score
  Final_utile: string;
  Final_categorie: string; // The AI category
  Final_sentiment: string; // The AI sentiment
  Final_gravity: string; // The AI gravity score
}

export const ImportPage = () => {
  const { addTickets } = useData();
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [importedCount, setImportedCount] = useState(0);
  const [steps, setSteps] = useState<ProcessingStep[]>([
    { id: 1, label: 'Lecture du fichier', description: 'Analyse du format CSV', status: 'pending' },
    { id: 2, label: 'Validation Structure', description: 'Vérification des colonnes', status: 'pending' },
    { id: 3, label: 'Importation', description: 'Mise à jour du Dashboard', status: 'pending' },
  ]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const determineSeverity = (ai_gravity: string): 'faible' | 'moyenne' | 'élevée' | 'critique' => {
    const score = parseInt(ai_gravity);
    if (isNaN(score)) return 'faible';
    if (score < -5) return 'critique'; // Negative gravity in your dataset seems to mean bad/urgent
    if (score < 0) return 'élevée';
    if (score > 3) return 'faible'; // Positive seems good
    return 'moyenne';
  };

  const updateStep = (index: number, status: StepStatus) => {
    setSteps(prev => prev.map((s, idx) => idx === index ? { ...s, status } : s));
  };

  const startProcessing = async () => {
    if (!file) return;
    setIsProcessing(true);
    setImportedCount(0);
    
    // Step 1: Read File
    updateStep(0, 'processing');
    
    Papa.parse<CsvRow>(file, {
      header: true,
      skipEmptyLines: true,
      complete: async (results) => {
        updateStep(0, 'completed');
        
        // Step 2: Validate
        updateStep(1, 'processing');
        await new Promise(r => setTimeout(r, 800)); // UI delay for feel
        
        // Check for columns present in freemind_log_latest.csv
        if (!results.meta.fields?.includes('full_text') && !results.meta.fields?.includes('date_iso')) {
          updateStep(1, 'error');
          setIsProcessing(false);
          alert("Colonnes manquantes: Le fichier doit avoir 'full_text' et 'date_iso'");
          return;
        }
        updateStep(1, 'completed');

        // Step 3: Import direct
        updateStep(2, 'processing');
        const parsedTickets: Ticket[] = results.data.map((row, index) => {
          // Use AI columns if available, fallback to basic logic
          const ai_gravity = row.Final_gravity || row.A5_gravity || '0';
          const severity = determineSeverity(ai_gravity);
          
          // Map sentiment/category/type from AI columns
          const category = row.Final_categorie || row.A2_categorie || 'autre';
          const sentiment = row.Final_sentiment || row.A3_sentiment || 'neutre';
          const type = row.A4_type || 'autre';
          const utile = row.Final_utile === 'True' || row.A1_utile === 'True';
          const gravityScore = parseInt(ai_gravity) || 0;
          
          return {
            id: row.tweet_id || `csv-${index}-${Date.now()}`,
            source: 'twitter',
            subject: category.toUpperCase() + ': ' + (row.full_text || '').substring(0, 40) + '...',
            description: row.full_text || 'Pas de contenu',
            clean_text: row.clean_text,
            customer: row.screen_name ? `@${row.screen_name}` : 'Anonyme',
            created_at: row.date_iso || new Date().toISOString(),
            status: 'nouveau',
            severity: severity,
            channel: 'Twitter',
            // AI Analysis Fields
            type: type,
            sentiment: sentiment,
            categorie: category,
            gravity: gravityScore,
            favorite_count: parseInt(row.favorite_count) || 0,
            reply_count: parseInt(row.reply_count) || 0,
            utile: utile,
            agentResponses: {
              utile: row.A1_utile,
              categorie: row.A2_categorie,
              sentiment: row.A3_sentiment,
              type: row.A4_type,
              gravity: row.A5_gravity,
            }
          };
        });
        addTickets(parsedTickets);
        setImportedCount(parsedTickets.length);
        updateStep(2, 'completed');
        
        setIsProcessing(false);
      },
      error: (error) => {
        console.error(error);
        updateStep(0, 'error');
        setIsProcessing(false);
      }
    });
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h2 className="text-2xl font-bold text-gray-800">Import de Données</h2>
        <p className="text-gray-500">Chargez vos fichiers CSV (tweets, logs) pour alimenter le Dashboard</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload Area */}
        <div className="lg:col-span-2 space-y-6">
          <div 
            className={clsx(
              "border-2 border-dashed rounded-2xl p-10 flex flex-col items-center justify-center text-center transition-all duration-200 bg-white",
              dragActive ? "border-primary bg-red-50" : "border-gray-300 hover:border-primary/50",
              file ? "border-green-500 bg-green-50" : ""
            )}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="flex flex-col items-center animate-in zoom-in duration-300">
                <div className="h-16 w-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center mb-4">
                  <FileSpreadsheet size={32} />
                </div>
                <h3 className="text-lg font-bold text-gray-800">{file.name}</h3>
                <p className="text-sm text-gray-500 mb-6">{(file.size / 1024).toFixed(2)} KB</p>
                <div className="flex gap-3">
                  <button 
                    onClick={() => setFile(null)}
                    className="px-4 py-2 text-sm text-gray-500 hover:text-red-500 transition-colors"
                  >
                    Changer de fichier
                  </button>
                  {!isProcessing && steps[0].status !== 'completed' && (
                    <button 
                      onClick={startProcessing}
                      className="px-6 py-2 bg-primary text-white rounded-xl shadow-soft hover:shadow-lg transition-all active:scale-95 flex items-center gap-2"
                    >
                      <ArrowRight size={18} />
                      Lancer le traitement
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <>
                <div className="h-16 w-16 bg-gray-50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <UploadCloud size={32} className="text-gray-400" />
                </div>
                <h3 className="text-lg font-bold text-gray-800 mb-2">Glissez votre fichier CSV ici</h3>
                <p className="text-sm text-gray-500 mb-6 max-w-sm">
                  Format supporté : CSV (Export Tweets).
                </p>
                <label className="px-6 py-2 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-50 transition-colors font-medium text-gray-700">
                  Parcourir les fichiers
                  <input type="file" className="hidden" accept=".csv" onChange={handleChange} />
                </label>
              </>
            )}
          </div>

          {/* Quick Help / Info */}
          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 flex gap-4 items-start">
            <AlertCircle className="text-blue-500 shrink-0 mt-1" size={24} />
            <div>
              <h4 className="font-bold text-blue-900">Format attendu</h4>
              <p className="text-sm text-blue-700 mt-1">
                Le fichier doit contenir les colonnes : <code>created_at</code>, <code>full_text</code>, <code>screen_name</code>. 
                L'IA calculera automatiquement la "Gravité" (Severity) basée sur les mots-clés et l'engagement.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Processing Status */}
        <div className="bg-white rounded-2xl shadow-soft p-6 h-fit">
          <h3 className="text-lg font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Loader2 className={clsx("text-primary", isProcessing && "animate-spin")} size={20} />
            État du traitement
          </h3>
          
          <div className="space-y-6 relative before:absolute before:left-4 before:top-2 before:h-[85%] before:w-0.5 before:bg-gray-100">
            {steps.map((step) => (
              <div key={step.id} className="relative flex gap-4">
                <div className={clsx(
                  "h-8 w-8 rounded-full flex items-center justify-center shrink-0 z-10 transition-colors duration-300 border-2",
                  step.status === 'completed' ? "bg-green-500 border-green-500 text-white" :
                  step.status === 'processing' ? "bg-white border-primary text-primary" :
                  step.status === 'error' ? "bg-red-100 border-red-500 text-red-500" :
                  "bg-white border-gray-200 text-gray-300"
                )}>
                  {step.status === 'completed' ? <CheckCircle2 size={18} /> : 
                   step.status === 'processing' ? <Loader2 size={18} className="animate-spin" /> : 
                   step.status === 'error' ? <AlertCircle size={18} /> :
                   <span className="text-xs font-bold">{step.id}</span>}
                </div>
                
                <div className="pt-1">
                  <h4 className={clsx(
                    "text-sm font-bold transition-colors",
                    step.status === 'completed' ? "text-green-600" :
                    step.status === 'processing' ? "text-primary" :
                    step.status === 'error' ? "text-red-600" :
                    "text-gray-500"
                  )}>
                    {step.label}
                  </h4>
                  <p className="text-xs text-gray-400 mt-1">{step.description}</p>
                </div>
              </div>
            ))}
          </div>

          {steps[steps.length - 1].status === 'completed' && (
             <div className="mt-8 p-4 bg-green-50 text-green-700 rounded-xl text-center text-sm font-medium animate-in slide-in-from-bottom-2">
               Import réussi !
               <br/>
               <span className="text-xs opacity-80">{importedCount} tickets ajoutés au Dashboard.</span>
               
               <div className="mt-4 flex justify-center">
                 <Link 
                   to="/" 
                   className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors shadow-sm"
                 >
                   <LayoutDashboard size={18} />
                   Voir le Dashboard
                 </Link>
               </div>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};
