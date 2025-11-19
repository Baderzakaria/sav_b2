Parfait — voici la **mise à jour du rapport “FreeMind”** intégrant :



* l’**Agent Checker A2A** (validation logique multi-agents),

* l’**orchestration simultanée** des agents,

* le **respect strict du texte d’origine** (aucune modification du `full_text`),

* la **déduplication**, l’**exploration** et la **connexion des tweets liés** comme **contexte** pour l’étiquetage.



---



# Architecture technique de la solution FreeMind (version SLM, sans Docker)



## 1) Objectif & principes



* **But** : transformer un volume de tweets SAV en **labels structurés** (utile, catégorie, sentiment, type_problème, gravité), avec **cohérence inter-agents** garantie par un **Checker A2A**.

* **Agents LLM** : hébergés **chez toi**, appelés via **un provider local** (HTTP simple).

* **Orchestration** : **exécution simultanée** (parallèle) via **LangGraph**, puis **validation A2A**.

* **Intégrité des données** : **jamais d’édition du `full_text`**. Toute préparation se fait en **métadonnées** et **contexte** séparés.

* **Contexte conversationnel** : les tweets **reliés** (réponses, retweets, citations) sont **agrégés et passés au prompt**, sans toucher au `full_text` du tweet cible.



---



## 2) Pipeline de traitement



| Étape                                    | Module             | Entrées                 | Sorties                                    | Règles clés                                                                              |

| ---------------------------------------- | ------------------ | ----------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |

| 1. Chargement                            | Loader CSV         | `free tweet export.csv` | DataFrame brut                             | Schéma colonnes d’entrée ci-dessous                                                      |

| 2. Prétraitement (sans édition de texte) | Dedupe & Explore   | DF brut                 | DF filtré + métriques EDA                  | **Dédup** sur `id`; typage dates; stats globales                                         |

| 3. Builder de contexte                   | Thread/Relations   | DF prétraité            | Col. `ctx_before`, `ctx_after`, `ctx_refs` | **A partir de** `in_reply_to`, `retweeted_status`, `quoted_status`, `url`, `screen_name` |

| 4. Orchestration parallèle               | LangGraph          | (tweet, contexte)       | Sorties Agents 1→5                         | Appels simultanés                                                                        |

| 5. Validation A2A                        | Agent 6 (Checker)  | Résultats agents        | `checker_status`, corrections/minor flags  | Règles logiques + seuils                                                                 |

| 6. Écriture & UI                         | SQLite + Streamlit | Résultats consolidés    | Table `tweets_enriched` + dashboard        | Export CSV enrichi                                                                       |



### 2.1 Schéma de colonnes d’entrée (extrait du jeu fourni)



Obligatoires :

`id, created_at, full_text, screen_name, name, user_id, in_reply_to, retweeted_status, quoted_status, url`



Optionnelles utiles : `media, media_tags, favorite_count, retweet_count, bookmark_count, quote_count, reply_count, views_count, favorited, retweeted, bookmarked, profile_image_url`



> **Règle** : `full_text` est **source de vérité** et **n’est jamais modifié**.



---



## 3) Labels à produire (définition officielle)



### 3.1 Liste des labels (agents 1→5)



* **`utile`** *(bool)* : tweet pertinent pour le SAV (exclut pubs/spam/hors sujet/RT promos).

* **`categorie`** *(string ∈ {probleme, question, retour_client}).

* **`sentiment`** *(string ∈ {outrage_critique, tres_negatif, negatif, mecontent, legerement_negatif, neutre, legerement_positif, positif, tres_positif, mixte}).

* **`type_probleme`** *(string ∈ {panne, facturation, abonnement, resiliation, information, autre}).

* **`score_gravite`** *(int ∈ [-10, +10])* : sévérité perçue (ton + mots clés + signaux d’urgence).



### 3.2 Schéma JSON unique retourné (par tweet)



```json

{

  "utile": true,

  "categorie": "probleme",

  "sentiment": "negatif",

  "type_probleme": "panne",

  "score_gravite": 9

}

```



> **Validation Pydantic** côté orchestrateur avant stockage.



### 3.3 Taxonomie de sentiment (v0.2 — résumé)



- `outrage_critique` — polarité: négatif, base_gravity: 9, allowed_range: [7,10]

- `tres_negatif` — polarité: négatif, base_gravity: 7, allowed_range: [5,9]

- `negatif` — polarité: négatif, base_gravity: 5, allowed_range: [3,7]

- `mecontent` — polarité: négatif, base_gravity: 4, allowed_range: [2,6]

- `legerement_negatif` — polarité: négatif, base_gravity: 2, allowed_range: [1,4]

- `neutre` — polarité: neutre, base_gravity: 0, allowed_range: [-1,1]

- `legerement_positif` — polarité: positif, base_gravity: -2, allowed_range: [-4,-1]

- `positif` — polarité: positif, base_gravity: -4, allowed_range: [-6,-2]

- `tres_positif` — polarité: positif, base_gravity: -6, allowed_range: [-8,-4]

- `mixte` — polarité: mixte, base_gravity: 0, allowed_range: [-3,3]



---



## 4) Prétraitement (strict, non destructif)



* **Déduplication** : suppression des doublons par `id` (conserver la 1ʳᵉ occurrence).

* **Typage** : `created_at` → datetime (TZ Europe/Paris si utile pour EDA).

* **Aucune modification de `full_text`**.

* **Exploration rapide (EDA)** :



  * distribution temporelle (tweets/jour),

  * top `screen_name`, répartition `retweeted_status`/`quoted_status`,

  * part de tweets liés (`in_reply_to` non null),

  * stats d’engagement (médiane/quartiles : favorites/retweets/replies).

* **Construction du contexte** *(sans toucher au texte cible)* :



  * `ctx_before` : texte du **parent** (si `in_reply_to` ≠ null).

  * `ctx_after` : **réponses directes** si disponibles (même `in_reply_to` pointant sur l’ID cible).

  * `ctx_refs` : **contenu référencé** (RT, quote) via `retweeted_status`, `quoted_status`, ou `url` vers un tweet.

  * Limiter chaque champ à **N caractères** (par ex. 800) pour rester SLM-friendly ; **ne pas tronquer `full_text`**.



> **But** : donner **plus de contexte aux agents** sans altérer le tweet cible.



---



## 5) Orchestration simultanée (LangGraph)



### 5.1 Nœuds



* **Prompts centralisés**: `prompts/freemind_prompts.json` (`version: freemind_prompts_v0.2`, `global.system_header`, placeholders `{{full_text}}`, `{{ctx_before}}`, `{{ctx_after}}`, `{{ctx_refs}}`).

* `preprocess_node` → produit `ctx_before/ctx_after/ctx_refs` + métadonnées (mais **pas** de `text_norm`).

* `agents_parallel_node` → lance **en parallèle** :



  * **A1 Filtrage → `utile`**

  * **A2 Thématique → `categorie`**

  * **A3 Sentiment → `sentiment`**

  * **A4 Type de problème → `type_probleme`**

  * **A5 Gravité → `score_gravite`**
    * Dépend de `A1..A4`; utilise la taxonomie (base_gravity/allowed_range) pour calibrer le score.

* `checker_node` (Agent 6) → **A2A** : vérifie les combinaisons et applique **flags**/corrections légères si la règle est non ambiguë.

* `writer_node` → insertion SQLite + log `a2a_trace`.



### 5.2 État minimal partagé



```python

State = {

  "tweet": { "id": ..., "full_text": ..., "created_at": ..., "screen_name": ... },

  "context": { "ctx_before": "...", "ctx_after": "...", "ctx_refs": "..." },

  "results": { "A1": {...}, "A2": {...}, "A3": {...}, "A4": {...}, "A5": {...} },

  "checked": { "final": {...}, "checker_status": "ok|warn|fail", "a2a_trace": {...} },

  "meta": { "model": "...", "prompt_version": "...", "run_id": "..." }

}

```



---



## 6) Agent Checker (A2A) — règles & sortie



### 6.1 Règles de cohérence (exemples)



1. **Utile vs Catégorie**



* Si `utile=false` → forcer `categorie="retour_client"` et `type_probleme="autre"` (ou **vider** ces champs) ; `score_gravite` → 0 par défaut.

* Si `utile=true` & `categorie` manquant → **warn**.



2. **Catégorie vs Type de problème**



* Si `categorie="probleme"` & `type_probleme ∈ {information, autre}` → **warn** (incohérence probable).

* Si `categorie="question"` & `type_probleme="panne"` **et** `sentiment ∈ {legerement_positif, positif, tres_positif}` → **warn**.



3. **Sentiment vs Gravité** (taxonomie v0.2)



* Si `sentiment` a une polarité **positive** (`legerement_positif`, `positif`, `tres_positif`) ⇒ `score_gravite ≤ -1` et dans l’`allowed_range` du label (clamp si hors plage).

* Si `sentiment` a une polarité **négative** (`legerement_negatif`, `mecontent`, `negatif`, `tres_negatif`, `outrage_critique`) ⇒ `score_gravite ≥ 1` et dans l’`allowed_range` du label (clamp si hors plage).

* Si `sentiment ∈ {neutre, mixte}` ⇒ `score_gravite` proche de 0 (respect de l’`allowed_range`).

* Cas fort: `sentiment="outrage_critique"` & `score_gravite < 7` ⇒ **warn**; corriger `score_gravite` à `≥7` (dans l’`allowed_range`).



4. **Contexte lié (A4 dépend de contexte)**



* Si `ctx_before` mentionne **panne/incident/rupture fibre** et A4≠`panne` → **warn** (proposer correction).

* Si `retweeted_status` provient d’un **compte officiel incident** (ex. `Free_1337`) et `categorie!="probleme"` → **warn**.



5. **Règles durs (fail)**



* Sortie JSON invalide (clé manquante, domaine invalide) → **fail**.

* Conflits multiples (≥3 warns) → **fail**.



### 6.2 Stratégie de correction



* **Conservatrice** :



  * Appliquer **clamp** sur `score_gravite` dans les cas 3),

  * Corriger `type_probleme` vers `panne` si **contexte** explicite incident + `categorie="probleme"`.

* Sinon, ne pas modifier les labels, mais **étiqueter** `checker_status="warn"` et journaliser la proposition.



### 6.3 Sortie du Checker



```json

{

  "final": {

    "utile": true,

    "categorie": "probleme",

    "sentiment": "negatif",

    "type_probleme": "panne",

    "score_gravite": 9

  },

  "checker_status": "ok",

  "a2a_trace": {

    "inputs": {

      "A1": {"utile": true},

      "A2": {"categorie": "probleme"},

      "A3": {"sentiment": "negatif"},

      "A4": {"type_probleme": "panne"},

      "A5": {"score_gravite": 9}

    },

    "rules_fired": ["CTX_INCIDENT→TYPE=panne", "SENT_NEG→GRAV≥2"],

    "corrections": []

  }

}

```



---



## 7) Prompts (extraits) — respect du texte et du contexte



> **Contraintes communes (en-tête système pour tous les agents)**

>

> * **N’édite jamais** le texte : utilise **exactement** `full_text`.

> * Tu reçois un **contexte optionnel** (`ctx_before`, `ctx_after`, `ctx_refs`) pour comprendre le fil — **ne modifie pas** le texte cible.

> * **Réponds uniquement en JSON valide** conforme au schéma demandé.



### 7.1 A2 — Thématique



```

SYSTEM:

Tu es un classifieur JSON strict. Réponds uniquement en JSON.



USER:

Texte (ne pas modifier):

{{full_text}}



Contexte (si présent, seulement pour comprendre):

- Avant: {{ctx_before}}

- Après: {{ctx_after}}

- Références: {{ctx_refs}}



Tâche: "categorie" ∈ {"probleme","question","retour_client"}.

Retourne: {"categorie":"..."}

```



### 7.2 A4 — Type de problème



```

SYSTEM:

JSON strict. Ne modifie pas le texte. Utilise le contexte pour interpréter.



USER:

Texte: {{full_text}}

Contexte: avant={{ctx_before}} après={{ctx_after}} refs={{ctx_refs}}

Tâche: "type_probleme" ∈ {"panne","facturation","abonnement","resiliation","information","autre"}.

Retourne: {"type_probleme":"..."}

```



### 7.3 A3 — Sentiment (échelle 10)



```

SYSTEM:

JSON strict. Ne modifie pas le texte. Utilise le contexte pour interpréter.



USER:

Texte: {{full_text}}

Contexte: avant={{ctx_before}} après={{ctx_after}} refs={{ctx_refs}}

Tâche: Classer le "sentiment" selon l'échelle à 10 options (voir taxonomie):

{outrage_critique, tres_negatif, negatif, mecontent, legerement_negatif, neutre, legerement_positif, positif, tres_positif, mixte}.

Retourne: {"sentiment":"<label>"}

```



*(Prompts centralisés en JSON: `prompts/freemind_prompts.json` — version `freemind_prompts_v0.2`.)*



---



## 8) Modèle de données (SQLite)



### Table `tweets_enriched`



```sql

CREATE TABLE tweets_enriched(

  id TEXT PRIMARY KEY,

  created_at TEXT,

  full_text TEXT,              -- intact, source de vérité

  screen_name TEXT,

  name TEXT,

  user_id TEXT,

  in_reply_to TEXT,

  retweeted_status TEXT,

  quoted_status TEXT,

  url TEXT,



  -- Contexte (non destructif)

  ctx_before TEXT,

  ctx_after TEXT,

  ctx_refs TEXT,



  -- Labels finaux (checker)

  utile BOOLEAN,

  categorie TEXT,

  sentiment TEXT,

  type_probleme TEXT,

  score_gravite INTEGER,



  -- Traçabilité

  checker_status TEXT,         -- ok|warn|fail

  a2a_trace TEXT,              -- JSON

  llm_model TEXT,

  prompt_version TEXT,

  run_id TEXT

);

CREATE INDEX idx_labels ON tweets_enriched(categorie, sentiment, type_probleme);

```



---



## 9) Interface & export



* **Streamlit** :



  * Upload du CSV → exécution pipeline → tableau filtrable.

  * Filtres : période, `screen_name`, `categorie`, `type_probleme`, `sentiment`, gravité, `checker_status`.

  * **KPI** : % utile, top types de problème, moyenne gravité, nb `warn/fail`.

  * **Export** CSV enrichi (conserve **toutes** les colonnes originales + colonnes ajoutées).



---



## 10) MLOps léger & traçabilité



* **Git** : version du code.

* **Prompts** : JSON `version:` dans `prompts/freemind_prompts.json` (ex. `freemind_prompts_v0.2`).

* **Run ID** : `YYYYMMDD-HHMM-<gitsha7>`.

* **MLflow (optionnel)** : params (modèle/versions prompt), métriques (latence/coverage), artefacts (CSV enrichi).

* **Logs A2A** : règle(s) déclenchée(s), corrections appliquées, décisions non-appliquées.



---



## 11) Notes spécifiques au dataset fourni



* Les **tweets “RT @free …”** présents dans tes exemples deviennent **contexte** si le tweet cible est une réponse/quote/RT ; sinon ils sont **labellisés** comme tout autre tweet (A1 peut marquer `utile=false` pour autopromo pure).

* Les **comptes officiels incident** (ex. `Free_1337`) sont des **indicateurs forts de “panne”** pour le Checker A2A quand ils apparaissent en contexte (`ctx_refs` ou `ctx_before`).



---



### TL;DR (exécution)



1. Charger CSV → **dédupliquer** par `id`.

2. Construire `ctx_before/ctx_after/ctx_refs` **sans modifier `full_text`**.

3. Lancer **A1..A5 en parallèle** → réunir sorties.

4. **Agent 6 Checker** applique règles A2A → *final + checker_status + a2a_trace*.

5. Écrire en **SQLite** et **afficher** dans **Streamlit** → **export**.



Si tu veux, je peux te donner **le JSON Schema Pydantic** des sorties + **le pseudo-code LangGraph** des nœuds `agents_parallel_node` et `checker_node` (directement prêt à coller dans le repo).



