# Sistema di Stima Continua del Consenso Elettorale Italiano (senza sondaggi)

**Documento tecnico-concettuale — v1.0**
Team: Data Scientist (bayesiano) · Statistico elettorale · Data Engineer · Sviluppatore Python senior · Esperto ETL/MongoDB

---

## 0. Sintesi esecutiva e tesi centrale

L'obiettivo non è prevedere una singola elezione, ma stimare in continuo lo **stato latente del consenso nazionale** dei partiti, trattando ogni elezione reale come una **misura rumorosa e distorta** di quello stato.

> Idea chiave: il consenso "vero" `θ_t` non è osservabile. Ogni elezione (politica, europea, regionale, comunale, referendum) è una **osservazione** che vede `θ_t` attraverso una lente deformante (tipo di elezione, affluenza differenziale, copertura territoriale, forza locale del partito). Il modello stima `θ_t` separando il segnale strutturale dalle distorsioni della specifica elezione.

**Approccio raccomandato: State Space Model bayesiano gerarchico** (Dynamic Linear Model su uno stato latente nel simplesso), con:
- **equazione di stato** = random walk dell'evoluzione del consenso nel tempo;
- **equazione di osservazione** = mappa elezione→stato con termini di bias per tipo-elezione e correzione affluenza;
- **gerarchia territoriale** sezione → comune → provincia → regione → nazione (partial pooling);
- **modulo separato di flussi elettorali** via ecological inference bayesiana.

Gli altri approcci (HMM, Kalman puro, updating bayesiano semplice) sono **casi particolari o componenti** di questo schema, non alternative concorrenti. Sotto spiego perché.

---

## 1. Architettura generale

### 1.1 Vista a livelli

```
┌──────────────────────────────────────────────────────────────┐
│  FONTI (Eligendo, ISTAT, Min. Interno, opz. sondaggi)          │
└───────────────┬──────────────────────────────────────────────┘
                │  scraping / download / API / CSV
┌───────────────▼──────────────────────────────────────────────┐
│  ETL LAYER                                                     │
│   - ingestion (raw, immutabile)                                │
│   - normalizzazione (partiti, geografie, codici ISTAT)         │
│   - reconciliation entità partito (alias/fusioni/scissioni)    │
│   - feature build (affluenza, demografia per area)             │
└───────────────┬──────────────────────────────────────────────┘
                │  MongoDB (staging → curated)
┌───────────────▼──────────────────────────────────────────────┐
│  MODELING LAYER                                                │
│   - State Space gerarchico (stima consenso θ_t)                │
│   - Modulo flussi elettorali (ecological inference)            │
│   - Modulo correzione tipo-elezione / affluenza                │
└───────────────┬──────────────────────────────────────────────┘
                │  posterior samples / summaries
┌───────────────▼──────────────────────────────────────────────┐
│  ESTIMATION STORE (versionato, immutabile per run)             │
│   estimations, flow_models, model_runs                         │
└───────────────┬──────────────────────────────────────────────┘
                │  REST API (Flask)
┌───────────────▼──────────────────────────────────────────────┐
│  SERVING / OUTPUT                                              │
│   nazionale, regionale, trend, mappa, P(>soglia), distribuz.   │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Principi architetturali

1. **Separazione netta tra dato grezzo e dato modellato.** Il raw non si modifica mai (audit, riproducibilità).
2. **Event-sourcing sulle elezioni.** Ogni elezione è un evento immutabile; le stime sono *derivate* e versionate.
3. **Idempotenza dell'ETL.** Re-ingestione della stessa fonte non duplica.
4. **Stime versionate e firmate.** Ogni run produce una stima con `model_run_id`, hash del codice, hash dei dati di input, iperparametri.
5. **Il modello è offline-batch** (ri-stima a ogni nuovo evento elettorale o su schedule), non real-time. Il "consenso" cambia su scala di mesi, non di secondi.

---

## 2. Modello matematico

### 2.1 Spazio dello stato: il simplesso

Con `K` partiti, la quota di consenso è un vettore sul simplesso `Δ^{K-1}` (componenti ≥0, somma 1). Lavorare direttamente sulle percentuali viola i vincoli. Si usa la trasformazione **log-ratio additiva (ALR)** rispetto a un partito di riferimento `r` (es. il più stabile, o "Altri"):

```
η_k,t = log( p_k,t / p_r,t ),   k ≠ r
```

Lo stato latente è `η_t ∈ R^{K-1}`. La trasformazione inversa (softmax) riporta sul simplesso:

```
p_k,t = exp(η_k,t) / ( 1 + Σ_j exp(η_j,t) )      p_r,t = 1 / (1 + Σ_j exp(η_j,t))
```

Questo garantisce automaticamente che le stime sommino a 1 e restino in (0,1), e rende lineare/gaussiana l'equazione di stato.

### 2.2 Equazione di stato (evoluzione temporale)

Il consenso evolve come **random walk** (eventualmente con drift locale), perché tra un'elezione e l'altra non sappiamo *come* si muove ma sappiamo che si muove gradualmente:

```
η_t = η_{t-1} + ω_t ,    ω_t ~ N(0, Q)
```

- `Q` = matrice di covarianza dell'innovazione. Controlla **quanto velocemente** il consenso può cambiare. Va stimata: troppo grande → stime nervose; troppo piccola → modello rigido.
- Il tempo `t` è in **scala continua** (mesi). L'innovazione scala con il gap temporale: `ω_t ~ N(0, Q · Δt)`. Così due elezioni vicine vincolano molto, due lontane lasciano più libertà.
- Variante consigliata: **local linear trend** (random walk + componente di trend lentamente variabile) per catturare derive strutturali (es. crescita pluriennale di un partito).

### 2.3 Equazione di osservazione (l'elezione come misura distorta)

Questo è il cuore concettuale. Una elezione `e` di tipo `τ(e)` (politiche/europee/regionali/comunali/referendum) osservata al tempo `t(e)` produce, per il partito `k`, una quota osservata `y_{k,e}` (log-ratio della percentuale realmente presa). Il modello la lega allo stato latente nazionale:

```
ỹ_{k,e} = η_{k,t(e)}  +  β_{k,τ(e)}  +  γ_{k,e}(affluenza, copertura)  +  ε_{k,e}
```

dove:

- **`β_{k,τ}` = bias strutturale tipo-elezione × partito.** È il termine che codifica "le europee non sono le politiche". Esempio: un partito europeista prende sistematicamente di più alle europee → `β` positivo per quel partito a quel tipo. Sono parametri **persistenti** stimati su tutta la storia: catturano regolarità ricorrenti.
- **`γ_{k,e}`** = effetto **affluenza differenziale** e **copertura territoriale**. Le comunali coprono solo alcuni comuni in un dato anno; le regionali una regione. Quindi una singola elezione non misura l'Italia intera: misura un sottoinsieme. `γ` corregge la composizione del campione (vedi §2.5 sul peso e §2.6 sull'affluenza).
- **`ε_{k,e}` ~ N(0, R_{e})`** = rumore residuo di misura. **R non è il margine di errore di un sondaggio**: il dato elettorale aggregato è quasi esatto. `R` rappresenta lo scarto irriducibile tra "ciò che questa specifica elezione misura" e "il consenso nazionale strutturale". Si modella inversamente proporzionale alla rappresentatività dell'elezione: una politica nazionale ha `R` piccolo (misura quasi tutto), una comunale in pochi comuni ha `R` grande.

> Conseguenza pratica fondamentale: il filtro **pesa automaticamente** le elezioni. Le politiche spostano molto il posterior, una tornata di comunali in 200 comuni lo sposta poco. Esattamente il comportamento voluto.

### 2.4 Identificabilità dei bias

`η` e `β` insieme sono non identificabili senza vincolo (puoi spostare costanti tra loro). Si fissa l'ancora: **le elezioni politiche nazionali hanno `β_{k,politiche} = 0`** per definizione. Le politiche *definiscono* la scala del consenso nazionale; tutti gli altri tipi sono misurati come *scostamento* dalle politiche. Questo è statisticamente pulito e politicamente sensato (l'oggetto "consenso nazionale" è ciò che conta alle politiche).

### 2.5 Gerarchia territoriale (partial pooling)

Lo stato non è solo nazionale: si stima a più livelli con **shrinkage** verso il livello superiore. Per il partito `k`, regione `g`, al tempo `t`:

```
η^{naz}_{k,t}                                    (stato nazionale)
η^{reg}_{k,g,t} = η^{naz}_{k,t} + δ_{k,g} + u_{k,g,t}     (offset regionale)
η^{prov}_{k,h,t} = η^{reg}_{k,g(h),t} + δ_{k,h} + ...
η^{com}_{k,m,t}  = η^{prov}_{k,h(m),t} + δ_{k,m} + ...
```

- `δ_{k,g}` = effetto regionale **persistente** del partito (es. SVP solo in Trentino-AA, forza di un partito al Sud). Con prior gerarchico: `δ_{k,g} ~ N(0, σ²_k)`. I partiti con elettorato omogeneo hanno `σ²_k` piccolo; quelli territorialmente concentrati grande.
- **Partial pooling = il punto di forza.** Le comunali e regionali, abbondanti e frequenti, informano gli offset locali; questi a loro volta, tramite la struttura gerarchica, **vincolano e aggiornano la stima nazionale** anche in assenza di elezioni politiche. È così che "i livelli locali propagano informazione fino al nazionale".
- **Aggregazione bottom-up coerente.** La stima nazionale implicata dai livelli locali è la media degli `η` locali **pesata per gli aventi diritto / popolazione** (da ISTAT e Min. Interno), non la media semplice. Questo riconcilia top-down e bottom-up.

### 2.6 Trattamento dell'affluenza

L'affluenza non è solo rumore: è informativa e distorcente.

- Modello l'affluenza per area `a` come variabile osservata `T_a` (da Min. Interno).
- Effetto sul risultato: partiti diversi hanno **elasticità all'affluenza** diverse (`λ_k`). Quando crolla l'affluenza, alcuni elettorati restano a casa più di altri.
- Si entra in `γ_{k,e}` un termine `λ_k · (T_e − T̄_τ)`: scostamento dell'affluenza dell'elezione rispetto alla media storica del suo tipo. Così il modello "sa" che bassa affluenza gonfia/sgonfia certi partiti e **corregge** verso il consenso strutturale.

### 2.7 Perché lo State Space Model e non gli altri

| Approccio | Ruolo nel sistema | Verdetto |
|---|---|---|
| **State Space Model (DLM) gerarchico bayesiano** | Schema completo: stato latente + osservazioni distorte + tempo + gerarchia | **Scelta principale.** Tutto il resto è un sottoinsieme |
| **Bayesian Updating** | È *il meccanismo* con cui il filtro aggiorna il posterior a ogni elezione | Incluso per costruzione (il filtraggio è updating sequenziale) |
| **Dynamic Bayesian Model** | Sinonimo del DLM dinamico qui adottato | Coincide con la scelta |
| **Kalman Filter** | Algoritmo di inferenza esatto se tutto fosse lineare-gaussiano | Utile come **proposal/inizializzazione**; sul simplesso e con bias gerarchici la non-linearità (softmax, varianze ignote) richiede MCMC/HMC o Particle/Ensemble Kalman |
| **Hidden Markov Model** | Stato latente **discreto** (es. "regime di crescita / stabilità / declino") | **Utile come layer aggiuntivo**: un HMM di regime sopra il random walk cattura cambi di fase (scandalo, leadership change). Opzionale |
| **Modelli alternativi (GP temporali, Dirichlet dynamic)** | Gaussian Process sul tempo invece del random walk = prior più liscio; Dynamic Dirichlet = stato direttamente sul simplesso | Considerati come varianti di `Q`/della dinamica; non cambiano l'impianto |

**Motivazione della scelta.** Il problema ha quattro requisiti simultanei: (1) stato continuo evolutivo nel tempo, (2) osservazioni eterogenee e distorte, (3) struttura gerarchica territoriale, (4) quantificazione piena dell'incertezza (distribuzioni, non punti). Solo lo **State Space bayesiano gerarchico** li soddisfa tutti nello stesso framework coerente. Kalman puro fallisce su non-linearità e varianze ignote; updating semplice ignora il tempo e la gerarchia; HMM da solo è troppo grezzo (stato discreto). Inferenza: **HMC/NUTS** (Stan/PyMC/NumPyro) per la stima completa; **filtro particellare/Ensemble Kalman** per l'aggiornamento incrementale rapido quando arriva una nuova elezione senza rifare tutto.

### 2.8 Output probabilistico

L'inferenza produce **campioni dal posterior** `{η_t^{(s)}}` (S campioni). Da questi, per ogni partito, via softmax si ottengono campioni di `p_{k,t}`. Quindi:

- **Media stimata** = media dei campioni.
- **Intervallo credibile 95%** = quantili 2.5–97.5 dei campioni.
- **Distribuzione di probabilità** = istogramma/KDE dei campioni (la "forma" della stima).
- **P(p_k > soglia)** = frazione di campioni sopra soglia. Es. `P(>10%) = #{s: p^{(s)} > 0.10}/S`.
- **P(crescita)** = `P(p_{k,t} > p_{k,t-Δ})`.

Esempio di output coerente con la richiesta:

```
Partito X — stima al 2026-06
Media: 9.8%   IC95%: 8.9%–10.7%
P(>10%) = 38%   P(>12%) = 6%   P(in crescita vs 6 mesi fa) = 71%
```

---

## 3. Struttura database (MongoDB)

Convenzione: collection **raw** (immutabili), **curated** (normalizzate), **derived** (output modello). Tutti i documenti hanno `_meta` con `source`, `ingested_at`, `source_hash`.

### 3.1 Anagrafiche e geografia

**`geographies`** — albero territoriale unificato (chiave: codice ISTAT).
```json
{
  "_id": "ISTAT:058091",
  "level": "comune",            // sezione|comune|provincia|regione|nazione
  "name": "Roma",
  "parent": "ISTAT:RM",         // provincia
  "region": "ISTAT:12",
  "istat_code": "058091",
  "valid_from": "2001-01-01", "valid_to": null,   // confini cambiano nel tempo
  "centroid": {"type":"Point","coordinates":[12.49,41.90]}
}
```
> Nota: i comuni nascono/muoiono/si fondono. `valid_from/valid_to` + tabella di mapping storico (`geo_remap`) sono indispensabili per confrontare elezioni a distanza di anni.

**`parties`** — entità partito **canonica e stabile nel tempo** (vedi §6).
```json
{
  "_id": "party:M5S",
  "canonical_name": "Movimento 5 Stelle",
  "family": "populista",        // tassonomia opzionale
  "active_from": "2009", "active_to": null
}
```

**`party_aliases`** — ogni etichetta di lista come appare nei verbali, mappata a una entità canonica con validità temporale e tipo di relazione.
```json
{
  "_id": "...",
  "raw_label": "MOVIMENTO 5 STELLE",
  "party_id": "party:M5S",
  "relation": "alias",          // alias|merge|split|coalition_member|civic
  "weight": 1.0,                // per scissioni/fusioni: quota attribuita
  "valid_from": "2018", "valid_to": null,
  "source": "eligendo"
}
```

**`coalitions`** — composizione coalizioni per elezione (uninominali, liste collegate).
```json
{ "_id":"...", "election_id":"...", "coalition_name":"Centrodestra",
  "members":[{"party_id":"party:FDI","list_label":"..."}, ...] }
```

### 3.2 Elezioni e risultati (raw/curated)

**`elections`** — un documento per evento elettorale.
```json
{
  "_id": "elez:2022_politiche_camera",
  "type": "politiche",          // politiche|europee|regionali|comunali|referendum
  "chamber": "camera",          // camera|senato|na
  "date": "2022-09-25",
  "scope": {"level":"nazionale","geo_ids":["ISTAT:IT"]},
  "electoral_system": "rosatellum",
  "n_sezioni": 61566
}
```

**`party_results`** — risultato per (elezione × partito × area), al livello più fine disponibile (idealmente **sezione**).
```json
{
  "_id":"...",
  "election_id":"elez:2022_politiche_camera",
  "geo_id":"ISTAT:058091",
  "geo_level":"comune",
  "party_id":"party:M5S",        // già riconciliato
  "raw_label":"MOVIMENTO 5 STELLE",
  "votes": 45213,
  "valid_votes_area": 412000,    // denominatore per la quota
  "share": 0.1097
}
```
Indici: `{election_id, geo_id}`, `{party_id, geo_id, date}`.

**`turnout`** — affluenza per (elezione × area), più aventi diritto.
```json
{ "_id":"...","election_id":"...","geo_id":"...","geo_level":"comune",
  "eligible": 2300000, "voters": 1380000, "turnout": 0.60,
  "blank": 12000, "invalid": 9000 }
```

### 3.3 Dati di contesto (ISTAT)

**`demographics`** — profilo socio-demografico per area, con anno (sono serie storiche).
```json
{ "_id":"...","geo_id":"ISTAT:058091","year":2024,
  "population": 2750000, "median_age": 45.2,
  "age_bands": {"18-34":0.21,"35-54":0.31,"55+":0.48},
  "income_avg": 24500, "employment_rate": 0.61,
  "education": {"laurea":0.18,"diploma":0.42,"obbligo":0.40},
  "urbanization": "urbano", "density": 2200 }
```
Uso: covariate per i prior gerarchici (`δ`), per la stima dell'elasticità all'affluenza, e per imputare aree non votanti in una data tornata.

### 3.4 Storage del modello (derived, versionato)

**`model_runs`** — manifest immutabile di ogni esecuzione.
```json
{ "_id":"run:2026-06-19T...","model_version":"ssm-1.2.0",
  "code_hash":"git:abc123","input_data_hash":"sha256:...",
  "hyperparams":{"Q_scale":0.04,"ref_party":"party:ALTRI"},
  "elections_used":["elez:2022_politiche_camera", ...],
  "inference":{"method":"NUTS","draws":4000,"rhat_max":1.01},
  "created_at":"2026-06-19T10:00:00Z","status":"completed" }
```

**`estimations`** — stima per (run × partito × area × tempo). **Non si sovrascrive mai**; nuove stime = nuovo `run_id`.
```json
{ "_id":"...","run_id":"run:2026-06-19T...",
  "party_id":"party:M5S","geo_id":"ISTAT:IT","geo_level":"nazionale",
  "as_of":"2026-06-01",
  "mean":0.098,"median":0.097,"sd":0.0046,
  "ci80":[0.092,0.104],"ci95":[0.089,0.107],
  "quantiles":{"0.05":0.090,"0.25":0.095,"0.5":0.097,"0.75":0.101,"0.95":0.106},
  "prob_thresholds":{">0.10":0.38,">0.12":0.06,">0.15":0.003},
  "prob_growth_6m":0.71,
  "posterior_samples_ref":"gridfs://samples/run.../M5S_IT.npz" }
```
> I campioni grezzi del posterior (pesanti) vanno in **GridFS** o su object storage, referenziati; in `estimations` stanno solo i sommari serviti dall'API.

**`flow_models`** — matrici di transizione stimate tra due elezioni (vedi §5).
```json
{ "_id":"...","run_id":"...","from_election":"elez:2022_...","to_election":"elez:2024_europee",
  "geo_scope":"ISTAT:IT","level":"nazionale",
  "parties_from":["party:PD","party:M5S","party:FDI","astensione"],
  "parties_to":["party:PD","party:M5S","party:FDI","astensione"],
  "transfer_matrix_mean":[[...]],     // righe = da, colonne = verso, righe sommano a 1
  "transfer_matrix_sd":[[...]],
  "loyalty":{"party:PD":0.74,"party:M5S":0.61},
  "method":"bayesian_ecological_inference" }
```

**`backtests`** — risultati di validazione (vedi §7).
```json
{ "_id":"...","scheme":"train_until_2018","target":"elez:2022_politiche_camera",
  "metrics":{"mae":0.012,"crps":0.009,"coverage95":0.94},
  "per_party":[{"party_id":"party:M5S","pred_mean":0.155,"actual":0.155,"in_ci95":true}] }
```

**`audit_log`** — chi/cosa/quando su ETL e run (governance).

### 3.5 Indici chiave
- `party_results`: `(election_id, geo_id)`, `(party_id, geo_level, geo_id)`.
- `estimations`: `(run_id, geo_level, as_of)`, `(party_id, geo_id, as_of)`.
- `geographies`: `(level, parent)`, geo-index `2dsphere` su `centroid` per le mappe.
- `party_aliases`: `(raw_label, valid_from)`.

---

## 4. Pipeline ETL

### 4.1 Stadi

1. **Ingestion (raw).** Scraper/loader per Eligendo (risultati, spesso CSV/JSON per livello), Min. Interno (affluenza, sezioni), ISTAT (API/CSV demografia). Salva il payload **grezzo** + `source_hash`. Idempotente: se l'hash esiste già, skip.
2. **Validation.** Schema check (pydantic/cerberus), quadrature: la somma dei voti di lista in un'area = voti validi area; affluenza ∈ [0,1]; somme sezione→comune→provincia coerenti. Anomalie → quarantena, non scartate.
3. **Geo-normalization.** Mappa ogni codice/nome area sul nodo `geographies` valido alla data dell'elezione (gestione fusioni comuni via `geo_remap`).
4. **Party reconciliation.** Mappa `raw_label` → `party_id` via `party_aliases`; etichette nuove → coda di **revisione manuale** (vedi §6). Nessun match silenzioso e arbitrario.
5. **Feature build.** Calcola quote, affluenza differenziale, join con demografia ISTAT più recente ≤ data elezione, denominatori coerenti.
6. **Curated load.** Scrive `party_results`, `turnout` normalizzati. Transazionale per elezione.
7. **Trigger modeling.** Inserisce un messaggio in coda "nuova elezione disponibile" → scatena il run del modello.

### 4.2 Caratteristiche
- **Append-only sul raw**, idempotenza via hash.
- **Lineage**: ogni documento curated referenzia il raw da cui deriva.
- **Backfill** rieseguibile: una correzione su `party_aliases` deve poter **ri-processare** lo storico senza toccare il raw.
- **Quarantena** per dati sospetti con alert.

### 4.3 Scheduler e orchestrazione
- **APScheduler** (semplice) o **Prefect/Airflow** (robusto, raccomandato in produzione) per: polling periodico delle fonti, ri-ingestione affluenza in giornata elettorale, run notturni di ri-stima.
- Coda (Redis/RabbitMQ + Celery) per disaccoppiare ingestion, modeling, serving.
- **Trigger event-driven**: nuova elezione caricata → job di stima; aggiornamento ISTAT → ricalcolo prior, non urgente.

### 4.4 Versionamento delle stime
- Ogni run → nuovo `model_runs._id`; `estimations` taggati col run; **niente update in place**.
- L'API serve di default l'**ultimo run completato e validato** (`status=completed`, `rhat_max<1.05`), ma può servire run storici (riproducibilità, confronto "come vedevamo il consenso a gennaio").
- Code + data + hyperparams hashati nel manifest → riproducibilità totale.

---

## 5. Gestione dei flussi elettorali

Obiettivo: stimare la **matrice di transizione** `P` tra due elezioni consecutive, dove `P_{ij}` = frazione di chi ha votato `i` nell'elezione A e vota `j` nell'elezione B (incluse colonne/righe **astensione**). Le righe sommano a 1.

### 5.1 Il problema: ecological inference
Non osserviamo il comportamento individuale, solo **aggregati per sezione/comune**: quote di A e quote di B nella stessa area. Dedurre transazioni individuali da marginali aggregati è *ecological inference* — sottodeterminato, va vincolato statisticamente.

### 5.2 Modello bayesiano gerarchico dei flussi
Per ogni area `a` con `N_a` elettori:
- vettore quote elezione A: `x_a` (incluse astensione); elezione B: `y_a`.
- **Vincolo contabile**: `y_a = P_a^T x_a` (i voti che arrivano a `j` = somma su `i` di chi passa i→j).
- Ogni riga `P_{i·,a}` (destinazioni di chi veniva da `i`) ~ **Dirichlet** centrata su una matrice nazionale `P̄`:
  ```
  P_{i·,a} ~ Dirichlet( α · P̄_{i·} )
  ```
  `α` controlla l'omogeneità territoriale dei flussi (α grande → comportamento uniforme tra aree).
- **Likelihood**: i conteggi osservati di B per area seguono una Multinomiale data `P_a^T x_a`.
- **Prior strutturali sensati**: forte massa sulla **diagonale** (fedeltà: la maggioranza resta), penalità su transizioni politicamente implausibili (es. flussi tra poli opposti) tramite il prior su `P̄` — *soft*, non hard, per non imporre conclusioni.

> Va dichiarato il limite: l'ecological inference soffre della **ecological fallacy**. Stimiamo flussi *compatibili con gli aggregati e con i prior*, non certezze individuali. Le aree con alta varianza interna (es. grandi città) sono meno informative; meglio sfruttare la **granularità di sezione** dove disponibile (sezioni piccole e omogenee → inferenza molto più stretta).

### 5.3 Output dei flussi
- `P̄` nazionale e `P_a` locali (campioni posterior → incertezza su ogni flusso).
- **Fedeltà (loyalty)** del partito `i` = `P̄_{ii}` (diagonale): stabilità dell'elettorato.
- **Provenienza dei voti guadagnati** da `j` = colonna `j` normalizzata: da dove arrivano.
- **Destinazione dei voti persi** da `i` = riga `i`: dove vanno.
- **Intensità** = entità dei flussi off-diagonale.

### 5.4 Collegamento con lo State Space
I flussi non sono solo output: **informano `Q`** (l'innovazione dello stato). Periodi di alta volatilità nei flussi → `Q` più grande in quel periodo → il filtro consente cambiamenti più rapidi del consenso. È il ponte tra il "come si muovono i voti" (flussi) e il "quanto può muoversi lo stato" (dinamica).

---

## 6. Problema dei partiti (entità nel tempo)

Sfide: cambi di nome, fusioni, scissioni, coalizioni, liste civiche locali.

### 6.1 Modello entità-relazione temporale
- **Entità canonica** (`parties`) stabile; le **etichette** osservate (`party_aliases`) hanno validità temporale e una **relazione tipata**:
  - `alias`: stesso partito, nome diverso (LeU→…), `weight=1`.
  - `merge`: due liste → una entità; le storiche confluiscono con pesi.
  - `split`: una entità → due; si attribuiscono quote `weight` (richiede decisione modellistica/storica, da documentare).
  - `coalition_member`: la lista contribuisce a una coalizione ma resta entità propria.
  - `civic`: lista civica locale → mappata a "civiche" o a un partito sponsor se identificabile, altrimenti entità `civic:<comune>`.
- **Continuità del consenso vs continuità giuridica.** Decisione esplicita per ogni caso: il modello segue la **continuità dell'elettorato** (chi votava X ora vota Y?), non l'identità legale. Le scelte vanno in `party_aliases` con `source` e nota, **versionate e auditabili**.

### 6.2 Processo operativo
- Ogni `raw_label` mai vista → **coda di reconciliation** con suggerimento automatico (fuzzy match sul nome + co-occorrenza territoriale) e **conferma umana**. Mai auto-merge silenzioso.
- I flussi elettorali (§5) trattano fusioni/scissioni come transizioni: una scissione è "parte di chi votava A ora vota A1, parte A2" — il modello dei flussi *misura* effettivamente lo split, riducendo l'arbitrarietà dei pesi.
- Le **coalizioni** si gestiscono a due livelli: stima per **lista** (entità) e stima per **coalizione** (somma membri all'elezione). L'utente può chiedere entrambe.

---

## 7. Strategia di validazione

### 7.1 Backtesting temporale (out-of-sample, rigoroso)
- **Schema**: addestra su tutte le elezioni fino a una data `T` (es. fino a 2018), poi **predici** lo stato latente alle date delle elezioni successive (2019 europee, 2022 politiche, 2024 europee…) e confronta la stima nazionale/regionale con il risultato reale.
- È un test onesto perché lo State Space *predice* `η_t` proiettando il random walk in avanti senza aver visto l'elezione target; poi quell'elezione viene "rivelata".

### 7.2 Metriche
- **Punto**: MAE/RMSE sulle quote per partito.
- **Probabilistiche (le più importanti, perché l'output è una distribuzione)**:
  - **CRPS** (Continuous Ranked Probability Score): premia distribuzioni ben calibrate e affilate.
  - **Log predictive density** dell'esito reale.
  - **Calibrazione/Coverage**: l'IC95% deve contenere il valore vero ~95% delle volte (su tante elezioni-partito). Plot di calibrazione (PIT histogram).
- **Flussi**: validazione difficile (manca ground truth individuale). Si valida **indirettamente**: i flussi stimati devono *ricostruire* i marginali B (errore di ricostruzione) e, dove esistono, confrontarsi con dati di panel/exit-poll accademici come riferimento esterno (non come training).

### 7.3 Diagnostica e robustezza
- **Convergenza MCMC**: R̂ < 1.01, ESS adeguato, no divergenze (HMC).
- **Posterior predictive checks**: rigenerare elezioni passate dal modello e confrontarle.
- **Sensitivity analysis** sugli iperparametri chiave (`Q`, `α` dei flussi, scelta partito di riferimento).
- **Ablation**: con/senza correzione affluenza, con/senza bias tipo-elezione, con/senza gerarchia → quantificare il contributo di ogni componente.
- **Leave-one-election-out**: togliere una elezione e vedere se il modello la "ri-prevede".

### 7.4 Soglia di accettazione
Un run va in produzione solo se: converge (R̂ ok), coverage95 ∈ [0.92, 0.97] sul backtest, CRPS non peggiore della baseline (es. "ultimo risultato politiche + random walk semplice").

---

## 8. Criticità e limiti

1. **Sparsità e irregolarità temporale.** Le politiche sono ogni ~5 anni. Tra una e l'altra il modello vive di europee/regionali/comunali, che misurano l'Italia *parzialmente e distorta*. L'incertezza nazionale **cresce** lontano dalle politiche — ed è corretto che sia così; il modello deve mostrarlo onestamente (IC che si allargano).
2. **Identificabilità bias × tempo.** `β_{k,τ}` assume regolarità del *delta tipo-elezione*, ma quel delta **evolve** (un partito può europeizzarsi). Mitigazione: rendere `β` lentamente variabile nel tempo (altro random walk), al costo di più parametri/dati.
3. **Ecological fallacy nei flussi** (§5.2). Stime *compatibili*, non certezze. Onestà nell'output: bande larghe dove i dati aggregati non vincolano.
4. **Cambio confini e sistemi elettorali.** Riforme (Rosatellum, soglie, premi) cambiano *come* i voti si traducono. Vanno modellate o segmentate; confronti tra sistemi diversi richiedono cautela.
5. **Coverage geografica non casuale.** I comuni che votano in un dato anno **non sono un campione casuale** d'Italia (dipende dalla scadenza dei mandati). `γ`/peso correggono parzialmente, ma resta bias di selezione → da monitorare.
6. **Liste civiche e localismi** difficili da mappare su entità nazionali; rischio di rumore. Vanno isolate in categorie dedicate.
7. **Eventi-shock** (scissioni improvvise, scandali) violano il random walk liscio. Mitigazione: layer HMM di regime (§2.7) o `Q` adattivo guidato dai flussi.
8. **Non è un sostituto dei sondaggi per il *breaking news*.** Il sistema misura il consenso *strutturale* a media frequenza; non coglie scossoni dell'ultima settimana finché non c'è un'elezione.
9. **Astensione come "partito".** Trattarla come opzione è essenziale per i flussi, ma la sua eterogeneità (astensione fisiologica vs di protesta) è un'ulteriore fonte di incertezza.

---

## 9. Miglioramenti futuri

1. **Sondaggi come osservazione opzionale e separata.** Aggiungerli come *ulteriore* equazione di osservazione con il proprio bias di casa (`house effect`) e rumore grande, **disattivabile**. Il sistema resta "senza sondaggi" by default; i sondaggi possono solo stringere l'incertezza tra elezioni, mai dominare.
2. **Covariate demografiche nei prior** (`δ` funzione di età/reddito/istruzione/urbanizzazione ISTAT): MRP-like (Multilevel Regression and Poststratification) per stimare aree non votanti in una tornata e migliorare il bottom-up.
3. **β e Q tempo-varianti** per catturare l'evoluzione degli effetti tipo-elezione e della volatilità.
4. **Modello dei flussi gerarchico multi-livello** (sezione→comune→regione) con pooling, sfruttando il dato di sezione dove disponibile.
5. **Nowcasting** integrando indicatori esogeni *opzionali* (affluenza parziale in giornata, dati socioeconomici ad alta frequenza) — sempre come segnali deboli, separati.
6. **Online/streaming inference** (Particle filter / SMC) per aggiornare il posterior in minuti all'arrivo dei dati di sezione, senza rifare l'intero NUTS.
7. **Spiegabilità**: decomposizione di ogni stima nei contributi (quanto pesa l'ultima europea? quanto la gerarchia regionale?) per trasparenza.
8. **API di scenario**: "se l'affluenza scendesse al 50%, come cambierebbe la stima?".

---

## 10. Stack di implementazione (quando si passerà al codice)

- **Inferenza**: PyMC o NumPyro (HMC/NUTS) per la stima full-batch; opzionale filtro particellare custom per l'update incrementale.
- **ETL**: Python + pydantic (validazione) + Prefect/Airflow (orchestrazione) + Celery/Redis (code).
- **Storage**: MongoDB (documenti) + GridFS/object storage (posterior samples) + indici come §3.5.
- **Serving**: Flask (REST) → endpoint per stima nazionale/regionale, trend, mappa (GeoJSON), P(>soglia), distribuzioni.
- **Riproducibilità**: ogni run firmato (code hash + data hash + hyperparams) in `model_runs`.

---

### Nota finale del team
La scommessa metodologica è una sola e va tenuta ferma: **trattare le elezioni come misurazioni distorte di uno stato latente continuo**, non come previsioni da indovinare. Tutto il resto (gerarchia, bias tipo-elezione, affluenza, flussi) sono *correzioni della lente*. Questo rende il sistema un vero "motore di aggiornamento continuo del consenso", coerente, onesto sull'incertezza e indipendente dai sondaggi.
