import pandas as pd
import yfinance as yf
import numpy as np
import json
from datetime import datetime
import os

def clean_float(v):
    if pd.isna(v) or np.isinf(v): return 0.0
    return float(v)

def main():
    FILE_PATH = "ETF Challenge - IFA MARCONI.xlsx"
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} no encontrado.")
        return

    print("Leyendo Excel...")
    df = pd.read_excel(FILE_PATH)
    
    COMPETITION_START = pd.to_datetime("2026-03-02")
    DEADLINE = pd.to_datetime("2026-03-06 23:59:59")
    RISK_FREE_RATE = 0.04
    
    df['Hora de inicio'] = pd.to_datetime(df['Hora de inicio'])
    df_validos = df[df['Hora de inicio'] <= DEADLINE].copy()

    participants = []
    
    # Agregamos los benchmarks principales para análisis comparativo
    tickers_set = set(['SPY', 'DIA', 'QQQ'])

    for index, row in df_validos.iterrows():
        name = row['Nombre completo del participante']
        if pd.isna(name): name = "Anónimo"
        etf_name = row['Nombre del ETF']
        if pd.isna(etf_name): etf_name = f"ETF_{name}"
            
        tickers = []
        for i in range(1, 6):
            t = str(row[f'Ticker {i}']).strip().upper()
            if t != 'NAN' and t:
                tickers.append(t)
                tickers_set.add(t)
        
        start_time = row['Hora de inicio']
        start_date = COMPETITION_START.normalize() if start_time < COMPETITION_START else (start_time + pd.Timedelta(days=1)).normalize()
            
        participants.append({
            'name': str(name),
            'etf_name': str(etf_name),
            'tickers': list(set(tickers)),
            'start_date': start_date
        })

    all_tickers = list(tickers_set)
    print(f"Descargando datos de {len(all_tickers)} activos mediante yfinance...")
    
    # Usamos YF para bajar el histórico
    data = yf.download(all_tickers, start="2026-03-02")['Close']
    if data.empty:
        print("No se pudieron descargar los datos.")
        return
        
    market_returns = data.pct_change().dropna(how='all').fillna(0)
    market_dates = [d.strftime("%Y-%m-%d") for d in market_returns.index]
    
    spy_active = market_returns['SPY'].values
    spy_var = np.var(spy_active, ddof=1) if len(spy_active) > 1 else 0
    
    # 1. Calcular métricas por Ticker individual (incluyendo benchmarks)
    ticker_data = {}
    for t in all_tickers:
        if t in market_returns.columns:
            t_ret = market_returns[t].values
            t_cum = (1 + t_ret).cumprod()
            t_cum_list = [1.0] + list(t_cum)[:-1] if len(t_cum) > 0 else []
            
            t_tot = t_cum[-1] - 1 if len(t_cum) > 0 else 0
            t_vol = np.std(t_ret, ddof=1) * np.sqrt(252) if len(t_ret) > 1 else 0
            
            # Max Drawdown
            peak = np.maximum.accumulate(t_cum)
            drawdowns = (t_cum - peak) / peak if len(peak) > 0 else [0]
            t_mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            ticker_data[t] = {
                'totalReturn': clean_float(t_tot),
                'volatility': clean_float(t_vol),
                'maxDrawdown': clean_float(t_mdd),
                'cumReturns': [clean_float(v) for v in t_cum_list]
            }

    # 2. Calcular métricas por Participante
    participants_stats = []
    for p in participants:
        p_start = p['start_date']
        p_cum_returns = []
        p_active_returns = []
        current_cum = 1.0
        
        for date_idx, date in enumerate(market_returns.index):
            if date < p_start:
                p_cum_returns.append(1.0)
                continue
                
            valid_tickers = [t for t in p['tickers'] if t in market_returns.columns]
            if not valid_tickers:
                p_cum_returns.append(current_cum)
                p_active_returns.append(0)
                continue
                
            day_sum = sum([market_returns[t].iloc[date_idx] for t in valid_tickers])
            day_avg = day_sum / len(valid_tickers)
            
            p_active_returns.append(day_avg)
            current_cum *= (1 + day_avg)
            p_cum_returns.append(current_cum)
            
        total_return = current_cum - 1
        
        if len(p_active_returns) > 1:
            arr = np.array(p_active_returns)
            mean_ret = np.mean(arr)
            volatility = np.std(arr, ddof=1) * np.sqrt(252)
            
            downside = arr[arr < 0]
            down_vol = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else 0
            
            sortino = ((mean_ret * 252) - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0
            sharpe = ((mean_ret * 252) - RISK_FREE_RATE) / volatility if volatility > 0 else 0
            
            # Beta vs SPY
            spy_aligned = spy_active[-len(arr):] # alinear fechas
            if spy_var > 0 and len(spy_aligned) == len(arr):
                cov = np.cov(arr, spy_aligned)[0][1]
                beta = cov / spy_var
            else:
                beta = 0
                
            # Max Drawdown
            p_cum_arr = np.array(p_cum_returns[-len(arr):])
            peak_p = np.maximum.accumulate(p_cum_arr)
            drawdowns_p = (p_cum_arr - peak_p) / peak_p if len(peak_p) > 0 else [0]
            max_dd = np.min(drawdowns_p) if len(drawdowns_p) > 0 else 0
            
        else:
            volatility, sortino, sharpe, beta, max_dd = 0, 0, 0, 0, 0
            
        participants_stats.append({
            'name': p['name'],
            'etf_name': p['etf_name'],
            'tickers': p['tickers'],
            'totalReturn': clean_float(total_return),
            'volatility': clean_float(volatility),
            'sharpe': clean_float(sharpe),
            'sortino': clean_float(sortino),
            'beta': clean_float(beta),
            'maxDrawdown': clean_float(max_dd),
            'cumReturns': [clean_float(v) for v in p_cum_returns]
        })

    # Serializar todo a JSON
    p_json = json.dumps(participants_stats)
    t_json = json.dumps(ticker_data)
    d_json = json.dumps(market_dates)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # HTML estático (usamos replace para evitar conflictos de llaves en JS/CSS)
    html_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF Challenge - Dashboard Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #f3f4f6; font-family: 'Inter', sans-serif; }
        .glass-panel { background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .stat-card { border-left: 4px solid; padding: 1rem 1.5rem; }
    </style>
</head>
<body class="text-gray-800">
<div class="max-w-7xl mx-auto px-4 py-8">
    
    <!-- HEADER -->
    <div class="text-center mb-10">
        <h1 class="text-4xl font-extrabold text-gray-900 tracking-tight mb-2"><i class="fa-solid fa-layer-group text-blue-600"></i> ETF Challenge Analytics</h1>
        <p class="text-lg text-gray-500">Evaluación Institucional (Retorno, Riesgo, Sharpe & Drawdown)</p>
        <p class="text-sm text-gray-400 mt-2"><i class="fa-regular fa-clock"></i> Última actualización: %%NOW%%</p>
    </div>

    <!-- MAIN CHART -->
    <div class="glass-panel p-6 mb-8">
        <h2 class="text-2xl font-bold mb-4 border-b pb-2">Evolución Global (Top 15 + Benchmarks)</h2>
        <div id="main-chart" style="width: 100%; height: 600px;"></div>
    </div>

    <!-- RANKINGS GRID -->
    <h2 class="text-3xl font-bold mb-6 text-gray-800">Clasificación General</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <!-- Retorno -->
        <div class="glass-panel p-4">
            <h3 class="font-bold text-green-700 mb-3"><i class="fa-solid fa-arrow-trend-up"></i> Top Retorno</h3>
            <table class="w-full text-sm text-left"><tbody id="rank-ret"></tbody></table>
        </div>
        <!-- Sharpe -->
        <div class="glass-panel p-4">
            <h3 class="font-bold text-purple-700 mb-3"><i class="fa-solid fa-scale-balanced"></i> Mejores Sharpe</h3>
            <table class="w-full text-sm text-left"><tbody id="rank-sha"></tbody></table>
        </div>
        <!-- Sortino -->
        <div class="glass-panel p-4">
            <h3 class="font-bold text-indigo-700 mb-3"><i class="fa-solid fa-bolt"></i> Mejores Sortino</h3>
            <table class="w-full text-sm text-left"><tbody id="rank-sor"></tbody></table>
        </div>
        <!-- Volatilidad -->
        <div class="glass-panel p-4">
            <h3 class="font-bold text-blue-700 mb-3"><i class="fa-solid fa-shield-halved"></i> Menor Riesgo (Vol)</h3>
            <table class="w-full text-sm text-left"><tbody id="rank-vol"></tbody></table>
        </div>
    </div>

    <!-- SECCIÓN: ANALIZADOR DE CARTERA -->
    <div class="glass-panel p-6 mb-8 border-t-4 border-blue-600">
        <h2 class="text-2xl font-bold mb-4">Analizador de Cartera Individual</h2>
        <select id="p-select" class="mb-6 block w-full max-w-md bg-gray-50 border border-gray-300 rounded-lg p-2.5 font-semibold text-gray-700 focus:ring-blue-500">
            <option value="">Selecciona un participante...</option>
        </select>
        
        <div id="p-view" class="hidden">
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                <div class="stat-card border-green-500 bg-gray-50"><p class="text-xs text-gray-500 uppercase">Retorno</p><p id="pv-ret" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-blue-500 bg-gray-50"><p class="text-xs text-gray-500 uppercase">Volatilidad</p><p id="pv-vol" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-purple-500 bg-gray-50"><p class="text-xs text-gray-500 uppercase">Ratio Sharpe</p><p id="pv-sha" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-indigo-500 bg-gray-50"><p class="text-xs text-gray-500 uppercase">Ratio Sortino</p><p id="pv-sor" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-red-500 bg-gray-50"><p class="text-xs text-gray-500 uppercase">Max Drawdown</p><p id="pv-mdd" class="text-xl font-bold text-red-600">--</p></div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-1">
                    <h4 class="font-bold text-gray-700 mb-2">Composición y Aportes</h4>
                    <p class="text-xs text-gray-500 mb-2">Beta de Cartera vs SPY: <span id="pv-beta" class="font-bold text-gray-800">--</span></p>
                    <table class="w-full text-sm text-left border rounded-lg bg-white overflow-hidden shadow-sm">
                        <thead class="bg-gray-100"><tr><th class="p-2">Ticker</th><th class="p-2 text-right">Ret. Total</th><th class="p-2 text-right">Vol</th></tr></thead>
                        <tbody id="pv-tickers"></tbody>
                    </table>
                </div>
                <div class="lg:col-span-2">
                    <div id="p-chart" style="width: 100%; height: 350px;"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- SECCIÓN: ANALIZADOR DE TICKER VS ÍNDICES -->
    <div class="glass-panel p-6 mb-8 border-t-4 border-gray-800">
        <h2 class="text-2xl font-bold mb-4">Deep Dive: Ticker vs Índices (SPY, QQQ, DIA)</h2>
        <select id="t-select" class="mb-6 block w-full max-w-md bg-gray-50 border border-gray-300 rounded-lg p-2.5 font-semibold text-gray-700 focus:ring-gray-800">
            <option value="">Selecciona un Ticker del Challenge...</option>
        </select>
        
        <div id="t-view" class="hidden">
            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="stat-card border-green-600 bg-white shadow-sm"><p class="text-xs text-gray-500 uppercase">Retorno</p><p id="tv-ret" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-blue-600 bg-white shadow-sm"><p class="text-xs text-gray-500 uppercase">Volatilidad</p><p id="tv-vol" class="text-xl font-bold">--</p></div>
                <div class="stat-card border-red-600 bg-white shadow-sm"><p class="text-xs text-gray-500 uppercase">Max Drawdown</p><p id="tv-mdd" class="text-xl font-bold text-red-600">--</p></div>
            </div>
            <div id="t-chart" style="width: 100%; height: 450px;"></div>
        </div>
    </div>

</div>

<script>
    const pData = %%P_JSON%%;
    const tData = %%T_JSON%%;
    const dates = %%D_JSON%%;
    
    const formatPct = (v) => (v * 100).toFixed(2) + '%';
    const formatNum = (v) => v.toFixed(2);
    
    // Sort functions
    const byRet = [...pData].sort((a,b) => b.totalReturn - a.totalReturn);
    const bySha = [...pData].sort((a,b) => b.sharpe - a.sharpe);
    const bySor = [...pData].sort((a,b) => b.sortino - a.sortino);
    const byVol = [...pData].sort((a,b) => a.volatility - b.volatility);

    // Render Rankings
    const renderMiniRank = (id, arr, key, isFmtNum = false) => {
        let h = '';
        arr.slice(0,10).forEach((x,i) => {
            let val = isFmtNum ? formatNum(x[key]) : formatPct(x[key]);
            h += `<tr class="border-b hover:bg-gray-50"><td class="py-2 text-gray-400 w-6">${i+1}</td><td class="py-2 font-semibold text-gray-800">${x.etf_name}</td><td class="py-2 text-right font-medium">${val}</td></tr>`;
        });
        document.getElementById(id).innerHTML = h;
    };
    renderMiniRank('rank-ret', byRet, 'totalReturn');
    renderMiniRank('rank-sha', bySha, 'sharpe', true);
    renderMiniRank('rank-sor', bySor, 'sortino', true);
    renderMiniRank('rank-vol', byVol, 'volatility');

    // Main Chart setup
    const colors = ["#E63946","#F4A261","#2A9D8F","#457B9D","#A8DADC","#6A4C93","#F77F00","#52B788","#219EBC","#FB8500"];
    const mainTraces = byRet.slice(0, 15).map((p, i) => ({
        x: dates, y: p.cumReturns, mode: 'lines', name: `${p.etf_name} (${formatPct(p.totalReturn)})`, 
        line: {width: i<3?2.5:1.5, color: colors[i%colors.length]}, opacity: i<3?1:0.6
    }));
    // Add Benchmarks to Main Chart
    const bmarks = [
        {tic: 'SPY', col: '#111827', name: 'S&P 500 (SPY)'},
        {tic: 'QQQ', col: '#0284C7', name: 'Nasdaq (QQQ)'}
    ];
    bmarks.forEach(b => {
        if(tData[b.tic]) {
            mainTraces.push({ x: dates, y: tData[b.tic].cumReturns, mode: 'lines', name: b.name, line: {width: 2.5, color: b.col, dash: 'dash'} });
        }
    });

    const layoutBase = {
        margin: {t:20, r:10, b:30, l:40}, xaxis: {showgrid:true, gridcolor:'#e5e7eb'}, 
        yaxis: {tickformat:'.1%', showgrid:true, gridcolor:'#e5e7eb'}, 
        hovermode:'x unified', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
    };
    Plotly.newPlot('main-chart', mainTraces, {...layoutBase, legend: {orientation: "h", y: -0.15}}, {responsive:true});

    // --- LOGIC: PARTICIPANT ANALYZER ---
    const pSelect = document.getElementById('p-select');
    byRet.forEach(p => { pSelect.add(new Option(`${p.etf_name} - ${p.name}`, p.etf_name)); });
    
    pSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        const view = document.getElementById('p-view');
        if(!val) { view.classList.add('hidden'); return; }
        
        const p = pData.find(x => x.etf_name === val);
        view.classList.remove('hidden');
        
        document.getElementById('pv-ret').innerText = formatPct(p.totalReturn);
        document.getElementById('pv-vol').innerText = formatPct(p.volatility);
        document.getElementById('pv-sha').innerText = formatNum(p.sharpe);
        document.getElementById('pv-sor').innerText = formatNum(p.sortino);
        document.getElementById('pv-mdd').innerText = formatPct(p.maxDrawdown);
        document.getElementById('pv-beta').innerText = formatNum(p.beta);
        
        // Tickers list
        let h = '';
        p.tickers.forEach(t => {
            const td = tData[t];
            if(!td) return;
            const retCol = td.totalReturn >= 0 ? 'text-green-600' : 'text-red-600';
            h += `<tr class="border-b"><td class="p-2 font-bold">${t}</td><td class="p-2 text-right ${retCol} font-semibold">${formatPct(td.totalReturn)}</td><td class="p-2 text-right text-gray-500">${formatPct(td.volatility)}</td></tr>`;
        });
        document.getElementById('pv-tickers').innerHTML = h;
        
        // Mini Chart (Participant vs SPY)
        const pTraces = [
            { x: dates, y: p.cumReturns, mode: 'lines', name: p.etf_name, line: {width: 3, color: '#2563EB'} }
        ];
        if(tData['SPY']) pTraces.push({ x: dates, y: tData['SPY'].cumReturns, mode: 'lines', name: 'SPY', line: {width: 2, color: '#9CA3AF', dash: 'dot'} });
        Plotly.newPlot('p-chart', pTraces, {...layoutBase, margin:{t:10,r:10,b:30,l:40}}, {responsive:true});
    });

    // --- LOGIC: TICKER ANALYZER ---
    const tSelect = document.getElementById('t-select');
    const allTics = Object.keys(tData).sort();
    allTics.forEach(t => { tSelect.add(new Option(t, t)); });

    tSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        const view = document.getElementById('t-view');
        if(!val) { view.classList.add('hidden'); return; }
        
        const td = tData[val];
        view.classList.remove('hidden');
        
        document.getElementById('tv-ret').innerText = formatPct(td.totalReturn);
        document.getElementById('tv-ret').className = `text-xl font-bold ${td.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`;
        document.getElementById('tv-vol').innerText = formatPct(td.volatility);
        document.getElementById('tv-mdd').innerText = formatPct(td.maxDrawdown);

        // Chart against Indices
        const tTraces = [{ x: dates, y: td.cumReturns, mode: 'lines', name: val, line: {width: 3, color: '#DC2626'} }];
        if(tData['SPY']) tTraces.push({ x: dates, y: tData['SPY'].cumReturns, mode: 'lines', name: 'SPY', line: {width: 2, color: '#111827', dash: 'dash'} });
        if(tData['QQQ']) tTraces.push({ x: dates, y: tData['QQQ'].cumReturns, mode: 'lines', name: 'QQQ', line: {width: 2, color: '#0284C7', dash: 'dot'} });
        if(tData['DIA']) tTraces.push({ x: dates, y: tData['DIA'].cumReturns, mode: 'lines', name: 'DIA', line: {width: 2, color: '#D97706', dash: 'dot'} });
        
        Plotly.newPlot('t-chart', tTraces, {...layoutBase, legend: {orientation: "h", y: -0.15}}, {responsive:true});
    });
</script>
</body>
</html>"""

    # Inyección limpia para no romper CSS/JS
    html_content = html_template.replace("%%P_JSON%%", p_json).replace("%%T_JSON%%", t_json).replace("%%D_JSON%%", d_json).replace("%%NOW%%", now_str)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Dashboard Pro generado exitosamente: index.html")

if __name__ == "__main__":
    main()
