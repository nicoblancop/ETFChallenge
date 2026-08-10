import pandas as pd
import yfinance as yf
import numpy as np
import json
from datetime import datetime
import os

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
    tickers_set = set(['SPY'])

    for index, row in df_validos.iterrows():
        name = row['Nombre completo del participante']
        if pd.isna(name):
            name = "Anónimo"
        etf_name = row['Nombre del ETF']
        if pd.isna(etf_name):
            etf_name = f"ETF_{name}"
            
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
    
    data = yf.download(all_tickers, start="2026-03-02")['Close']
    if data.empty:
        print("No se pudieron descargar los datos.")
        return
        
    market_returns = data.pct_change().dropna(how='all').fillna(0)
    market_dates = [d.strftime("%Y-%m-%d") for d in market_returns.index]
    
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
            
        # Métricas de riesgo y retorno
        total_return = current_cum - 1
        
        if len(p_active_returns) > 1:
            arr = np.array(p_active_returns)
            volatility = np.std(arr, ddof=1) * np.sqrt(252)
            downside = arr[arr < 0]
            down_vol = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else 0
            mean_ret = np.mean(arr)
            sortino = ((mean_ret * 252) - RISK_FREE_RATE) / down_vol if down_vol > 0 else 0
        else:
            volatility = 0
            sortino = 0
            
        participants_stats.append({
            'name': p['name'],
            'etf_name': p['etf_name'],
            'totalReturn': total_return,
            'volatility': volatility,
            'sortino': sortino,
            'cumReturns': p_cum_returns
        })
        
    # Benchmark SPY
    spy_active = market_returns['SPY'].values
    spy_cum = (1 + spy_active).cumprod()
    spy_cum_list = [1.0] + list(spy_cum)[:-1] if len(spy_cum) > 0 else [] 
    spy_volatility = np.std(spy_active, ddof=1) * np.sqrt(252) if len(spy_active) > 1 else 0
    spy_total_return = spy_cum[-1] - 1 if len(spy_cum) > 0 else 0
    
    benchmark = {
        'totalReturn': spy_total_return,
        'volatility': spy_volatility,
        'cumReturns': spy_cum.tolist()
    }
    
    # Inyectar datos en formato JSON a string
    p_json = json.dumps(participants_stats)
    b_json = json.dumps(benchmark)
    d_json = json.dumps(market_dates)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Generar el HTML final estático
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF Challenge - Dashboard en Tiempo Real</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ background-color: #f0f4f8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .glass-panel {{ background: rgba(255, 255, 255, 0.95); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    </style>
</head>
<body class="text-gray-800">
<div class="max-w-7xl mx-auto px-4 py-8">
    <div class="text-center mb-10">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-2"><i class="fa-solid fa-chart-line text-blue-600"></i> ETF Challenge - IFA MARCONI</h1>
        <p class="text-lg text-gray-500">Dashboard de rendimiento (Equal-weight vs SPY)</p>
        <p class="text-sm text-gray-400 mt-2">Última actualización de datos: {now_str}</p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="glass-panel p-6 border-t-4 border-green-500">
            <h3 class="text-sm uppercase font-bold text-gray-500 mb-1">Líder Actual</h3>
            <div id="card-leader-name" class="text-2xl font-bold">--</div>
            <div id="card-leader-return" class="text-xl font-semibold text-green-600">--</div>
        </div>
        <div class="glass-panel p-6 border-t-4 border-blue-500">
            <h3 class="text-sm uppercase font-bold text-gray-500 mb-1">Mejor Riesgo (Menor Vol)</h3>
            <div id="card-vol-name" class="text-2xl font-bold">--</div>
            <div id="card-vol-value" class="text-xl font-semibold text-blue-600">--</div>
        </div>
        <div class="glass-panel p-6 border-t-4 border-purple-500">
            <h3 class="text-sm uppercase font-bold text-gray-500 mb-1">Benchmark (SPY)</h3>
            <div id="card-spy-return" class="text-2xl font-bold">--</div>
            <div id="card-spy-vol" class="text-md text-gray-500">--</div>
        </div>
    </div>

    <div class="glass-panel p-4 md:p-6 mb-8">
        <h2 class="text-2xl font-bold mb-4">Evolución de Rendimiento Acumulado</h2>
        <div id="main-chart" style="width: 100%; height: 600px;"></div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div class="glass-panel p-6">
            <h2 class="text-xl font-bold mb-4"><i class="fa-solid fa-trophy text-yellow-500 mr-2"></i>Top 10 - Mayor Retorno Total</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead class="bg-gray-100 text-gray-600 text-sm uppercase"><tr><th class="p-3">#</th><th class="p-3">ETF</th><th class="p-3 text-right">Retorno</th></tr></thead>
                    <tbody id="table-return-body" class="text-sm"></tbody>
                </table>
            </div>
        </div>
        <div class="glass-panel p-6">
            <h2 class="text-xl font-bold mb-4"><i class="fa-solid fa-shield-halved text-blue-500 mr-2"></i>Top 10 - Menor Volatilidad</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead class="bg-gray-100 text-gray-600 text-sm uppercase"><tr><th class="p-3">#</th><th class="p-3">ETF</th><th class="p-3 text-right">Volatilidad</th></tr></thead>
                    <tbody id="table-vol-body" class="text-sm"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
    // Los datos matemáticos están inyectados estáticamente aquí.
    const data = {p_json};
    const spy = {b_json};
    const dates = {d_json};
    const colors = ["#E63946","#F4A261","#2A9D8F","#457B9D","#A8DADC","#6A4C93","#F77F00","#52B788","#219EBC","#FB8500"];
    
    const formatPct = (v) => (v * 100).toFixed(2) + '%';
    
    const byRet = [...data].sort((a,b) => b.totalReturn - a.totalReturn);
    const byVol = [...data].sort((a,b) => a.volatility - b.volatility);
    
    if(byRet.length > 0) {{
        document.getElementById('card-leader-name').innerText = byRet[0].etf_name;
        document.getElementById('card-leader-return').innerText = formatPct(byRet[0].totalReturn);
        document.getElementById('card-vol-name').innerText = byVol[0].etf_name;
        document.getElementById('card-vol-value').innerText = formatPct(byVol[0].volatility);
    }}
    
    document.getElementById('card-spy-return').innerText = formatPct(spy.totalReturn);
    document.getElementById('card-spy-vol').innerText = `Volatilidad: ${{formatPct(spy.volatility)}}`;
    
    const renderT = (id, arr, key) => {{
        let h = '';
        arr.slice(0,10).forEach((x,i) => {{
            let bg = i === 0 ? 'bg-yellow-50' : (i % 2 === 0 ? 'bg-white' : 'bg-gray-50');
            h += `<tr class="${{bg}} border-b"><td class="p-3">${{i+1}}</td><td class="p-3 font-semibold">${{x.etf_name}}<div class="text-xs font-normal text-gray-500">${{x.name}}</div></td><td class="p-3 text-right font-medium">${{formatPct(x[key])}}</td></tr>`;
        }});
        let sVal = key === 'totalReturn' ? spy.totalReturn : spy.volatility;
        h += `<tr class="bg-gray-200 font-bold border-t-2 border-gray-300"><td class="p-3">—</td><td class="p-3">SPY (Benchmark)</td><td class="p-3 text-right">${{formatPct(sVal)}}</td></tr>`;
        document.getElementById(id).innerHTML = h;
    }};
    renderT('table-return-body', byRet, 'totalReturn');
    renderT('table-vol-body', byVol, 'volatility');
    
    const traces = byRet.slice(0, 15).map((p, i) => ({{
        x: dates, y: p.cumReturns, mode: 'lines', name: p.etf_name, 
        line: {{width: i<3?3:1.5, color: colors[i%colors.length]}}, opacity: i<3?1:0.7
    }}));
    traces.push({{ x: dates, y: spy.cumReturns, mode: 'lines', name: 'SPY Benchmark', line: {{width: 3, color: '#111', dash: 'dot'}} }});
    
    Plotly.newPlot('main-chart', traces, {{
        margin: {{t:20, r:20, b:40, l:40}}, xaxis: {{showgrid:true, gridcolor:'#e5e7eb'}}, 
        yaxis: {{tickformat:'.0%', showgrid:true, gridcolor:'#e5e7eb'}}, 
        hovermode:'x unified', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        legend: {{ orientation: "h", y: -0.15 }}
    }}, {{responsive:true}});
</script>
</body>
</html>"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Dashboard generado exitosamente: index.html")

if __name__ == "__main__":
    main()
