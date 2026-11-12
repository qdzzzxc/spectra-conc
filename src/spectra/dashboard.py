"""Генерация автономного HTML-дашборда (Plotly с CDN, данные встроены в страницу)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"



def build_dashboard(results: pd.DataFrame, oof: pd.DataFrame, ds, out_path: str, title="Концентрация по спектрам"):
    res = results[results["split"].isin(["group_cv", "extrapolation"])].copy()
    cv = res[res.split == "group_cv"].sort_values("rmse").reset_index(drop=True)
    ex = res[res.split == "extrapolation"]
    payload = {
        "title": title,
        "info": ds.summary(),
        "cv": cv.replace({np.nan: None}).to_dict("records"),
        "ex": ex.replace({np.nan: None}).to_dict("records"),
        "wl": [float(x) for x in ds.wl],
        "mean_spec": [float(x) for x in ds.X.mean(0)],
        "spectra": [[float(v) for v in ds.X[i]] for i in np.linspace(0, len(ds) - 1, min(len(ds), 30)).astype(int)],
        "spectra_y": [float(ds.y[i]) for i in np.linspace(0, len(ds) - 1, min(len(ds), 30)).astype(int)],
        "oof": oof[["model", "prep", "y_true", "y_pred"]].round(5).to_dict("records") if len(oof) else [],
    }
    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><script src="{PLOTLY}"></script>
<style>
:root{{--bg:#fff;--fg:#1c2733;--mut:#667;--card:#f5f7fa;--bd:#dde3ea}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12161c;--fg:#e6ebf1;--mut:#9aa6b2;--card:#1a2029;--bd:#2b3440}}}}
body{{font-family:Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}.mut{{color:var(--mut);font-size:13px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:5px 8px;border-bottom:1px solid var(--bd);text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}select{{padding:4px}}
</style></head><body>
<h1>{title}</h1><div class="mut" id="info"></div>
<h2>Спектры</h2><div class="card"><div id="spec" style="height:300px"></div></div>
<h2>Рейтинг пайплайнов (вложенная CV, разбиение по группам)</h2>
<div class="card"><div id="bar" style="height:380px"></div><div style="overflow:auto"><table id="tbl"></table></div></div>
<h2>Предсказано vs истинное</h2>
<div class="card"><select id="sel"></select><div id="sc" style="height:380px"></div></div>
<h2>Экстраполяция: обучение на низких концентрациях, тест на верхних 20%</h2>
<div class="card"><div id="ex" style="height:340px"></div><div class="mut">Если RMSE здесь сильно выше, чем в CV, модель плохо выходит за пределы калибровочного диапазона (типично для деревьев).</div></div>
<script>
const D={json.dumps(payload, ensure_ascii=False)};
document.getElementById('info').textContent=D.info;
const th=getComputedStyle(document.body);const L={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:th.color}},margin:{{t:20,l:55,r:15,b:90}}}};
Plotly.newPlot('spec',D.spectra.map((s,i)=>({{x:D.wl,y:s,mode:'lines',line:{{width:1}},name:'c='+D.spectra_y[i].toFixed(1),showlegend:false}})),{{...L,xaxis:{{title:'длина волны / сдвиг'}},yaxis:{{title:'сигнал'}}}});
const lab=r=>r.prep+' | '+r.model;
Plotly.newPlot('bar',[{{type:'bar',x:D.cv.map(lab),y:D.cv.map(r=>r.rmse),marker:{{color:'#2a7fff'}}}}],{{...L,yaxis:{{title:'RMSE (CV)'}},xaxis:{{tickangle:-60}}}});
const cols=['prep','model','rmse','mae','bias','r2','rpd','sec'];
document.getElementById('tbl').innerHTML='<tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>'+D.cv.map(r=>'<tr>'+cols.map(c=>'<td>'+(typeof r[c]==='number'?r[c].toFixed(c==='r2'?4:3):r[c])+'</td>').join('')+'</tr>').join('');
const sel=document.getElementById('sel');D.cv.forEach(r=>{{const o=document.createElement('option');o.value=lab(r);o.textContent=lab(r);sel.appendChild(o)}});
function draw(){{const k=sel.value;const pts=D.oof.filter(p=>p.prep+' | '+p.model===k);const lo=Math.min(...pts.map(p=>p.y_true)),hi=Math.max(...pts.map(p=>p.y_true));
Plotly.newPlot('sc',[{{x:pts.map(p=>p.y_true),y:pts.map(p=>p.y_pred),mode:'markers',marker:{{size:5,opacity:.6}},name:'предсказания'}},{{x:[lo,hi],y:[lo,hi],mode:'lines',line:{{dash:'dash',color:'gray'}},name:'y=x'}}],{{...L,xaxis:{{title:'истинное'}},yaxis:{{title:'предсказанное'}}}})}}
sel.onchange=draw;if(D.cv.length)draw();
const exm=[...new Set(D.ex.map(r=>r.model))];
Plotly.newPlot('ex',exm.map(m=>({{type:'bar',name:m,x:D.ex.filter(r=>r.model===m).map(r=>r.prep),y:D.ex.filter(r=>r.model===m).map(r=>r.rmse)}})),{{...L,barmode:'group',yaxis:{{title:'RMSE на верхних концентрациях'}}}});
</script></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
