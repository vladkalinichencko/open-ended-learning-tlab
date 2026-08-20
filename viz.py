"""MLflow + runs/buffer.json -> one self-contained interactive page.

Three things have to be on the same page to mean anything: what the student can solve,
what the levels it fails on actually look like, and what the teacher is putting in the
buffer meanwhile. Separately each is a number without a referent.

    python viz.py                      # -> runs/report.html
"""

import argparse
import json
import pathlib

import mlflow

import levels

DEV = ["SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
       "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3"]

TEMPLATE = """<title>UED — курикулум против теста</title>
<style>
:root { --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; --wall:#2d3748; --path:#dd6b20; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; --wall:#5a6274; --path:#f6ad55; } }
:root[data-theme=dark] { --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038;
  --wall:#5a6274; --path:#f6ad55; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,system-ui,sans-serif;
       margin:0 auto; max-width:1100px; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:26px 0 8px; font-weight:600; }
p.note { color:var(--mut); margin:2px 0 12px; }
table { border-collapse:collapse; font-size:13px; } td,th { padding:3px 10px 3px 0; text-align:right; }
th:first-child,td:first-child { text-align:left; }
th { border-bottom:1px solid var(--line); font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; }
.card { border:1px solid var(--line); border-radius:6px; padding:8px 10px; overflow-x:auto; }
.card b { font-size:12px; } .card span { color:var(--mut); font-size:11px; }
svg { display:block; margin-top:4px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--mut); margin:6px 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
.ax { stroke:var(--line); } .tick { fill:var(--mut); font-size:10px; }
</style>
<h1>UED: чему учит учитель и на чём проваливается студент</h1>
<p class="note">Dev-уровни нарисованы руками и в обучении не используются. Буфер взят
прямо из чекпойнта прогона.</p>
<div id="app"></div>
<script>
const DATA = __DATA__;
const PAL = ["#2b6cb0","#c05621","#2f855a","#805ad5","#b83280","#4a5568"];
const SVG = new Set(["svg","g","path","line","text","rect","circle"]);
function el(tag, attrs, kids) {
  const n = document.createElementNS(SVG.has(tag) ? "http://www.w3.org/2000/svg"
    : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs||{})) n.setAttribute(k, attrs[k]);
  for (const c of (kids||[])) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
}
const fmt = v => v == null ? "—" : (Math.abs(v) >= 1000 ? v.toExponential(1) : (+v.toFixed(3)).toString());

function chart(series, o) {
  o = Object.assign({w: 500, h: 260, pad: 42}, o);
  const pts = series.flatMap(s => s.pts).filter(p => isFinite(p[1]));
  if (!pts.length) return el("svg", {width: o.w, height: o.h});
  let x0 = Math.min(...pts.map(p=>p[0])), x1 = Math.max(...pts.map(p=>p[0]));
  let y0 = Math.min(...pts.map(p=>p[1])), y1 = Math.max(...pts.map(p=>p[1]));
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= .5; y1 += .5; }
  const my = (y1-y0)*.08; y0 -= my; y1 += my;
  const X = v => o.pad + (v-x0)/(x1-x0)*(o.w-o.pad-10);
  const Y = v => o.h-24 - (v-y0)/(y1-y0)*(o.h-34);
  const g = el("svg", {width:o.w, height:o.h});
  g.appendChild(el("line",{x1:o.pad,y1:o.h-24,x2:o.w-10,y2:o.h-24,class:"ax"}));
  g.appendChild(el("line",{x1:o.pad,y1:12,x2:o.pad,y2:o.h-24,class:"ax"}));
  g.appendChild(el("text",{x:2,y:16,class:"tick"},[fmt(y1-my)]));
  g.appendChild(el("text",{x:2,y:o.h-26,class:"tick"},[fmt(y0+my)]));
  g.appendChild(el("text",{x:o.pad,y:o.h-8,class:"tick"},[fmt(x0)]));
  g.appendChild(el("text",{x:o.w-10,y:o.h-8,class:"tick","text-anchor":"end"},[fmt(x1)]));
  series.forEach((s,i) => g.appendChild(el("path",{fill:"none","stroke-width":1.8,
    stroke: s.color || PAL[i%PAL.length],
    d: s.pts.filter(p=>isFinite(p[1])).map((p,j)=>(j?"L":"M")+X(p[0])+" "+Y(p[1])).join(" ")})));
  return g;
}
function card(t, sub, svg) {
  return el("div",{class:"card"},[el("b",{},[t]), sub?el("span",{},[" — "+sub]):"", svg]);
}
function maze(lv, cell) {
  cell = cell || 11;
  const h = lv.walls.length, w = lv.walls[0].length;
  const g = el("svg",{width:w*cell, height:h*cell});
  for (let r=0;r<h;r++) for (let c=0;c<w;c++) if (lv.walls[r][c])
    g.appendChild(el("rect",{x:c*cell,y:r*cell,width:cell,height:cell,fill:"var(--wall)"}));
  if (lv.path && lv.path.length)
    g.appendChild(el("path",{fill:"none",stroke:"var(--path)","stroke-width":2.5,
      d: lv.path.map((p,i)=>(i?"L":"M")+(p[1]*cell+cell/2)+" "+(p[0]*cell+cell/2)).join(" ")}));
  g.appendChild(el("circle",{cx:lv.agent[1]*cell+cell/2,cy:lv.agent[0]*cell+cell/2,r:cell*0.32,
    fill:"#3182ce"}));
  g.appendChild(el("rect",{x:lv.goal[1]*cell+cell*0.2,y:lv.goal[0]*cell+cell*0.2,
    width:cell*0.6,height:cell*0.6,fill:"#38a169"}));
  return g;
}

const app = document.getElementById("app");

// --- solve rate по прогонам
if (Object.keys(DATA.runs).length) {
  app.appendChild(el("h2",{},["Solve rate на dev-наборе"]));
  const names = Object.keys(DATA.runs);
  app.appendChild(el("div",{class:"legend"}, names.map((n,i)=>
    el("span",{},[el("i",{style:"background:"+PAL[i%PAL.length]}), n]))));
  app.appendChild(card("среднее по восьми уровням","апдейт",
    chart(names.map(n=>({pts:DATA.runs[n].mean})), {w:640,h:300})));
  const box = el("div",{class:"grid"});
  for (const lvl of DATA.levels_order)
    box.appendChild(card(lvl, "", chart(names.map(n=>({pts:(DATA.runs[n].per_level[lvl]||[])})),
      {w:180,h:120,pad:28})));
  app.appendChild(box);
}

// --- dev-уровни целиком
app.appendChild(el("h2",{},["Все восемь dev-уровней"]));
app.appendChild(el("p",{class:"note"},["Синий круг — старт, зелёный квадрат — цель, "
  + "оранжевое — кратчайший путь. Агент видит только 5×5 вокруг себя."]));
{
  const box = el("div",{class:"grid"});
  for (const [name, lv] of Object.entries(DATA.dev)) {
    const f = lv.features;
    box.appendChild(card(name,
      "коридор " + f["длина коридора"] + " · развилок " + f["развилок"] + " · путь " + f["путь"],
      maze(lv)));
  }
  app.appendChild(box);
}

// --- буфер учителя
if (DATA.buffer && DATA.buffer.checkpoints.length) {
  app.appendChild(el("h2",{},["Что учитель держит в буфере"]));
  app.appendChild(el("p",{class:"note"},["Взвешено настоящей вероятностью переигрывания "
    + "PLR, то есть это распределение того, что студент видит, а не что просто лежит."]));
  const cps = DATA.buffer.checkpoints;
  const keys = Object.keys(cps[0]["признаки"]);
  const box = el("div",{class:"grid"});
  for (const k of keys)
    box.appendChild(card(k, "апдейт", chart([
      {pts: cps.map(c=>[c.step, c["признаки"][k]["по переигрыванию"]])},
      {pts: cps.map(c=>[c.step, c["признаки"][k]["p90"]]), color:"#888"}], {w:230,h:150,pad:34})));
  app.appendChild(box);
  app.appendChild(el("div",{class:"legend"},["цветное — среднее по переигрыванию, серое — p90"]));

  app.appendChild(el("h2",{},["С чем коррелирует score"]));
  app.appendChild(el("p",{class:"note"},["Score должен находить уровни на границе "
    + "способностей студента. Если он коррелирует со структурным признаком, значит "
    + "меряет сложность уровня, а не обучаемость."]));
  const t = el("table",{},[el("tr",{},["апдейт"].concat(keys).map(h=>el("th",{},[h])))]);
  for (const c of cps)
    t.appendChild(el("tr",{}, [String(c.step)].concat(
      keys.map(k=>fmt(c["признаки"][k]["корреляция со score"]))).map(v=>el("td",{},[v]))));
  app.appendChild(t);
}
</script>
"""


def dev_levels():
    out = {}
    for name in DEV:
        walls, agent, goal = levels.from_prefab(name)
        _, path = levels.shortest_path(walls, agent, goal)
        out[name] = {"walls": walls.astype(int).tolist(), "agent": list(agent),
                     "goal": list(goal), "path": [list(p) for p in path],
                     "features": levels.features(walls, agent, goal)}
    return out


def from_mlflow(uri, names):
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    runs = {}
    for name in names:
        df = mlflow.search_runs(experiment_names=["tlab-ued"],
                                filter_string=f"params.run_name='{name}'")
        if not len(df):
            continue
        rid = df.iloc[0]["run_id"]
        mean = [[m.step, m.value] for m in client.get_metric_history(rid, "solve_rate/mean")]
        per = {lvl: [[m.step, m.value]
                     for m in client.get_metric_history(rid, f"solve_rate/{lvl}")] for lvl in DEV}
        if mean:
            runs[name] = {"mean": mean, "per_level": per}
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["full_accel_s0", "full_plr_s0", "full_dr_s0"])
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--buffer", default="runs/buffer.json")
    p.add_argument("--out", default="runs/report.html")
    args = p.parse_args()

    buf = pathlib.Path(args.buffer)
    data = {"runs": from_mlflow(args.tracking_uri, args.runs),
            "levels_order": DEV,
            "dev": dev_levels(),
            "buffer": json.loads(buf.read_text()) if buf.exists() else None}
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"{len(data['runs'])} прогонов -> {args.out}")


if __name__ == "__main__":
    main()
