"""JollyMountain Predict Lab

A local web app with two parts:
  1. Ask the papers: upload research PDFs about AI predictive systems (scam and
     spam detection) and ask questions. The answer is built from the PDFs and
     shows which file and page it used (Retrieval-Augmented Generation).
  2. Check a message: paste a text message and our own trained model
     (04_train_model.py) predicts how likely it is to be a scam.

Everything runs on this computer. The language model runs through Ollama.

Run:  python 06_app.py   then open http://127.0.0.1:5001
"""
import json
import os
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, redirect, render_template_string, request
from pypdf import PdfReader
from werkzeug.utils import secure_filename

OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gemma3:1b")
VECTOR_MODEL = os.environ.get("VECTOR_MODEL", "nomic-embed-text")

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "library"            # uploaded PDFs live here (not in GitHub)
INDEX_FILE = LIBRARY / "index.json"   # text pieces + their vectors
SCAM_MODEL_FILE = HERE / "scam_model.joblib"

PIECE_CHARS = 900     # target size of one text piece
PIECES_PER_ANSWER = 4
MIN_MATCH = 0.35

INSTRUCTIONS = (
    "You are the JollyMountain research helper for a student project on AI predictive "
    "systems, focused on detecting scam and spam text messages. Answer the question "
    "using only the numbered excerpts. Mention the source in brackets, for example "
    "[paper.pdf p.3]. If the excerpts do not answer the question, reply: "
    "'The uploaded papers do not cover this.' Keep the answer short and clear."
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ----------------------------- index on disk -----------------------------
def load_index():
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {"papers": {}, "pieces": []}


def save_index(index):
    LIBRARY.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")


# ----------------------------- Ollama calls ------------------------------
def to_vectors(texts):
    vectors = []
    for start in range(0, len(texts), 16):
        reply = requests.post(f"{OLLAMA}/api/embed", timeout=240,
                              json={"model": VECTOR_MODEL, "input": texts[start:start + 16]})
        reply.raise_for_status()
        vectors.extend(reply.json()["embeddings"])
    return vectors


def ask_model(system_text, question):
    reply = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
        "model": ANSWER_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system_text},
                     {"role": "user", "content": question}]})
    reply.raise_for_status()
    return reply.json()["message"]["content"]


# ----------------------------- PDF to pieces -----------------------------
def cut_into_pieces(page_text):
    """Group whole sentences into pieces of about PIECE_CHARS characters.
    The last sentence of a piece is repeated at the start of the next one."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(page_text.split())) if s.strip()]
    pieces, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s) for s in current) >= PIECE_CHARS:
            pieces.append(" ".join(current))
            current = [sentence]
    if current and (not pieces or len(current) > 1):
        pieces.append(" ".join(current))
    return [p for p in pieces if len(p) > 60]


def add_paper(pdf_path):
    reader = PdfReader(str(pdf_path))
    found = []
    for number, page in enumerate(reader.pages, start=1):
        for piece in cut_into_pieces(page.extract_text() or ""):
            found.append({"paper": pdf_path.name, "page": number, "text": piece})
    if not found:
        raise ValueError("No text could be read from this PDF (is it a scan?).")
    for piece, vector in zip(found, to_vectors([p["text"] for p in found])):
        piece["vector"] = vector
    index = load_index()
    index["pieces"] = [p for p in index["pieces"] if p["paper"] != pdf_path.name] + found
    index["papers"][pdf_path.name] = {"pages": len(reader.pages), "pieces": len(found)}
    save_index(index)
    return len(found)


def best_pieces(question):
    index = load_index()
    if not index["pieces"]:
        return []
    q = to_vectors([question])[0]
    q_len = sum(x * x for x in q) ** 0.5

    def match(v):
        v_len = sum(x * x for x in v) ** 0.5
        return sum(a * b for a, b in zip(q, v)) / (q_len * v_len) if q_len and v_len else 0.0

    ranked = sorted(((match(p["vector"]), p) for p in index["pieces"]), key=lambda t: -t[0])
    return [(score, p) for score, p in ranked[:PIECES_PER_ANSWER] if score >= MIN_MATCH]


# ------------------------------- routes ----------------------------------
@app.get("/")
def home():
    index = load_index()
    return render_template_string(PAGE, papers=index["papers"], note=request.args.get("note", ""),
                                  model=ANSWER_MODEL, has_scam_model=SCAM_MODEL_FILE.exists())


@app.post("/upload")
def upload():
    file = request.files.get("paper")
    name = secure_filename(file.filename) if file and file.filename else ""
    if not name.lower().endswith(".pdf"):
        return redirect("/?note=Please choose a PDF file.#library")
    LIBRARY.mkdir(exist_ok=True)
    target = LIBRARY / name
    file.save(target)
    try:
        count = add_paper(target)
        note = f"Added {name}: {count} text pieces indexed."
    except requests.RequestException:
        target.unlink(missing_ok=True)
        note = f"Could not reach Ollama. Start Ollama and pull {VECTOR_MODEL}, then try again."
    except Exception as problem:
        target.unlink(missing_ok=True)
        note = f"Could not add {name}: {problem}"
    return redirect(f"/?note={note}#library")


@app.post("/remove")
def remove():
    name = secure_filename(request.form.get("name", ""))
    index = load_index()
    index["papers"].pop(name, None)
    index["pieces"] = [p for p in index["pieces"] if p["paper"] != name]
    save_index(index)
    (LIBRARY / name).unlink(missing_ok=True)
    return redirect(f"/?note=Removed {name}.#library")


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify(error="Type a question first."), 400
    try:
        if body.get("use_papers", True):
            hits = best_pieces(question)
            excerpts = "\n\n".join(f"({i}) [{p['paper']} p.{p['page']}] {p['text']}"
                                   for i, (_, p) in enumerate(hits, start=1)) or "(no matching excerpts)"
            answer = ask_model(INSTRUCTIONS + "\n\nEXCERPTS:\n" + excerpts, question)
            sources = [{"paper": p["paper"], "page": p["page"], "match": round(s, 2)} for s, p in hits]
        else:
            answer = ask_model("You are a helpful assistant. Keep the answer short and clear.", question)
            sources = []
    except requests.RequestException:
        return jsonify(error=f"Could not reach Ollama. Is it running with {ANSWER_MODEL} and {VECTOR_MODEL}?"), 502
    return jsonify(answer=answer, sources=sources)


@app.post("/check")
def check():
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()
    if not text:
        return jsonify(error="Paste a message first."), 400
    if not SCAM_MODEL_FILE.exists():
        return jsonify(error="No trained model yet. Run: python 04_train_model.py"), 400
    import joblib  # only needed for this feature
    chance = float(joblib.load(SCAM_MODEL_FILE).predict_proba([text])[0][1])
    return jsonify(scam_probability=round(chance, 3),
                   verdict="Likely scam" if chance >= 0.5 else "Looks legitimate")


# ------------------------------- the page --------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JollyMountain Predict Lab</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f3f6fa;color:#1c2733}
 header{background:#12355b;color:#fff;padding:18px 24px}
 header h1{margin:0;font-size:1.3rem} header p{margin:4px 0 0;opacity:.8;font-size:.9rem}
 main{max-width:860px;margin:0 auto;padding:20px 16px;display:grid;gap:18px}
 section{background:#fff;border:1px solid #d9e1ea;border-radius:10px;padding:18px}
 h2{margin:0 0 10px;font-size:1.1rem;color:#12355b}
 textarea{width:100%;box-sizing:border-box;padding:10px;border:1px solid #c5d0dc;border-radius:8px;font:inherit}
 button{background:#1f6fb2;color:#fff;border:0;border-radius:8px;padding:8px 16px;font:inherit;cursor:pointer;margin-top:8px}
 button.plain{background:#fff;color:#b23b1f;border:1px solid #b23b1f;padding:3px 10px;margin:0}
 .out{white-space:pre-wrap;background:#f3f6fa;border-radius:8px;padding:12px;margin-top:12px;min-height:20px}
 .src{font-size:.85rem;color:#566575;margin-top:6px}
 .note{background:#e7f3e8;border:1px solid #b9dcbc;padding:10px 14px;border-radius:8px}
 table{width:100%;border-collapse:collapse;margin-top:10px} td,th{text-align:left;padding:6px;border-bottom:1px solid #e3e9f0;font-size:.92rem}
 label{font-size:.9rem;color:#566575}
</style></head><body>
<header><h1>JollyMountain Predict Lab</h1>
<p>AI predictive systems: scam text detection, with answers from our own research papers ({{ model }} via Ollama)</p></header>
<main>
{% if note %}<div class="note">{{ note }}</div>{% endif %}

<section id="ask"><h2>1. Ask the papers</h2>
<textarea id="q" rows="2" placeholder="Example: Which features help a model detect spam text messages?"></textarea>
<label><input type="checkbox" id="use" checked> Use the uploaded papers</label><br>
<button onclick="ask()">Ask</button>
<div class="out" id="answer"></div><div class="src" id="sources"></div></section>

<section id="scam"><h2>2. Check a message with our trained model</h2>
{% if not has_scam_model %}<p class="src">Train the model first: <code>python 04_train_model.py</code></p>{% endif %}
<textarea id="msg" rows="2" placeholder="Paste a text message here"></textarea>
<button onclick="checkMsg()">Check</button>
<div class="out" id="verdict"></div></section>

<section id="library"><h2>3. Paper library</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
<input type="file" name="paper" accept="application/pdf" required> <button>Upload PDF</button></form>
{% if papers %}<table><tr><th>Paper</th><th>Pages</th><th>Text pieces</th><th></th></tr>
{% for name, info in papers.items() %}<tr><td>{{ name }}</td><td>{{ info.pages }}</td><td>{{ info.pieces }}</td>
<td><form action="/remove" method="post"><input type="hidden" name="name" value="{{ name }}"><button class="plain">Remove</button></form></td></tr>
{% endfor %}</table>{% else %}<p class="src">No papers uploaded yet.</p>{% endif %}</section>
</main>
<script>
async function post(url, data){
  const r = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  const j = await r.json(); if(!r.ok) throw new Error(j.error||"Something went wrong"); return j;
}
async function ask(){
  const a=document.getElementById("answer"), s=document.getElementById("sources");
  a.textContent="Thinking..."; s.textContent="";
  try{
    const j=await post("/ask",{question:document.getElementById("q").value,use_papers:document.getElementById("use").checked});
    a.textContent=j.answer;
    if(j.sources.length) s.textContent="Sources: "+j.sources.map(x=>x.paper+" p."+x.page).join(", ");
  }catch(e){a.textContent=e.message}
}
async function checkMsg(){
  const v=document.getElementById("verdict"); v.textContent="Checking...";
  try{
    const j=await post("/check",{text:document.getElementById("msg").value});
    v.textContent=j.verdict+" (scam probability: "+Math.round(j.scam_probability*100)+"%)";
  }catch(e){v.textContent=e.message}
}
</script></body></html>"""

if __name__ == "__main__":
    LIBRARY.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5001, debug=False)
