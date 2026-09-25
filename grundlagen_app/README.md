# Mathe 1 – Übungsaufgaben

Streamlit-App mit Aufgaben zu Grundlagen (Brüche, Dreisatz, Prozent, Logarithmus, Potenzen/Wurzeln, Exponentialgleichungen).

Die 500 Aufgaben und ihre kleinschrittigen Lösungen liegen in **`data/aufgaben.json`**
(LaTeX in Strings). Die App rendert sie als Markdown mit LaTeX.

## Logo

- `logo.png` – Favicon / Browser-Tab (wird in der App eingebunden)
- `logo.svg` – Vektorversion für Webseiten, Folien oder Verlinkungen

## Lokal starten (PowerShell, Repo-Root)

```powershell
pip install -r .\grundlagen_app\requirements.txt
streamlit run .\grundlagen_app\app.py
```

## Streamlit Cloud

- Main file: `grundlagen_app/app.py`
- Requirements: `grundlagen_app/requirements.txt`

## Datenbank erweitern

Neue Einträge in `tasks` mit:

- `id` (eindeutig)
- `topic_id` (muss zu `topics` passen)
- `aufgabe` (LaTeX, z. B. `\\[ ... \\]` oder Fließtext mit `\\(...\\)`)
- `loesung_schritte` (Liste von Strings, je ein Lösungsschritt)
- `loesung_kurz` (Endergebnis)

Aufgaben reproduzierbar neu erzeugen (genau 500 Stück):

```powershell
py .\grundlagen_app\generate_aufgaben.py
```

Der Generator prüft Anzahl, Themenverteilung, fortlaufende IDs, Duplikate und Nullnenner,
bevor er die JSON-Datei schreibt.

Orientierung am Skript `01_grundlagen.tex` (Notation: „größter gemeinsamer Teiler“,
„kleinstes gemeinsames Vielfaches“ statt `\gcd` / `\kgV`).

**Regeln pro Aufgabentyp** (Lösungsschritte, LaTeX, Bruch-Erweitern mit `\cdot`, Add/Sub mit separatem Kürzen, …): siehe **[AUFGABEN_REGELN.md](./AUFGABEN_REGELN.md)**.
