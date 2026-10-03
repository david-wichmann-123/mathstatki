# Lineare Gleichungssysteme (LGS)

Streamlit-Übungsapp mit **70 Aufgaben** zu linearen Gleichungssystemen (1–4 Unbekannte).

## Verteilung

| Unbekannte | Anzahl Aufgaben |
|------------|-----------------|
| 1          | 10              |
| 2          | 15              |
| 3          | 25              |
| 4          | 20              |

In jeder Kategorie sind zwei Aufgaben quadratisch und homogen: eine mit nur der trivialen Lösung und eine mit unendlich vielen Lösungen.

Mix aus **unterbestimmten**, **quadratischen** und **überbestimmten** Systemen sowie Lösungstypen: eindeutige Lösung, keine Lösung, unendlich viele Lösungen. Notation, Gaußverfahren und ausgewählte Beispielsysteme orientieren sich an Kapitel 03 des Mathe-1-Skripts.

Die Gleichungen werden wie im Skript untereinander ausgerichtet und ohne eine linke Systemklammer dargestellt. Die Lösungen zeigen die elementaren Zeilenoperationen, die jeweilige Zeilenstufenform, das Rückwärtseinsetzen beziehungsweise eine Parameterdarstellung.

## Lokal starten

```bash
pip install -r lgs_app/requirements.txt
streamlit run lgs_app/app.py
```

## Aufgaben neu erzeugen

Nach Anpassungen am Generator:

```bash
python lgs_app/generate_aufgaben.py
```

Die App liest `lgs_app/data/aufgaben.json` (für Streamlit Cloud ins Repo committen).

## Streamlit Cloud

- **Main file path:** `lgs_app/app.py`
- **Working directory:** Repo-Root (wie bei den anderen Apps)
