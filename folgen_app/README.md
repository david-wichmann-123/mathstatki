# Folgen und Reihen

Streamlit-App zum Definieren, Tabellieren und Plotten von Folgen und Partialsummen (immer die ersten 100 Glieder, Start bei \(n=1\)).

## Logo

- `logo.png` – Favicon / Browser-Tab (wird in der App eingebunden)
- `logo.svg` – Vektorversion für Webseiten, Folien oder Verlinkungen (z. B. als `<img src=".../logo.svg">`)

Farbe: Hochschule Esslingen `#002C5C`, Motiv: diskrete Folge (Punkte) und **n**.

## Starten (Windows / PowerShell)

Im Repository-Hauptordner:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\folgen_app\requirements.txt
streamlit run .\folgen_app\app.py
```

Danach öffnet Streamlit die App normalerweise automatisch im Browser. Falls PowerShell die Aktivierung blockiert:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Anschließend die Aktivierung wiederholen.

## Eingabe

- **Explizit:** Formel in \(n\), z. B. `0.5**n`, `1/n`, `(-1)**n/n`
- **Rekursiv:** Startwerte \(a_1\) (und ggf. \(a_2\)) plus Rekursion; Buttons fügen `a_{n-1}` und `a_{n-2}` ein (intern als `a_nm1`, `a_nm2`)

Die Graphen lassen sich mit dem Mausrad zoomen und durch Ziehen verschieben (Plotly-Toolbar).
