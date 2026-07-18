# Differenzialrechnung – Grundlagen

Eine kleine Streamlit-App zum Eingeben und Visualisieren von Funktionen.

Der Graph laesst sich mit dem Mausrad zoomen und durch Ziehen verschieben. Nach
jeder Aenderung des sichtbaren x-Bereichs wird die Kurve neu abgetastet und die
y-Achse passend skaliert.

## Starten (Windows / PowerShell)

Im Repository-Hauptordner:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\differentialrechnung_app\requirements.txt
streamlit run .\differentialrechnung_app\app.py
```

Danach öffnet Streamlit die App normalerweise automatisch im Browser. Falls PowerShell die Aktivierung blockiert, führe einmal in derselben PowerShell aus:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

und wiederhole dann die Aktivierung.

## Unterstützte Eingaben

- Potenzen: `x^2`, `x^(1/2)`
- Exponentialfunktion: `exp(x)`, `e^x`
- Logarithmen: `ln(x)`, `log(x)`
- Trigonometrie: `sin(x)`, `cos(x)`, `tan(x)`
- Wurzel: `sqrt(x)`
- Implizite Multiplikation: `2x + 1`
