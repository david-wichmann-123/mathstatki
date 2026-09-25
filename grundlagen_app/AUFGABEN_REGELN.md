# Regeln für Aufgaben und Lösungen (`grundlagen_app`)

Referenz für manuelle Bearbeitung von `data/aufgaben.json` und für `generate_aufgaben.py`.  
Orientierung: Skript `01_grundlagen.tex` (Mathe 1 WNB). Keine Aufgaben zu Zahlenmengen / Grundrechenarten.

Bei Änderungen an den Regeln: Generator anpassen, dann `py .\grundlagen_app\generate_aufgaben.py` ausführen (oder JSON manuell konsistent halten).

---

## Allgemein

| Thema | Regel |
|--------|--------|
| LaTeX in JSON | Fließtext: `\(...\)`, größere Ausdrücke: `\[...\]` |
| Brüche in Display | Ganzen Ausdruck in **einer** `\[...\]`-Zeile, z. B. `\[\frac{a}{b}+\frac{c}{d}=...\]` |
| Doppelbrüche | `\dfrac{...}{...}` |
| ggT / kgV | Deutsch ausgeschrieben: „größter gemeinsamer Teiler“, „kleinstes gemeinsames Vielfaches“ — **kein** `\gcd`, `\kgV` |
| `loesung_kurz` | Nur das Endergebnis (meist ein Display-Block) |
| `loesung_schritte` | Didaktische Zwischenschritte, je String ein Schritt in der App |
| Aufgaben-IDs | Generator vergibt `aufg0001` … `aufg0500`; `nummer` 1–500 |
| Verständlichkeit | Jede Umformung wird benannt und in einem eigenen, nachvollziehbaren Schritt gezeigt |
| Eindeutigkeit | Keine identischen Kombinationen aus Aufgabe und Kurzlösung |
| Ganze Ergebnisse | Ganze Zahlen als `4`, nicht als `\frac{4}{1}` ausgeben |

---

## Themen (`topic_id`)

- `rechenregeln` — Klammern, Punkt vor Strich  
- `brueche_kuerzen` — Kürzen **und** Erweitern (Erweiterungsaufgaben gehören hierher)  
- `brueche_add_sub` — Addition / Subtraktion  
- `brueche_mul_div` — Multiplikation / Division (Kehrwert bei Division)  
- `brueche_doppel` — Doppelbrüche  
- `potenzen_wurzel`, `logarithmus`, `prozent`, `dreisatz`, `gleichungen`, `ungleichungen`

---

## Bruch **kürzen**

1. ggT von Zähler und Nenner nennen.  
2. Teilen: `\[\frac{a}{b}=\frac{a'}{b'}\]`

---

## Bruch **erweitern** (Nenner vorgegeben)

1. **Faktor:** Nenner-Ziel geteilt durch Nenner, z. B. `\(100:4=25\)`.  
2. **Erweitern** mit sichtbarem Faktor in Zähler **und** Nenner:  
   `\[\frac{n}{d}=\frac{n\cdot f}{d\cdot f}=\frac{n'}{d'}\]`  
   (Faktor `f` explizit mit `\cdot`; nicht nur Sprung von `\frac{7}{4}` zu `\frac{175}{100}`.)

---

## Bruch **Addition / Subtraktion**

Reihenfolge der Schritte:

1. **kgV** der Nenner angeben.  
2. **Erweitern** beider Brüche auf den Hauptnenner `k`:  
   - Wenn Erweiterungsfaktor `> 1`: Kette  
     `\frac{n}{d}=\frac{n\cdot m}{d\cdot m}=\frac{n'}{k}`  
   - Wenn Faktor `1`: `\frac{n}{d}=\frac{n}{k}` (Bruch bereits auf `k`).  
3. **Addieren / Subtrahieren** nur mit **gleichem Nenner** `k` — Ergebnis **noch nicht kürzen**:  
   `\[\frac{n_1'}{k}\pm\frac{n_2'}{k}=\frac{N}{k}\]`  
   (`N` = ungekürzter Zähler.)  
4. **Kürzen** (eigener Schritt, nur wenn ggT(`|N|`, `k`) > 1):  
   ggT nennen, dann `\[\frac{N}{k}=\frac{N'}{k'}\]`.  
   Wenn bereits vollständig gekürzt: Schritt 4 entfällt.
5. Negative Ergebnisse sprachlich erklären. Bei einem unechten Ergebnisbruch darauf hinweisen, dass der Zähler größer als der Nenner ist.

Skript-Beispiel `2/7 + 5/12`: kgV 84 → Erweitern → Summe `59/84` (kein weiteres Kürzen nötig).

---

## Bruch **Multiplikation / Division**

1. Rechenregel / Kehrwert (Division).  
2. Zähler · Zähler, Nenner · Nenner (explizit `\cdot` in Bruchschreibweise).  
3. **Kürzen** als eigener Schritt zum Endbruch (ggT erwähnen, wenn sinnvoll).
   Ist der Zwischenbruch bereits vollständig gekürzt, entfällt der Kürzschritt.

---

## Doppelbrüche

Aufgaben: einfache Doppelbrüche **und** solche, bei denen **Zähler und/oder Nenner** selbst eine **Summe, Differenz oder ein Produkt** von Brüchen sind (in `\dfrac{...}{...}` mit `\dfrac` für die Einzelbrüche).

Lösungsschritte:

1. Falls Zähler/Nenner zusammengesetzt: zuerst **Zähler** bzw. **Nenner** vereinfachen (Add/Sub/Mul wie oben; bei Summe/Differenz kgV, Erweitern, Rechnung, ggf. Kürzen).  
2. Division durch Bruch = Multiplikation mit dem Kehrwert.  
3. **Kürzen** des Ergebnisses als eigener Schritt, wenn möglich.

---

## Rechenregeln

- Klammeraufgabe: zuerst Klammer, dann äußere Operation.  
- Punkt vor Strich: zuerst Multiplikation, dann Addition/Subtraktion.

---

## Potenzen / Wurzeln

- Gleiche Basis: beim **Multiplizieren** Exponenten addieren, beim **Dividieren** Exponenten subtrahieren. Die konkrete Exponentenrechnung als eigenen Schritt zeigen.  
- Wurzeln: Faktorzerlegung, dann `\sqrt{a^2\cdot rest}=a\sqrt{rest}`.  
- Potenz einer Potenz: Exponenten multiplizieren.

---

## Logarithmus

Zuerst die Definition `\log_b(x)=e \Leftrightarrow b^e=x` nennen, dann die passende Potenz suchen.

---

## Prozent, Dreisatz, Gleichungen, Ungleichungen

- Prozent: „von Hundert“ erklären, dann `\frac{p}{100}` mal Grundwert. Die Aufgaben verwenden ganzzahlige Prozentwerte.  
- Dreisatz: exakten Einheitspreis bestimmen, dann skalieren (Euro/kg); keine versteckte Zwischenrundung.  
- Lineare Gleichung: zuerst den konstanten Term entfernen, anschließend die sichtbare Division zum Isolieren von `x`; Exponentialgleichung: gleiche Basis herstellen, Exponenten vergleichen und gegebenenfalls dividieren.  
- Ungleichung: jeden Umformungsschritt nennen; exakte Brüche statt langer Dezimalzahlen. Beim Teilen durch die hier positive Zahl ausdrücklich erklären, warum das Zeichen gleich bleibt.

---

## Generator

- Logik und Schritt-Templates: `generate_aufgaben.py`  
- Seed `42` für eine reproduzierbare Aufgabenmenge von **genau 500 Aufgaben**  
- Der Generator verhindert identische Aufgaben und validiert Anzahl, Themenverteilung, IDs sowie Nullnenner vor dem Schreiben.  
- Soll-Verteilung: Rechenregeln 70; Kürzen/Erweitern 55; Add/Sub 50; Mul/Div 40; Doppelbrüche 35; Potenzen/Wurzeln 55; Logarithmus 25; Prozent 30; Dreisatz 30; Gleichungen 65; Ungleichungen 45.  
- Nach Regeländerungen immer JSON neu generieren und in der App alle Themen sowie Sonderfälle stichprobenartig prüfen.
