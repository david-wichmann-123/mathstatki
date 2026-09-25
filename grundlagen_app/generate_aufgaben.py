"""Erzeugt genau 500 Aufgaben im Stil des Skripts 01_grundlagen.tex.

Schrittfolgen und Notation: siehe AUFGABEN_REGELN.md im gleichen Ordner.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from fractions import Fraction
from math import gcd
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "data" / "aufgaben.json"
RNG = random.Random(42)
SEEN_TASKS: set[tuple[str, str]] = set()


def kgv(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def inline(value: str) -> str:
    return rf"\({value}\)"


def display(body: str) -> str:
    return rf"\[{body}\]"


def frac_tex(n: int, d: int) -> str:
    if n < 0:
        return rf"-\frac{{{abs(n)}}}{{{d}}}"
    return rf"\frac{{{n}}}{{{d}}}"


def number_tex(value: Fraction | int) -> str:
    """Ganze Zahlen ohne /1, sonst einen vollständig gekürzten Bruch ausgeben."""
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    if fraction.numerator < 0:
        return rf"-\frac{{{abs(fraction.numerator)}}}{{{fraction.denominator}}}"
    return frac_tex(fraction.numerator, fraction.denominator)


def decimal_comma(cents: int) -> str:
    return f"{cents // 100},{cents % 100:02d}"


def frac_expand_chain(n: int, d: int, factor: int, target_d: int) -> str:
    """Erweitern mit explizitem \\cdot-Faktor (siehe AUFGABEN_REGELN.md)."""
    if factor == 1:
        return f"{frac_tex(n, d)}={frac_tex(n, target_d)}"
    return (
        f"{frac_tex(n, d)}="
        f"\\frac{{{n}\\cdot {factor}}}{{{d}\\cdot {factor}}}="
        f"{frac_tex(n * factor, target_d)}"
    )


def dfrac_tex(n: int, d: int) -> str:
    return rf"\dfrac{{{n}}}{{{d}}}"


def random_proper_frac() -> tuple[int, int]:
    for _ in range(40):
        d = RNG.randint(2, 9)
        n = RNG.randint(1, d - 1)
        if gcd(n, d) == 1:
            return n, d
    d = RNG.randint(3, 9)
    return 1, d


def combine_op_symbol(op: str) -> str:
    return {"+": "+", "-": "-", "mul": r"\cdot"}[op]


def combine_frac_steps(part_label: str, op: str, n1: int, d1: int, n2: int, d2: int) -> tuple[int, int, list[str]]:
    """Zähler/Nenner aus Summe, Differenz oder Produkt zweier Brüche (Lösungsschritte)."""
    sym = combine_op_symbol(op)
    steps: list[str] = []

    if op == "mul":
        num_u, den_u = n1 * n2, d1 * d2
        g = gcd(num_u, den_u)
        steps.append(
            f"{part_label}: Zähler mal Zähler und Nenner mal Nenner:"
        )
        steps.append(
            display(
                f"{dfrac_tex(n1, d1)}{sym}{dfrac_tex(n2, d2)}="
                f"\\frac{{{n1}\\cdot {n2}}}{{{d1}\\cdot {d2}}}="
                f"{frac_tex(num_u, den_u)}"
            )
        )
        if g > 1:
            steps.append(
                f"Der größte gemeinsame Teiler von {inline(str(num_u))} und "
                f"{inline(str(den_u))} ist {inline(str(g))}."
            )
            steps.append(
                f"Kürzen: {display(f'{frac_tex(num_u, den_u)}={number_tex(Fraction(num_u, den_u))}')}"
            )
        return num_u // g, den_u // g, steps

    k = kgv(d1, d2)
    m1, m2 = k // d1, k // d2
    num_raw = n1 * m1 + n2 * m2 if op == "+" else n1 * m1 - n2 * m2
    g = gcd(abs(num_raw), k)
    num_f, den_f = num_raw // g, k // g
    expand_body = f"{frac_expand_chain(n1, d1, m1, k)},\\quad {frac_expand_chain(n2, d2, m2, k)}"
    op_label = "Addieren" if op == "+" else "Subtrahieren"
    steps.append(
        f"{part_label} — das kleinste gemeinsame Vielfache der Nenner "
        f"{inline(str(d1))} und {inline(str(d2))} ist {inline(str(k))}."
    )
    steps.append(f"Erweitern: {display(expand_body)}")
    steps.append(
        f"{op_label}: {display(f'{frac_tex(n1 * m1, k)}{sym}{frac_tex(n2 * m2, k)}={frac_tex(num_raw, k)}')}"
    )
    if g > 1:
        steps.append(
            f"Der größte gemeinsame Teiler von {inline(str(abs(num_raw)))} und {inline(str(k))} ist {inline(str(g))}."
        )
        steps.append(f"Kürzen: {display(f'{frac_tex(num_raw, k)}={frac_tex(num_f, den_f)}')}")
    elif num_raw < 0:
        steps.append(
            "Das Ergebnis ist negativ, weil bei der Subtraktion der zweite Bruch größer war."
        )
    return num_f, den_f, steps


def inner_frac_tex(op: str | None, n1: int, d1: int, n2: int | None = None, d2: int | None = None) -> str:
    if op is None:
        return dfrac_tex(n1, d1)
    assert n2 is not None and d2 is not None
    sym = combine_op_symbol(op)
    return f"{dfrac_tex(n1, d1)}{sym}{dfrac_tex(n2, d2)}"


def doppelbruch_display(num_inner: str, den_inner: str) -> str:
    return rf"\dfrac{{{num_inner}}}{{{den_inner}}}"


def simplify_doppelbruch(n_top: int, d_top: int, n_bot: int, d_bot: int) -> tuple[int, int, int]:
    num_u = n_top * d_bot
    den_u = d_top * n_bot
    g = gcd(num_u, den_u)
    return num_u // g, den_u // g, g


def append_task(
    tasks: list, topic_id: str, aufgabe: str, steps: list[str], kurz: str
) -> bool:
    """Aufgabe nur einmal aufnehmen; alle Generatoren würfeln bei Duplikaten neu."""
    signature = (aufgabe.strip(), kurz.strip())
    if signature in SEEN_TASKS:
        return False
    if not steps or any(not step.strip() for step in steps):
        raise ValueError(f"Leerer Lösungsschritt bei {aufgabe!r}")
    SEEN_TASKS.add(signature)
    tasks.append(
        {
            "id": "pending",
            "topic_id": topic_id,
            "aufgabe": aufgabe,
            "loesung_schritte": steps,
            "loesung_kurz": kurz,
        }
    )
    return True


def gen_rechenregeln(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        kind = RNG.choice(["klammer_plus", "klammer_minus", "punkt_plus", "punkt_minus"])
        if kind == "klammer_plus":
            a, b, c = RNG.randint(2, 18), RNG.randint(2, 15), RNG.randint(2, 9)
            inner, val = a + b, (a + b) * c
            aufgabe = f"Berechne:\n{display(f'({a}+{b})\\cdot {c}')}"
            steps = [
                f"Klammern werden zuerst berechnet: {display(f'{a}+{b}={inner}')}",
                f"Danach multiplizieren: {display(f'{inner}\\cdot {c}={val}')}",
            ]
        elif kind == "klammer_minus":
            b = RNG.randint(2, 12)
            a = RNG.randint(b + 1, b + 15)
            c = RNG.randint(2, 9)
            inner, val = a - b, (a - b) * c
            aufgabe = f"Berechne:\n{display(f'({a}-{b})\\cdot {c}')}"
            steps = [
                f"Klammern werden zuerst berechnet: {display(f'{a}-{b}={inner}')}",
                f"Danach multiplizieren: {display(f'{inner}\\cdot {c}={val}')}",
            ]
        else:
            b, c = RNG.randint(2, 10), RNG.randint(2, 9)
            product = b * c
            if kind == "punkt_plus":
                a = RNG.randint(2, 30)
                val = a + product
                expression = f"{a}+{b}\\cdot {c}"
                final = f"{a}+{product}={val}"
                operation = "addieren"
            else:
                a = RNG.randint(product, product + 30)
                val = a - product
                expression = f"{a}-{b}\\cdot {c}"
                final = f"{a}-{product}={val}"
                operation = "subtrahieren"
            aufgabe = f"Berechne:\n{display(expression)}"
            steps = [
                "Punktrechnung kommt vor Strichrechnung.",
                f"Zuerst multiplizieren: {display(f'{b}\\cdot {c}={product}')}",
                f"Danach {operation}: {display(final)}",
            ]
        if append_task(tasks, "rechenregeln", aufgabe, steps, display(str(val))):
            created += 1


def gen_brueche_kuerzen(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        g = RNG.randint(2, 12)
        a = RNG.randint(1, 18) * g
        b = RNG.randint(2, 18) * g
        ggt = gcd(a, b)
        na, nb = a // ggt, b // ggt
        aufgabe = f"Kürze den Bruch soweit wie möglich:\n{display(frac_tex(a, b))}"
        steps = [
            "Zum vollständigen Kürzen suchen wir den größten gemeinsamen Teiler von Zähler und Nenner.",
            f"Der größte gemeinsame Teiler von {inline(str(a))} und {inline(str(b))} ist {inline(str(ggt))}.",
            f"Zähler und Nenner durch {inline(str(ggt))} teilen:",
            display(
                f"{frac_tex(a, b)}="
                f"\\frac{{{a}:{ggt}}}{{{b}:{ggt}}}={number_tex(Fraction(na, nb))}"
            ),
        ]
        if append_task(
            tasks, "brueche_kuerzen", aufgabe, steps, display(number_tex(Fraction(na, nb)))
        ):
            created += 1


def gen_brueche_erweitern(tasks: list, count: int) -> None:
    targets = [12, 20, 24, 30, 40, 50, 60, 100]
    created = 0
    while created < count:
        n, d = random_proper_frac()
        valid = [target for target in targets if target % d == 0 and target != d]
        if not valid:
            continue
        target = RNG.choice(valid)
        factor = target // d
        nn = n * factor
        aufgabe = (
            f"Erweitere {inline(frac_tex(n, d))}, sodass der Nenner {inline(str(target))} ist."
        )
        steps = [
            "Beim Erweitern werden Zähler und Nenner mit derselben Zahl multipliziert.",
            f"Erweiterungsfaktor bestimmen: {display(f'{target}:{d}={factor}')}",
            f"Zähler und Nenner mit {inline(str(factor))} multiplizieren:",
            display(frac_expand_chain(n, d, factor, target)),
        ]
        if append_task(
            tasks, "brueche_kuerzen", aufgabe, steps, display(frac_tex(nn, target))
        ):
            created += 1


def gen_brueche_add_sub(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        op = RNG.choice(["+", "-"])
        n1, d1 = random_proper_frac()
        n2, d2 = random_proper_frac()
        k = kgv(d1, d2)
        m1, m2 = k // d1, k // d2
        num_raw = n1 * m1 + n2 * m2 if op == "+" else n1 * m1 - n2 * m2
        g = gcd(abs(num_raw), k)
        num, den = num_raw // g, k // g
        aufgabe = f"Berechne:\n{display(f'{frac_tex(n1, d1)}{op}{frac_tex(n2, d2)}')}"
        op_label = "Addieren" if op == "+" else "Subtrahieren"
        expand_body = (
            f"{frac_expand_chain(n1, d1, m1, k)},\\quad {frac_expand_chain(n2, d2, m2, k)}"
        )
        steps = [
            "Brüche können nur mit gleichem Nenner addiert oder subtrahiert werden.",
            f"Das kleinste gemeinsame Vielfache der Nenner {inline(str(d1))} und {inline(str(d2))} ist {inline(str(k))}.",
            f"Beide Brüche auf den Hauptnenner {inline(str(k))} erweitern:",
            display(expand_body),
            f"Jetzt die Zähler {op_label.lower()}, der Nenner bleibt stehen:",
            display(
                f"{frac_tex(n1 * m1, k)}{op}{frac_tex(n2 * m2, k)}="
                f"\\frac{{{n1 * m1}{op}{n2 * m2}}}{{{k}}}={frac_tex(num_raw, k)}"
            ),
        ]
        if num_raw < 0:
            steps.append(
                "Das Ergebnis ist negativ, weil der abgezogene Bruch größer als der erste Bruch ist."
            )
        if g > 1:
            steps.append(
                f"Der größte gemeinsame Teiler von {inline(str(abs(num_raw)))} und {inline(str(k))} ist {inline(str(g))}."
            )
            steps.append(
                f"Zum Schluss kürzen: {display(f'{frac_tex(num_raw, k)}={number_tex(Fraction(num, den))}')}"
            )
        if abs(num) > den:
            steps.append("Das Ergebnis ist ein unechter Bruch; der Zähler ist größer als der Nenner.")
        if append_task(
            tasks, "brueche_add_sub", aufgabe, steps, display(number_tex(Fraction(num, den)))
        ):
            created += 1


def gen_brueche_mul_div(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        op = RNG.choice(["mul", "div"])
        n1, d1 = random_proper_frac()
        n2, d2 = random_proper_frac()
        if op == "mul":
            num_raw, den_raw = n1 * n2, d1 * d2
            sym = r"\cdot"
            steps = [
                "Beim Multiplizieren gilt: Zähler mal Zähler und Nenner mal Nenner.",
                display(
                    f"{frac_tex(n1, d1)}{sym}{frac_tex(n2, d2)}="
                    f"\\frac{{{n1}\\cdot {n2}}}{{{d1}\\cdot {d2}}}="
                    f"{frac_tex(num_raw, den_raw)}"
                ),
            ]
        else:
            num_raw, den_raw = n1 * d2, d1 * n2
            steps = [
                "Durch einen Bruch teilen heißt mit seinem Kehrwert multiplizieren.",
                f"Der Kehrwert von {inline(frac_tex(n2, d2))} ist {inline(frac_tex(d2, n2))}.",
                display(
                    f"{frac_tex(n1, d1)}:{frac_tex(n2, d2)}="
                    f"{frac_tex(n1, d1)}\\cdot{frac_tex(d2, n2)}="
                    f"\\frac{{{n1}\\cdot {d2}}}{{{d1}\\cdot {n2}}}="
                    f"{frac_tex(num_raw, den_raw)}"
                ),
            ]
        g = gcd(num_raw, den_raw)
        result = Fraction(num_raw, den_raw)
        op_sym = r"\cdot" if op == "mul" else ":"
        aufgabe = f"Berechne:\n{display(f'{frac_tex(n1, d1)}{op_sym}{frac_tex(n2, d2)}')}"
        if g > 1:
            steps.extend(
                [
                    f"Der größte gemeinsame Teiler von {inline(str(num_raw))} und "
                    f"{inline(str(den_raw))} ist {inline(str(g))}.",
                    f"Zum Schluss kürzen: "
                    f"{display(f'{frac_tex(num_raw, den_raw)}={number_tex(result)}')}",
                ]
            )
        if append_task(
            tasks, "brueche_mul_div", aufgabe, steps, display(number_tex(result))
        ):
            created += 1


def _random_compound_pair() -> tuple[str, int, int, int, int] | None:
    op = RNG.choice(["+", "-", "mul"])
    n1, d1 = random_proper_frac()
    n2, d2 = random_proper_frac()
    if op == "-" and n1 * d2 <= n2 * d1:
        return None
    return op, n1, d1, n2, d2


def gen_brueche_doppel(tasks: list, count: int) -> None:
    variants = ["simple", "num", "den", "both"]
    created = 0
    attempts = 0
    while created < count and attempts < count * 80:
        attempts += 1
        variant = RNG.choice(variants)
        steps: list[str] = []

        num_inner: str
        den_inner: str
        n_top: int
        d_top: int
        n_bot: int
        d_bot: int

        if variant == "simple":
            n_top, d_top = random_proper_frac()
            n_bot, d_bot = random_proper_frac()
            num_inner = inner_frac_tex(None, n_top, d_top)
            den_inner = inner_frac_tex(None, n_bot, d_bot)
        elif variant == "num":
            compound = _random_compound_pair()
            if compound is None:
                continue
            op, n1, d1, n2, d2 = compound
            n_bot, d_bot = random_proper_frac()
            num_inner = inner_frac_tex(op, n1, d1, n2, d2)
            den_inner = inner_frac_tex(None, n_bot, d_bot)
            n_top, d_top, part_steps = combine_frac_steps("Zähler", op, n1, d1, n2, d2)
            steps.extend(part_steps)
        elif variant == "den":
            compound = _random_compound_pair()
            if compound is None:
                continue
            op, n1, d1, n2, d2 = compound
            n_top, d_top = random_proper_frac()
            num_inner = inner_frac_tex(None, n_top, d_top)
            den_inner = inner_frac_tex(op, n1, d1, n2, d2)
            n_bot, d_bot, part_steps = combine_frac_steps("Nenner", op, n1, d1, n2, d2)
            steps.extend(part_steps)
        else:
            compound_num = _random_compound_pair()
            compound_den = _random_compound_pair()
            if compound_num is None or compound_den is None:
                continue
            op_n, n1, d1, n2, d2 = compound_num
            op_d, n3, d3, n4, d4 = compound_den
            num_inner = inner_frac_tex(op_n, n1, d1, n2, d2)
            den_inner = inner_frac_tex(op_d, n3, d3, n4, d4)
            n_top, d_top, steps_num = combine_frac_steps("Zähler", op_n, n1, d1, n2, d2)
            steps.extend(steps_num)
            n_bot, d_bot, steps_den = combine_frac_steps("Nenner", op_d, n3, d3, n4, d4)
            steps.extend(steps_den)

        num_u = n_top * d_bot
        den_u = d_top * n_bot
        num, den, g_final = simplify_doppelbruch(n_top, d_top, n_bot, d_bot)

        steps.append("Division durch einen Bruch = Multiplikation mit dem Kehrwert:")
        if g_final > 1:
            steps.append(
                display(
                    f"{frac_tex(n_top, d_top)}:{frac_tex(n_bot, d_bot)}="
                    f"{frac_tex(n_top, d_top)}\\cdot{frac_tex(d_bot, n_bot)}="
                    f"{frac_tex(num_u, den_u)}"
                )
            )
            steps.append(
                f"Der größte gemeinsame Teiler von {inline(str(num_u))} und "
                f"{inline(str(den_u))} ist {inline(str(g_final))}."
            )
            steps.append(
                f"Zum Schluss kürzen: "
                f"{display(f'{frac_tex(num_u, den_u)}={number_tex(Fraction(num, den))}')}"
            )
        else:
            steps.append(
                display(
                    f"{frac_tex(n_top, d_top)}:{frac_tex(n_bot, d_bot)}="
                    f"{frac_tex(n_top, d_top)}\\cdot{frac_tex(d_bot, n_bot)}={frac_tex(num, den)}"
                )
            )

        doppel = doppelbruch_display(num_inner, den_inner)
        aufgabe = f"Vereinfache den Doppelbruch:\n{display(doppel)}"
        if append_task(
            tasks, "brueche_doppel", aufgabe, steps, display(number_tex(Fraction(num, den)))
        ):
            created += 1
    if created != count:
        raise RuntimeError(f"Nur {created} von {count} Doppelbrüchen erzeugt.")


def gen_potenzen(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        kind = RNG.choices(["law", "sqrt", "combine"], weights=[4, 4, 2], k=1)[0]
        if kind == "law":
            operation = RNG.choice(["mul", "div"])
            base = RNG.randint(2, 8)
            if operation == "mul":
                e1, e2 = RNG.randint(1, 4), RNG.randint(1, 4)
                exp = e1 + e2
                symbol = r"\cdot"
                rule = "Bei gleicher Basis werden beim Multiplizieren die Exponenten addiert."
                calculation = f"{e1}+{e2}={exp}"
            else:
                e2 = RNG.randint(1, 4)
                e1 = RNG.randint(e2, e2 + 4)
                exp = e1 - e2
                symbol = ":"
                rule = "Bei gleicher Basis werden beim Dividieren die Exponenten subtrahiert."
                calculation = f"{e1}-{e2}={exp}"
            val = base**exp
            expression = f"{base}^{{{e1}}}{symbol}{base}^{{{e2}}}"
            aufgabe = f"Berechne:\n{display(expression)}"
            steps = [
                rule,
                f"Exponenten berechnen: {display(calculation)}",
                display(f"{expression}={base}^{{{exp}}}={val}"),
            ]
            kurz = display(str(val))
        elif kind == "sqrt":
            a = RNG.randint(2, 10)
            rest = RNG.choice([2, 3, 5, 6, 7, 10, 11, 13])
            n = a * a * rest
            aufgabe = f"Vereinfache:\n{display(rf'\sqrt{{{n}}}')}"
            steps = [
                "Wir suchen einen möglichst großen quadratischen Faktor unter der Wurzel.",
                f"Faktorzerlegung: {display(f'{n}={a**2}\\cdot {rest}={a}^2\\cdot {rest}')}",
                f"Mit {inline(r'\sqrt{a^2}=a')} kann der quadratische Faktor vor die Wurzel gezogen werden:",
                display(rf"\sqrt{{{n}}}=\sqrt{{{a**2}\cdot {rest}}}={a}\sqrt{{{rest}}}"),
            ]
            kurz = display(rf"{a}\sqrt{{{rest}}}")
        else:
            base = RNG.randint(2, 5)
            inner_exp = RNG.randint(2, 4)
            outer_exp = RNG.randint(2, 5)
            exp = inner_exp * outer_exp
            val = base**exp
            expression = f"({base}^{{{inner_exp}}})^{{{outer_exp}}}"
            aufgabe = f"Berechne:\n{display(expression)}"
            steps = [
                "Bei einer Potenz von einer Potenz werden die Exponenten multipliziert.",
                f"Exponenten multiplizieren: {display(f'{inner_exp}\\cdot {outer_exp}={exp}')}",
                display(f"{expression}={base}^{{{exp}}}={val}"),
            ]
            kurz = display(str(val))
        if append_task(tasks, "potenzen_wurzel", aufgabe, steps, kurz):
            created += 1


def gen_logarithmus(tasks: list, count: int) -> None:
    bases = [2, 3, 4, 5, 10]
    created = 0
    while created < count:
        b = RNG.choice(bases)
        e = RNG.randint(0, 6)
        val = b**e
        aufgabe = f"Berechne:\n{display(rf'\log_{{{b}}}({val})')}"
        steps = [
            f"Definition: {inline(rf'\log_{{{b}}}({val})=x')} bedeutet "
            f"{inline(rf'{b}^x={val}')}.",
            f"Passende Potenz finden: {display(f'{b}^{{{e}}}={val}')}",
            display(rf"\log_{{{b}}}({val})={e}"),
        ]
        if append_task(tasks, "logarithmus", aufgabe, steps, display(str(e))):
            created += 1


def gen_prozent(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        p = RNG.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75])
        base = RNG.choice([40, 50, 80, 100, 120, 150, 200, 240, 300, 350, 400, 500])
        val = Fraction(p * base, 100)
        if val.denominator != 1:
            continue
        aufgabe = f"Berechne {inline(f'{p}\\%')} von {inline(str(base))}."
        steps = [
            f"Prozent bedeutet „von Hundert“: {display(f'{p}\\%=\\frac{{{p}}}{{100}}')}",
            "Den Prozentwert erhält man mit Prozentzahl geteilt durch 100 mal Grundwert:",
            display(
                f"\\frac{{{p}}}{{100}}\\cdot {base}="
                f"\\frac{{{p}\\cdot {base}}}{{100}}={number_tex(val)}"
            ),
        ]
        if append_task(tasks, "prozent", aufgabe, steps, display(number_tex(val))):
            created += 1


def _consume_legacy_dreisatz_rng(count: int) -> None:
    """Alte Zufallsfolge verbrauchen, damit spätere Themen stabil bleiben."""
    unit_prices = [120, 150, 175, 200, 225, 250, 275, 300, 325, 350]
    created = 0
    seen: set[tuple[str, str]] = set()
    while created < count:
        n1 = RNG.randint(3, 8)
        unit_cents = RNG.choice(unit_prices)
        n2 = RNG.randint(n1 + 1, n1 + 7)
        price_s = decimal_comma(n1 * unit_cents)
        total_s = decimal_comma(unit_cents * n2)
        signature = (
            f"{n1} kg kosten {inline(price_s + r'\ \text{Euro}')}. "
            f"Wie viel kosten {inline(str(n2))} kg?",
            display(rf"{total_s}\ \text{{Euro}}"),
        )
        if signature in seen or signature in SEEN_TASKS:
            continue
        seen.add(signature)
        created += 1


def _euro(cents: int) -> str:
    return f"{decimal_comma(cents)} Euro"


def _math_amount(cents: int) -> str:
    return decimal_comma(cents).replace(",", "{,}")


def _direct_steps(
    reason: str,
    one_label: str,
    given_formula: str,
    one_formula: str,
    one_sentence: str,
    target_formula: str,
    result_sentence: str,
) -> list[str]:
    return [
        "Proportionaler Dreisatz: Die gesuchte Größe wächst im selben Verhältnis.",
        reason,
        f"Zuerst die Größe für 1 {one_label} bestimmen:",
        display(f"{given_formula}={one_formula}"),
        one_sentence,
        "Danach auf die gefragte Menge hochrechnen:",
        display(target_formula),
        result_sentence,
    ]


def _inverse_steps(
    work_sentence: str,
    product_formula: str,
    divide_formula: str,
    result_sentence: str,
) -> list[str]:
    return [
        "Umgekehrter Dreisatz: Mehr Helfer brauchen weniger Zeit, die Gesamtarbeit bleibt gleich.",
        work_sentence,
        display(product_formula),
        "Diese Gesamtarbeit auf die neue Anzahl verteilen:",
        display(divide_formula),
        result_sentence,
    ]


def _pick_other(local: random.Random, options: list[int], forbidden: int) -> int:
    return local.choice([value for value in options if value != forbidden])


def gen_dreisatz(tasks: list, count: int) -> None:
    _consume_legacy_dreisatz_rng(count)
    local = random.Random(42)
    goods = [
        "Äpfel",
        "Bio-Kartoffeln",
        "Kaffeebohnen",
        "Reis",
        "Weizenmehl",
    ]
    tickets = ["Kinokarten", "Museumstickets", "Schwimmbadkarten"]
    recipes = [
        ("Pfannkuchen", "Mehl", "g"),
        ("Tomatensoße", "passierte Tomaten", "g"),
        ("Kakao", "Milch", "ml"),
        ("Salatdressing", "Olivenöl", "ml"),
    ]
    created = 0
    while created < count:
        kind = local.choices(
            ["market", "tickets", "recipe", "travel", "machine", "wage", "map", "material", "inverse"],
            weights=[1, 2, 2, 2, 2, 2, 2, 2, 2],
            k=1,
        )[0]
        if kind == "market":
            good = local.choice(goods)
            n1 = local.randint(2, 5)
            unit = local.choice([120, 150, 180, 200, 250, 300])
            n2 = _pick_other(local, [2, 3, 4, 5, 6, 8], n1)
            aufgabe = (
                f"Auf dem Wochenmarkt kosten {n1} kg {good} {_euro(n1 * unit)}. "
                f"Wie viel kosten {n2} kg {good} zum selben Kilopreis?"
            )
            steps = _direct_steps(
                f"Hier ist der Preis proportional zur Menge {good}.",
                f"kg {good}",
                f"{_math_amount(n1 * unit)}:{n1}",
                rf"{_math_amount(unit)}\ \text{{Euro}}",
                f"1 kg {good} kostet also {_euro(unit)}.",
                rf"{n2}\cdot {_math_amount(unit)}={_math_amount(n2 * unit)}\ \text{{Euro}}",
                f"{n2} kg {good} kosten {_euro(n2 * unit)}.",
            )
            kurz = display(rf"{_math_amount(n2 * unit)}\ \text{{Euro}}")
        elif kind == "tickets":
            name = local.choice(tickets)
            n1 = local.choice([2, 3, 4, 5])
            unit = local.choice([650, 750, 800, 900, 1200, 1500])
            n2 = _pick_other(local, [2, 3, 4, 6, 8], n1)
            aufgabe = (
                f"{n1} {name} kosten zusammen {_euro(n1 * unit)}. "
                f"Wie viel kosten {n2} {name} zum selben Stückpreis?"
            )
            steps = _direct_steps(
                f"Jede der {name} hat denselben Preis.",
                name[:-1] if name.endswith("n") else "Stück",
                f"{_math_amount(n1 * unit)}:{n1}",
                rf"{_math_amount(unit)}\ \text{{Euro}}",
                f"Eine Karte kostet also {_euro(unit)}.",
                rf"{n2}\cdot {_math_amount(unit)}={_math_amount(n2 * unit)}\ \text{{Euro}}",
                f"{n2} {name} kosten {_euro(n2 * unit)}.",
            )
            kurz = display(rf"{_math_amount(n2 * unit)}\ \text{{Euro}}")
        elif kind == "recipe":
            dish, ingredient, unit_name = local.choice(recipes)
            n1 = local.choice([2, 3, 4, 6])
            per = local.choice([40, 50, 60, 75, 80, 100, 120])
            n2 = _pick_other(local, [2, 3, 4, 5, 8, 10, 12], n1)
            aufgabe = (
                f"Für {n1} Portionen {dish} braucht man {n1 * per} {unit_name} {ingredient}. "
                f"Wie viel {ingredient} braucht man für {n2} Portionen?"
            )
            steps = _direct_steps(
                "Die Zutatmenge ist proportional zur Portionszahl.",
                "Portion",
                f"{n1 * per}:{n1}",
                rf"{per}\ \text{{{unit_name}}}",
                f"Für 1 Portion braucht man {per} {unit_name} {ingredient}.",
                rf"{n2}\cdot {per}={n2 * per}\ \text{{{unit_name}}}",
                f"Für {n2} Portionen braucht man {n2 * per} {unit_name} {ingredient}.",
            )
            kurz = display(rf"{n2 * per}\ \text{{{unit_name}}}")
        elif kind == "travel":
            who = local.choice(
                ["Lea fährt mit dem Fahrrad", "Der Regionalzug fährt", "Ein Linienbus fährt"]
            )
            t1 = local.choice([2, 3, 4, 5])
            speed = local.choice([12, 15, 18, 20, 24, 30])
            t2 = _pick_other(local, [2, 3, 4, 6, 8], t1)
            aufgabe = (
                f"{who} in {t1} Stunden {t1 * speed} km, bei gleichbleibendem Tempo. "
                f"Wie viele Kilometer sind es in {t2} Stunden?"
            )
            steps = _direct_steps(
                "Bei gleichem Tempo ist der Weg proportional zur Zeit.",
                "Stunde",
                f"{t1 * speed}:{t1}",
                rf"{speed}\ \text{{km}}",
                f"In 1 Stunde sind es {speed} km.",
                rf"{t2}\cdot {speed}={t2 * speed}\ \text{{km}}",
                f"In {t2} Stunden sind es {t2 * speed} km.",
            )
            kurz = display(rf"{t2 * speed}\ \text{{km}}")
        elif kind == "machine":
            machine, output, unit_name = local.choice(
                [
                    ("Ein Drucker schafft", "Seiten", "Seiten"),
                    ("Eine Pumpe fördert", "Liter Wasser", "Liter"),
                    ("Eine Abfüllanlage füllt", "Liter Saft", "Liter"),
                ]
            )
            t1 = local.choice([2, 3, 4, 5, 6])
            per = local.choice([8, 10, 12, 15, 20, 25])
            t2 = _pick_other(local, [2, 3, 4, 8, 10, 12], t1)
            aufgabe = (
                f"{machine} in {t1} Minuten {t1 * per} {output}. "
                f"Wie viele {unit_name} sind es in {t2} Minuten bei gleichem Tempo?"
            )
            steps = _direct_steps(
                "Die Menge ist proportional zur Zeit.",
                "Minute",
                f"{t1 * per}:{t1}",
                rf"{per}\ \text{{{unit_name}}}",
                f"In 1 Minute sind es {per} {unit_name}.",
                rf"{t2}\cdot {per}={t2 * per}\ \text{{{unit_name}}}",
                f"In {t2} Minuten sind es {t2 * per} {unit_name}.",
            )
            kurz = display(rf"{t2 * per}\ \text{{{unit_name}}}")
        elif kind == "wage":
            job = local.choice(["Nachhilfe", "Aushilfe im Labor", "Korrektur von Übungsblättern"])
            n1 = local.choice([2, 3, 4, 5])
            unit = local.choice([1200, 1400, 1500, 1600, 1800, 2000])
            n2 = _pick_other(local, [2, 3, 6, 7, 8], n1)
            aufgabe = (
                f"Für {n1} Stunden {job} werden {_euro(n1 * unit)} bezahlt. "
                f"Wie viel Geld sind es für {n2} Stunden zum selben Stundenlohn?"
            )
            steps = _direct_steps(
                "Der Lohn ist proportional zur Arbeitszeit.",
                "Stunde",
                f"{_math_amount(n1 * unit)}:{n1}",
                rf"{_math_amount(unit)}\ \text{{Euro}}",
                f"Der Stundenlohn beträgt {_euro(unit)}.",
                rf"{n2}\cdot {_math_amount(unit)}={_math_amount(n2 * unit)}\ \text{{Euro}}",
                f"Für {n2} Stunden sind es {_euro(n2 * unit)}.",
            )
            kurz = display(rf"{_math_amount(n2 * unit)}\ \text{{Euro}}")
        elif kind == "map":
            cm1 = local.choice([2, 3, 4, 5])
            km_per = local.choice([2, 3, 4, 5])
            cm2 = _pick_other(local, [2, 3, 4, 6, 8, 10], cm1)
            aufgabe = (
                f"Auf einer Landkarte entsprechen {cm1} cm einer Strecke von {cm1 * km_per} km. "
                f"Welcher Strecke entsprechen {cm2} cm im selben Maßstab?"
            )
            steps = _direct_steps(
                "Im festen Maßstab ist die wirkliche Strecke proportional zur Kartenlänge.",
                "cm auf der Karte",
                f"{cm1 * km_per}:{cm1}",
                rf"{km_per}\ \text{{km}}",
                f"1 cm auf der Karte entspricht {km_per} km.",
                rf"{cm2}\cdot {km_per}={cm2 * km_per}\ \text{{km}}",
                f"{cm2} cm entsprechen {cm2 * km_per} km.",
            )
            kurz = display(rf"{cm2 * km_per}\ \text{{km}}")
        elif kind == "material":
            singular, plural, need, unit_name = local.choice(
                [
                    ("m² Wand", "m² Wand", "Farbe", "Liter"),
                    ("Beet", "Beete", "Mulch", "Liter"),
                    ("Regalbrett", "Regalbretter", "Holzlasur", "ml"),
                ]
            )
            n1 = local.choice([2, 3, 4, 5])
            per = local.choice([2, 3, 4, 5, 6]) if unit_name == "Liter" else local.choice([40, 50, 60, 80])
            n2 = _pick_other(local, [2, 3, 6, 8, 10], n1)
            aufgabe = (
                f"Für {n1} {plural} braucht man {n1 * per} {unit_name} {need}. "
                f"Wie viel {need} braucht man für {n2} {plural}?"
            )
            steps = _direct_steps(
                f"Der Verbrauch an {need} ist proportional zur Anzahl.",
                singular,
                f"{n1 * per}:{n1}",
                rf"{per}\ \text{{{unit_name}}}",
                f"Für 1 {singular} braucht man {per} {unit_name} {need}.",
                rf"{n2}\cdot {per}={n2 * per}\ \text{{{unit_name}}}",
                f"Für {n2} {plural} braucht man {n2 * per} {unit_name} {need}.",
            )
            kurz = display(rf"{n2 * per}\ \text{{{unit_name}}}")
        else:
            people1 = local.choice([2, 3, 4, 6])
            hours = local.choice([2, 3, 4, 5, 6])
            people2 = _pick_other(local, [2, 3, 4, 5, 8], people1)
            total_work = people1 * hours
            if total_work % people2 != 0:
                continue
            result = total_work // people2
            job = local.choice(
                ["ein Versuchsprotokoll", "das Aufräumen des Labors", "eine Inventur"]
            )
            aufgabe = (
                f"{people1} Studierende brauchen gemeinsam {hours} Stunden für {job}, "
                f"wenn alle gleich schnell arbeiten. Wie viele Stunden brauchen {people2} Studierende?"
            )
            steps = _inverse_steps(
                "Zuerst die gesamte Arbeit in Personenstunden bestimmen.",
                f"{people1}\\cdot {hours}={total_work}\\ \\text{{Personenstunden}}",
                f"{total_work}:{people2}={result}\\ \\text{{Stunden}}",
                f"{people2} Studierende brauchen {result} Stunden.",
            )
            kurz = display(rf"{result}\ \text{{Stunden}}")
        if append_task(tasks, "dreisatz", aufgabe, steps, kurz):
            created += 1


def gen_gleichungen(tasks: list, count: int) -> None:
    created = 0
    linear_target = round(count * 0.46)
    exp_target = round(count * 0.27)
    linear_created = 0
    exp_created = 0
    while created < count:
        if linear_created < linear_target:
            kind = "linear"
        elif exp_created < exp_target:
            kind = "exp"
        else:
            kind = "sqrt"
        if kind == "linear":
            a = RNG.randint(2, 9)
            b = RNG.randint(1, 20)
            x = RNG.randint(-5, 10)
            sign = RNG.choice([1, -1])
            signed_b = sign * b
            c = a * x + signed_b
            middle = f"+{b}" if signed_b > 0 else f"-{b}"
            aufgabe = f"Löse nach {inline('x')}:\n{display(f'{a}x{middle}={c}')}"
            inverse = f"{c}-{b}" if signed_b > 0 else f"{c}+{b}"
            isolated = c - signed_b
            steps = [
                "Zuerst den konstanten Term auf beiden Seiten rückgängig machen.",
                display(f"{a}x{middle}={c}\\quad\\Rightarrow\\quad {a}x={inverse}={isolated}"),
                f"Nun beide Seiten durch {inline(str(a))} teilen:",
                display(f"x={frac_tex(isolated, a)}={x}"),
            ]
            kurz = display(f"x={x}")
        elif kind == "exp":
            base = RNG.choice([2, 3, 4, 5])
            coefficient = RNG.choice([1, 2, 3])
            x = RNG.randint(1, 6)
            exponent = coefficient * x
            rhs = base**exponent
            left_exp = "x" if coefficient == 1 else f"{coefficient}x"
            aufgabe = f"Löse nach {inline('x')}:\n{display(f'{base}^{{{left_exp}}}={rhs}')}"
            steps = [
                "Die rechte Seite als Potenz mit derselben Basis schreiben:",
                display(f"{rhs}={base}^{{{exponent}}}"),
                "Bei gleichen Basen müssen die Exponenten gleich sein:",
                display(f"{left_exp}={exponent}"),
            ]
            if coefficient > 1:
                steps.extend(
                    [
                        f"Beide Seiten durch {inline(str(coefficient))} teilen:",
                        display(f"x=\\frac{{{exponent}}}{{{coefficient}}}={x}"),
                    ]
                )
            else:
                steps.append(display(f"x={x}"))
            kurz = display(f"x={x}")
        else:
            variant = RNG.choice(["plain", "shift", "isolate"])
            if variant == "plain":
                root = RNG.randint(2, 12)
                x = root * root
                aufgabe = f"Löse die Wurzelgleichung:\n{display(rf'\sqrt{{x}}={root}')}"
                steps = [
                    "Eine Quadratwurzel ist nur für nichtnegative Zahlen definiert und selbst nie negativ.",
                    "Beide Seiten quadrieren:",
                    display(rf"(\sqrt{{x}})^2={{{root}}}^2\quad\Rightarrow\quad x={x}"),
                    f"Probe in der Ausgangsgleichung: {inline(rf'\sqrt{{{x}}}={root}')}. Die Lösung passt.",
                ]
            elif variant == "shift":
                shift = RNG.randint(1, 9)
                root = RNG.randint(2, 10)
                x = root * root - shift
                if x < 0:
                    continue
                aufgabe = f"Löse die Wurzelgleichung:\n{display(rf'\sqrt{{x+{shift}}}={root}')}"
                steps = [
                    "Zuerst prüfen, wann der Ausdruck unter der Wurzel erlaubt ist: "
                    f"{inline(f'x+{shift}\\geq 0')}.",
                    "Beide Seiten quadrieren:",
                    display(rf"x+{shift}={{{root}}}^2={root * root}"),
                    f"Danach {inline(str(shift))} subtrahieren:",
                    display(f"x={root * root}-{shift}={x}"),
                    f"Probe: {inline(rf'\sqrt{{{x}+{shift}}}=\sqrt{{{root * root}}}={root}')}.",
                ]
            else:
                added = RNG.randint(1, 8)
                root = RNG.randint(2, 9)
                right = root + added
                x = root * root
                aufgabe = (
                    f"Löse die Wurzelgleichung:\n{display(rf'\sqrt{{x}}+{added}={right}')}"
                )
                steps = [
                    "Zuerst die Wurzel allein auf eine Seite bringen.",
                    display(rf"\sqrt{{x}}+{added}={right}\quad\Rightarrow\quad \sqrt{{x}}={right}-{added}={root}"),
                    "Beide Seiten quadrieren:",
                    display(rf"x={{{root}}}^2={x}"),
                    "Probe in der Ausgangsgleichung: "
                    f"{inline(rf'\sqrt{{{x}}}+{added}={root}+{added}={right}')}.",
                ]
            kurz = display(f"x={x}")
        if append_task(tasks, "gleichungen", aufgabe, steps, kurz):
            created += 1
            if kind == "linear":
                linear_created += 1
            elif kind == "exp":
                exp_created += 1


def gen_ungleichungen(tasks: list, count: int) -> None:
    created = 0
    while created < count:
        a = RNG.randint(2, 9)
        negative_coeff = RNG.random() < 0.5
        coeff = -a if negative_coeff else a
        relation = RNG.choice([">", "<"])
        result_relation = ("<" if relation == ">" else ">") if negative_coeff else relation
        pattern = RNG.choice(["only", "plus", "minus"])
        if pattern == "only":
            c = RNG.choice([value for value in range(-18, 25) if value != 0])
            left = f"{coeff}x" if coeff < 0 else f"{a}x"
            isolated = c
            prepare = (
                f"Vor {inline('x')} steht schon allein der Koeffizient {inline(str(coeff))}."
            )
            prepared = f"{left}{relation}{c}"
        else:
            b = RNG.randint(1, 12)
            c = RNG.randint(-15, 24)
            sign = "-" if coeff < 0 else ""
            if pattern == "plus":
                left = f"{sign}{a}x+{b}"
                isolated = c - b
                prepare = (
                    f"Zuerst {inline(str(b))} auf beiden Seiten subtrahieren. "
                    "Dabei dreht sich das Ungleichheitszeichen noch nicht um:"
                )
            else:
                left = f"{sign}{a}x-{b}"
                isolated = c + b
                prepare = (
                    f"Zuerst {inline(str(b))} auf beiden Seiten addieren. "
                    "Dabei dreht sich das Ungleichheitszeichen noch nicht um:"
                )
            prepared = f"{coeff}x{relation}{isolated}"
        bound = Fraction(isolated, coeff)
        if negative_coeff:
            divide_step = (
                f"Beide Seiten durch die negative Zahl {inline(str(coeff))} teilen. "
                "Beim Teilen durch eine negative Zahl dreht sich das Ungleichheitszeichen um:"
            )
            division = display(rf"{isolated}:({coeff})={number_tex(bound)}")
        else:
            divide_step = (
                f"Beide Seiten durch die positive Zahl {inline(str(a))} teilen. "
                "Darum bleibt das Ungleichheitszeichen unverändert:"
            )
            division = display(rf"{isolated}:{a}={number_tex(bound)}")
        aufgabe = f"Löse die Ungleichung:\n{display(f'{left}{relation}{c}')}"
        steps = [
            prepare,
            display(f"{left}{relation}{c}\\quad\\Rightarrow\\quad {prepared}"),
            divide_step,
            division,
            display(f"x{result_relation}{number_tex(bound)}"),
            f"Die Lösung sind alle Zahlen {inline('x')}, die "
            f"{'größer' if result_relation == '>' else 'kleiner'} als {inline(number_tex(bound))} sind.",
        ]
        if append_task(
            tasks,
            "ungleichungen",
            aufgabe,
            steps,
            display(f"x{result_relation}{number_tex(bound)}"),
        ):
            created += 1


def add_skript_vorlagen(tasks: list) -> None:
    append_task(
        tasks,
        "brueche_kuerzen",
        f"Kürze soweit wie möglich:\n{display(frac_tex(32, 8))}",
        [
            "Zum vollständigen Kürzen suchen wir den größten gemeinsamen Teiler von Zähler und Nenner.",
            f"Der größte gemeinsame Teiler von {inline('32')} und {inline('8')} ist {inline('8')}.",
            f"Zähler und Nenner durch {inline('8')} teilen:",
            display(r"\frac{32}{8}=\frac{32:8}{8:8}=\frac{4}{1}=4"),
        ],
        display("4"),
    )
    append_task(
        tasks,
        "brueche_add_sub",
        f"Berechne:\n{display(f'{frac_tex(2, 7)}+{frac_tex(5, 12)}')}",
        [
            "Brüche können nur mit gleichem Nenner addiert werden.",
            f"Das kleinste gemeinsame Vielfache der Nenner {inline('7')} und "
            f"{inline('12')} ist {inline('84')}.",
            f"Beide Brüche auf den Hauptnenner {inline('84')} erweitern:",
            display(
                f"{frac_expand_chain(2, 7, 12, 84)},\\quad {frac_expand_chain(5, 12, 7, 84)}"
            ),
            "Jetzt die Zähler addieren, der Nenner bleibt stehen:",
            display(
                f"{frac_tex(24, 84)}+{frac_tex(35, 84)}="
                f"\\frac{{24+35}}{{84}}={frac_tex(59, 84)}"
            ),
        ],
        display(frac_tex(59, 84)),
    )
    append_task(
        tasks,
        "brueche_mul_div",
        f"Berechne:\n{display(f'{frac_tex(2, 3)}:{frac_tex(4, 5)}')}",
        [
            "Durch einen Bruch teilen heißt mit seinem Kehrwert multiplizieren.",
            f"Der Kehrwert von {inline(frac_tex(4, 5))} ist {inline(frac_tex(5, 4))}.",
            display(
                f"{frac_tex(2, 3)}:{frac_tex(4, 5)}="
                f"{frac_tex(2, 3)}\\cdot{frac_tex(5, 4)}={frac_tex(10, 12)}"
            ),
            f"Der größte gemeinsame Teiler von {inline('10')} und {inline('12')} ist {inline('2')}.",
            f"Zum Schluss kürzen: {display(f'{frac_tex(10, 12)}={frac_tex(5, 6)}')}",
        ],
        display(frac_tex(5, 6)),
    )
    append_task(
        tasks,
        "brueche_doppel",
        f"Vereinfache den Doppelbruch:\n{display(r'\dfrac{\dfrac{3}{4}}{\dfrac{8}{9}}')}",
        [
            "Division durch einen Bruch = Multiplikation mit dem Kehrwert:",
            display(
                f"{frac_tex(3, 4)}:{frac_tex(8, 9)}="
                f"{frac_tex(3, 4)}\\cdot{frac_tex(9, 8)}={frac_tex(27, 32)}"
            ),
        ],
        display(frac_tex(27, 32)),
    )
    append_task(
        tasks,
        "brueche_doppel",
        f"Vereinfache den Doppelbruch:\n{display(doppelbruch_display(inner_frac_tex('+', 1, 2, 1, 3), inner_frac_tex(None, 4, 5)))}",
        [
            *combine_frac_steps("Zähler", "+", 1, 2, 1, 3)[2],
            "Division durch einen Bruch = Multiplikation mit dem Kehrwert:",
            display(
                f"{frac_tex(5, 6)}:{frac_tex(4, 5)}="
                f"{frac_tex(5, 6)}\\cdot{frac_tex(5, 4)}={frac_tex(25, 24)}"
            ),
        ],
        display(frac_tex(25, 24)),
    )


def validate_tasks(tasks: list[dict], expected_counts: dict[str, int]) -> None:
    """Strukturelle Qualitätskontrolle vor dem Schreiben der JSON-Datei."""
    if len(tasks) != 500:
        raise ValueError(f"Erwartet: 500 Aufgaben; erzeugt: {len(tasks)}")

    counts = Counter(task["topic_id"] for task in tasks)
    if counts != Counter(expected_counts):
        raise ValueError(f"Falsche Themenverteilung: {dict(counts)}")

    signatures = {(task["aufgabe"], task["loesung_kurz"]) for task in tasks}
    if len(signatures) != len(tasks):
        raise ValueError("Doppelte Aufgaben erkannt.")

    topic_ids = set(expected_counts)
    for index, task in enumerate(tasks, start=1):
        if task["nummer"] != index or task["id"] != f"aufg{index:04d}":
            raise ValueError(f"Fehlerhafte Nummerierung bei Position {index}.")
        if task["topic_id"] not in topic_ids:
            raise ValueError(f"Unbekanntes Thema: {task['topic_id']}")
        all_text = " ".join(
            [task["aufgabe"], *task["loesung_schritte"], task["loesung_kurz"]]
        )
        if re.search(r"\\(?:d?frac)\{[^{}]*\}\{0\}", all_text):
            raise ValueError(f"Nullnenner in {task['id']}.")
        if "2*e" in all_text:
            raise ValueError(f"Nicht ausgewerteter Exponent in {task['id']}.")


def main() -> None:
    topics = [
        {"id": "rechenregeln", "title": "Rechenregeln (Klammern, Punkt vor Strich)"},
        {"id": "brueche_kuerzen", "title": "Bruchrechnung: Kürzen und Erweitern"},
        {"id": "brueche_add_sub", "title": "Bruchrechnung: Addition und Subtraktion"},
        {"id": "brueche_mul_div", "title": "Bruchrechnung: Multiplikation und Division"},
        {"id": "brueche_doppel", "title": "Doppelbrüche"},
        {"id": "potenzen_wurzel", "title": "Potenzen und Wurzeln"},
        {"id": "logarithmus", "title": "Logarithmus"},
        {"id": "prozent", "title": "Prozentrechnung"},
        {"id": "dreisatz", "title": "Dreisatzrechnung"},
        {"id": "gleichungen", "title": "Gleichungen lösen"},
        {"id": "ungleichungen", "title": "Ungleichungen"},
    ]

    RNG.seed(42)
    SEEN_TASKS.clear()
    tasks: list[dict] = []
    add_skript_vorlagen(tasks)
    gen_rechenregeln(tasks, 70)
    gen_brueche_kuerzen(tasks, 39)
    gen_brueche_erweitern(tasks, 15)
    gen_brueche_add_sub(tasks, 49)
    gen_brueche_mul_div(tasks, 39)
    gen_brueche_doppel(tasks, 33)
    gen_potenzen(tasks, 55)
    gen_logarithmus(tasks, 25)
    gen_prozent(tasks, 30)
    gen_dreisatz(tasks, 30)
    gen_gleichungen(tasks, 65)
    gen_ungleichungen(tasks, 45)

    for index, item in enumerate(tasks, start=1):
        item["nummer"] = index
        item["id"] = f"aufg{index:04d}"

    expected_counts = {
        "rechenregeln": 70,
        "brueche_kuerzen": 55,
        "brueche_add_sub": 50,
        "brueche_mul_div": 40,
        "brueche_doppel": 35,
        "potenzen_wurzel": 55,
        "logarithmus": 25,
        "prozent": 30,
        "dreisatz": 30,
        "gleichungen": 65,
        "ungleichungen": 45,
    }
    validate_tasks(tasks, expected_counts)

    payload = {"topics": topics, "tasks": tasks}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Geschrieben: {OUTPUT} ({len(tasks)} Aufgaben)")


if __name__ == "__main__":
    main()
