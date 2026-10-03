"""Erzeugt 70 didaktische LGS-Aufgaben nach Kapitel 03 des Mathe-1-Skripts."""

from __future__ import annotations

import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "data" / "aufgaben.json"
RNG = random.Random(20261003)
VARIABLES = ("x", "y", "z", "w")
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")


def inline(text: str) -> str:
    return rf"\({text}\)"


def display(text: str) -> str:
    return rf"\[{text}\]"


def number_tex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return rf"{sign}\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def variable_term(coefficient: Fraction, variable: str) -> str:
    magnitude = abs(coefficient)
    factor = "" if magnitude == 1 else number_tex(magnitude)
    return f"{factor}{variable}"


def aligned_equation(
    row: list[Fraction], rhs: Fraction, n: int, index: int
) -> str:
    """Eine Zeile im Spaltenformat des Skripts: Variablen und Vorzeichen untereinander."""
    cells: list[str] = []
    seen_term = False
    for column, (coefficient, variable) in enumerate(zip(row[:n], VARIABLES[:n])):
        if column > 0:
            if coefficient != 0 and seen_term:
                cells.append("{}+{}" if coefficient > 0 else "{}-{}")
            else:
                cells.append("")
        if coefficient == 0:
            cells.append("")
        elif not seen_term and coefficient < 0:
            cells.append("-" + variable_term(coefficient, variable))
            seen_term = True
        else:
            cells.append(variable_term(coefficient, variable))
            seen_term = True
    if not seen_term:
        cells[-1] = "0"
    return (
        " & ".join(cells)
        + rf" &= {number_tex(rhs)} && \qquad\text{{({ROMAN[index]})}}"
    )


def system_tex(matrix: list[list[Fraction]], rhs: list[Fraction], n: int) -> str:
    """LGS ohne geschweifte Klammer, mit untereinander stehenden Variablentermen."""
    rows = [
        aligned_equation(row, value, n, index)
        for index, (row, value) in enumerate(zip(matrix, rhs))
    ]
    return display(
        rf"\begin{{alignedat}}{{{n + 1}}}"
        + r" \\ ".join(rows)
        + r"\end{alignedat}"
    )


def matrix_rank(matrix: list[list[Fraction]]) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    row = 0
    for column in range(len(work[0])):
        pivot = next((i for i in range(row, len(work)) if work[i][column] != 0), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        pivot_value = work[row][column]
        work[row] = [value / pivot_value for value in work[row]]
        for i in range(len(work)):
            if i == row or work[i][column] == 0:
                continue
            factor = work[i][column]
            work[i] = [
                work[i][j] - factor * work[row][j] for j in range(len(work[0]))
            ]
        row += 1
        if row == len(work):
            break
    return row


def classify(matrix: list[list[Fraction]], rhs: list[Fraction], n: int) -> str:
    rank_a = matrix_rank(matrix)
    rank_augmented = matrix_rank([row + [value] for row, value in zip(matrix, rhs)])
    if rank_augmented > rank_a:
        return "keine"
    if rank_a == n:
        return "eindeutig"
    return "unendlich"


def shape_name(m: int, n: int) -> str:
    if m < n:
        return "unterbestimmt"
    if m == n:
        return "quadratisch"
    return "überbestimmt"


def dot(row: list[Fraction], solution: list[Fraction]) -> Fraction:
    return sum((a * x for a, x in zip(row, solution)), Fraction(0))


def scale(row: list[Fraction], factor: int) -> list[Fraction]:
    return [factor * value for value in row]


def combine(rows: list[list[Fraction]], weights: list[int]) -> list[Fraction]:
    return [
        sum((weights[i] * rows[i][j] for i in range(len(rows))), Fraction(0))
        for j in range(len(rows[0]))
    ]


def random_independent_rows(n: int, rank: int) -> list[list[Fraction]]:
    rows: list[list[Fraction]] = []
    attempts = 0
    while len(rows) < rank:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Keine unabhängigen Zeilen gefunden")
        candidate = [Fraction(RNG.randint(-4, 4)) for _ in range(n)]
        if all(value == 0 for value in candidate):
            continue
        if matrix_rank(rows + [candidate]) > len(rows):
            rows.append(candidate)
    return rows


def has_exact_duplicate(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> bool:
    seen: set[tuple[Fraction, ...]] = set()
    for row, value in zip(matrix, rhs):
        key = tuple(row) + (value,)
        if key in seen:
            return True
        seen.add(key)
    return False


def dependent_row(
    base: list[list[Fraction]], forbidden: set[tuple[Fraction, ...]]
) -> tuple[list[Fraction], list[int]]:
    for _ in range(200):
        weights = [RNG.randint(-2, 2) for _ in base]
        if not any(weights) or sum(abs(value) for value in weights) > 5:
            continue
        # Eine exakte Kopie einer vorhandenen Zeile ist keine neue Gleichung.
        if sum(abs(value) for value in weights) == 1 and 1 in weights:
            continue
        row = combine(base, weights)
        if any(row) and tuple(row) not in forbidden:
            return row, weights
    raise RuntimeError("Keine neue abhängige Zeile gefunden")


def build_system(
    n: int, m: int, solution_type: str
) -> tuple[list[list[Fraction]], list[Fraction]]:
    if solution_type == "eindeutig":
        rank = n
    elif solution_type == "unendlich":
        rank = 0 if n == 1 else min(n - 1, m)
    else:
        rank = min(n, m - 1)

    solution = [Fraction(RNG.randint(-4, 4)) for _ in range(n)]
    if all(value == 0 for value in solution):
        solution[0] = Fraction(2)

    if rank == 0:
        matrix = [[Fraction(0)] * n for _ in range(m)]
        rhs = [Fraction(0)] * m
        return matrix, rhs

    base = random_independent_rows(n, rank)
    base_rhs = [dot(row, solution) for row in base]
    matrix = [row[:] for row in base]
    rhs = base_rhs[:]
    forbidden = {tuple(row) for row in matrix}

    while len(matrix) < m:
        row, weights = dependent_row(base, forbidden)
        value = sum(
            (weights[i] * base_rhs[i] for i in range(rank)), Fraction(0)
        )
        matrix.append(row)
        rhs.append(value)
        forbidden.add(tuple(row))

    if solution_type == "keine":
        rhs[-1] += Fraction(RNG.choice((-3, -2, -1, 1, 2, 3)))

    order = list(range(m))
    RNG.shuffle(order)
    matrix = [matrix[i] for i in order]
    rhs = [rhs[i] for i in order]
    return matrix, rhs


# Charakteristische Systeme aus den Beispielen und Übungen des Skriptkapitels.
# Sie werden beim ersten passenden Aufgabentyp eingesetzt.
SCRIPT_SYSTEMS: dict[tuple[int, int, str], list[tuple[list[list[int]], list[int]]]] = {
    (2, 2, "eindeutig"): [([[1, 2], [2, -1]], [6, 4])],
    (2, 2, "keine"): [([[2, 1], [4, 2]], [4, 10])],
    (2, 2, "unendlich"): [([[2, 1], [4, 2]], [4, 8])],
    (3, 3, "eindeutig"): [([[2, 2, 4], [1, 2, 1], [1, -1, 2]], [6, 7, 10])],
    (3, 3, "unendlich"): [([[1, 2, 1], [2, 1, -1], [3, 3, 0]], [4, 5, 9])],
    (4, 2, "keine"): [([[1, 1, 1, 1], [1, 1, 1, 1]], [5, 8])],
    (4, 3, "unendlich"): [
        ([[1, 1, 1, 1], [1, -1, 0, 0], [0, 1, -1, 0]], [10, 0, 0])
    ],
    (4, 4, "eindeutig"): [
        (
            [
                [2, -1, 3, 1],
                [-1, 4, 1, -2],
                [3, 2, -2, 4],
                [1, -3, 4, 2],
            ],
            [18, -11, 14, 25],
        )
    ],
}


def row_operation_tex(target: int, source: int, factor: Fraction) -> str:
    sign = "+" if factor > 0 else "-"
    magnitude = abs(factor)
    factor_text = "" if magnitude == 1 else number_tex(magnitude)
    return (
        rf"({ROMAN[target]}) {sign} {factor_text}({ROMAN[source]})"
        rf" \rightarrow ({ROMAN[target]})"
    )


def gaussian_elimination(
    matrix: list[list[Fraction]], rhs: list[Fraction], n: int
) -> tuple[list[list[Fraction]], list[Fraction], list[tuple[str, str]]]:
    a = [row[:] for row in matrix]
    b = rhs[:]
    steps: list[tuple[str, str]] = []
    pivot_row = 0

    for column in range(n):
        pivot = next(
            (i for i in range(pivot_row, len(a)) if a[i][column] != 0), None
        )
        if pivot is None:
            continue
        if pivot != pivot_row:
            a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
            b[pivot_row], b[pivot] = b[pivot], b[pivot_row]
            operation = rf"({ROMAN[pivot_row]}) \leftrightarrow ({ROMAN[pivot]})"
            steps.append((f"Zeilen tauschen: {inline(operation)}.", system_tex(a, b, n)))

        # Immer von der letzten Gleichung aufwärts eliminieren.
        for row in range(len(a) - 1, pivot_row, -1):
            if a[row][column] == 0:
                continue
            factor = -a[row][column] / a[pivot_row][column]
            a[row] = [
                a[row][j] + factor * a[pivot_row][j] for j in range(n)
            ]
            b[row] += factor * b[pivot_row]
            operation = row_operation_tex(row, pivot_row, factor)
            variable = VARIABLES[column]
            steps.append(
                (
                    f"Elimination von {inline(variable)} aus Gleichung "
                    f"{inline('(' + ROMAN[row] + ')')}: {inline(operation)}.",
                    system_tex(a, b, n),
                )
            )
        pivot_row += 1
        if pivot_row == len(a):
            break
    return a, b, steps


def rref(
    matrix: list[list[Fraction]], rhs: list[Fraction], n: int
) -> tuple[list[list[Fraction]], list[Fraction], list[int]]:
    work = [row[:] + [value] for row, value in zip(matrix, rhs)]
    pivot_columns: list[int] = []
    row = 0
    for column in range(n):
        pivot = next((i for i in range(row, len(work)) if work[i][column] != 0), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        pivot_value = work[row][column]
        work[row] = [value / pivot_value for value in work[row]]
        for i in range(len(work)):
            if i == row or work[i][column] == 0:
                continue
            factor = work[i][column]
            work[i] = [
                work[i][j] - factor * work[row][j] for j in range(n + 1)
            ]
        pivot_columns.append(column)
        row += 1
        if row == len(work):
            break
    return [line[:n] for line in work], [line[n] for line in work], pivot_columns


def affine_tex(constant: Fraction, terms: list[tuple[Fraction, str]]) -> str:
    result = "" if constant == 0 and terms else number_tex(constant)
    for coefficient, parameter in terms:
        if coefficient == 0:
            continue
        if not result:
            sign = "-" if coefficient < 0 else ""
        else:
            sign = " - " if coefficient < 0 else " + "
        magnitude = abs(coefficient)
        factor = "" if magnitude == 1 else number_tex(magnitude)
        result += f"{sign}{factor}{parameter}"
    return result or "0"


def unique_solution_steps(
    echelon_a: list[list[Fraction]], echelon_b: list[Fraction], n: int
) -> tuple[list[str], str]:
    values = [Fraction(0)] * n
    details: list[str] = []
    nonzero_rows = [
        (row, value)
        for row, value in zip(echelon_a, echelon_b)
        if any(coefficient != 0 for coefficient in row)
    ]
    for row, rhs in reversed(nonzero_rows):
        pivot = next(i for i, coefficient in enumerate(row) if coefficient != 0)
        known_sum = sum(
            (row[j] * values[j] for j in range(pivot + 1, n)), Fraction(0)
        )
        values[pivot] = (rhs - known_sum) / row[pivot]
        details.append(
            f"Aus Gleichung {inline(ROMAN[pivot])} folgt durch Rückwärtseinsetzen "
            f"{inline(VARIABLES[pivot] + ' = ' + number_tex(values[pivot]))}."
        )
    tuple_tex = r"\left(" + ", ".join(number_tex(value) for value in values) + r"\right)"
    details.append(
        "Damit besitzt das LGS genau eine Lösung: "
        + display(rf"\mathbb{{L}}=\left\{{{tuple_tex}\right\}}")
    )
    return details, display(rf"\mathbb{{L}}=\left\{{{tuple_tex}\right\}}")


def infinite_solution_steps(
    matrix: list[list[Fraction]], rhs: list[Fraction], n: int
) -> tuple[list[str], str]:
    reduced_a, reduced_b, pivots = rref(matrix, rhs, n)
    rank_a = len(pivots)
    free_columns = [column for column in range(n) if column not in pivots]
    free_names = [VARIABLES[column] for column in free_columns]

    expressions: dict[int, str] = {
        column: VARIABLES[column] for column in free_columns
    }
    for row_index, pivot in enumerate(pivots):
        terms = [
            (-reduced_a[row_index][column], VARIABLES[column])
            for column in free_columns
        ]
        expressions[pivot] = affine_tex(reduced_b[row_index], terms)

    assignments = [
        rf"{VARIABLES[column]}={expressions[column]}"
        for column in range(n)
        if column not in free_columns
    ]
    if len(free_names) == 1:
        free_text = f"Daher bleibt {inline(free_names[0])} frei wählbar."
    else:
        listed = ", ".join(inline(name) for name in free_names[:-1])
        free_text = (
            f"Daher bleiben {listed} und {inline(free_names[-1])} frei wählbar."
        )
    parameter_condition = ", ".join(
        rf"{name}\in\mathbb{{R}}" for name in free_names
    )
    pivot_label = "Pivotzeile" if rank_a == 1 else "Pivotzeilen"
    solution_body = (
        r"\mathbb{L}=\left\{\left("
        + ", ".join(expressions[column] for column in range(n))
        + rf"\right)\;\middle|\;{parameter_condition}\right\}}"
    )
    steps = [
        f"Die Zeilenstufenform besitzt nur {inline(str(rank_a))} {pivot_label} "
        f"für {inline(str(n))} Unbekannte. {free_text}",
    ]
    if assignments:
        steps.append(
            "Aus den verbleibenden Gleichungen folgt: "
            + ", ".join(inline(value) for value in assignments)
            + "."
        )
    steps.append("Damit gibt es unendlich viele Lösungen:" + display(solution_body))
    return steps, display(solution_body)


def no_solution_steps(
    echelon_a: list[list[Fraction]], echelon_b: list[Fraction]
) -> tuple[list[str], str]:
    contradiction = next(
        value
        for row, value in zip(echelon_a, echelon_b)
        if all(coefficient == 0 for coefficient in row) and value != 0
    )
    steps = [
        "In der Zeilenstufenform tritt die widersprüchliche Gleichung "
        + inline(rf"0={number_tex(contradiction)}")
        + " auf.",
        "Diese Gleichung kann für keine Wahl der Unbekannten erfüllt werden. "
        "Das LGS besitzt daher keine Lösung.",
    ]
    return steps, display(r"\mathbb{L}=\varnothing")


def build_homogeneous(
    n: int, solution_type: str
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Quadratisches homogenes System: alle Absolutglieder sind 0."""
    for _ in range(40):
        matrix, _rhs = build_system(n, n, solution_type)
        rhs = [Fraction(0)] * n
        if has_exact_duplicate(matrix, rhs):
            continue
        if classify(matrix, rhs, n) == solution_type:
            return matrix, rhs
    raise RuntimeError(f"Kein homogenes {n}×{n}-System vom Typ {solution_type}")


def make_task(
    number: int,
    n: int,
    m: int,
    solution_type: str,
    homogeneous: bool = False,
) -> dict[str, object]:
    script_queue = SCRIPT_SYSTEMS.get((n, m, solution_type), [])
    if homogeneous:
        matrix, rhs = build_homogeneous(n, solution_type)
    elif script_queue:
        integer_matrix, integer_rhs = script_queue.pop(0)
        matrix = [[Fraction(value) for value in row] for row in integer_matrix]
        rhs = [Fraction(value) for value in integer_rhs]
    else:
        matrix, rhs = [], []
        for _ in range(40):
            matrix, rhs = build_system(n, m, solution_type)
            if not has_exact_duplicate(matrix, rhs):
                break
        else:
            raise RuntimeError(f"Aufgabe {number} enthält doppelte Gleichungen")
    if has_exact_duplicate(matrix, rhs):
        raise AssertionError(f"Aufgabe {number} enthält zweimal dieselbe Gleichung")
    actual_type = classify(matrix, rhs, n)
    if actual_type != solution_type:
        raise AssertionError(f"Aufgabe {number}: {actual_type} statt {solution_type}")

    shape = shape_name(m, n)
    task_system = system_tex(matrix, rhs, n)
    echelon_a, echelon_b, elimination_steps = gaussian_elimination(matrix, rhs, n)
    homogeneity = ""
    if homogeneous:
        homogeneity = (
            " Alle Absolutglieder sind 0, das LGS ist daher zusätzlich **homogen**. "
            "Die triviale Lösung ist immer enthalten."
        )
    steps = [
        f"Es gilt {inline('m=' + str(m))} und {inline('n=' + str(n))}. "
        f"Das LGS ist daher **{shape}**.{homogeneity}",
        "Wir formen mit elementaren Zeilenoperationen zur Zeilenstufenform um. "
        "Jede Variable wird von der letzten Gleichung aufwärts eliminiert, "
        "und in jedem Schritt wird nur eine Gleichung verändert.",
    ]
    for explanation, snapshot in elimination_steps:
        steps.extend((explanation, snapshot))

    if solution_type == "eindeutig":
        ending, short = unique_solution_steps(echelon_a, echelon_b, n)
    elif solution_type == "unendlich":
        ending, short = infinite_solution_steps(matrix, rhs, n)
    else:
        ending, short = no_solution_steps(echelon_a, echelon_b)
    steps.extend(ending)

    return {
        "id": f"lgs{number:04d}",
        "topic_id": f"u{n}",
        "aufgabe": (
            "Bestimmen Sie die Lösungsmenge des folgenden "
            + ("homogenen LGS" if homogeneous else "LGS")
            + " mit dem Gaußverfahren.\n"
            + task_system
        ),
        "homogen": homogeneous,
        "loesung_schritte": steps,
        "loesung_kurz": short,
        "n_unbekannte": n,
        "n_gleichungen": m,
        "loesungstyp": solution_type,
        "systemform": shape,
        "nummer": number,
    }


# 70 Aufgaben, darunter je Kategorie zwei quadratische homogene Systeme.
SPECS = [
    # Eine Unbekannte (8)
    (1, 1, "eindeutig"), (1, 1, "eindeutig"), (1, 2, "eindeutig"),
    (1, 2, "keine"), (1, 3, "keine"),
    (1, 1, "eindeutig"), (1, 1, "eindeutig"), (1, 1, "eindeutig"),
    (1, 1, "eindeutig", True), (1, 1, "unendlich", True),
    # Zwei Unbekannte (13)
    (2, 1, "unendlich"), (2, 1, "unendlich"),
    (2, 2, "eindeutig"), (2, 2, "eindeutig"), (2, 2, "eindeutig"),
    (2, 2, "keine"), (2, 2, "unendlich"),
    (2, 3, "eindeutig"), (2, 3, "keine"), (2, 3, "unendlich"),
    (2, 2, "eindeutig"), (2, 2, "keine"), (2, 2, "unendlich"),
    (2, 2, "eindeutig", True), (2, 2, "unendlich", True),
    # Drei Unbekannte (23)
    (3, 1, "unendlich"), (3, 2, "unendlich"), (3, 2, "unendlich"),
    (3, 2, "keine"), (3, 2, "keine"),
    (3, 3, "eindeutig"), (3, 3, "eindeutig"), (3, 3, "eindeutig"),
    (3, 3, "eindeutig"), (3, 3, "keine"), (3, 3, "keine"),
    (3, 3, "unendlich"), (3, 3, "unendlich"),
    (3, 4, "eindeutig"), (3, 4, "eindeutig"), (3, 5, "eindeutig"),
    (3, 4, "keine"), (3, 5, "keine"),
    (3, 4, "unendlich"), (3, 5, "unendlich"),
    (3, 3, "eindeutig"), (3, 3, "keine"), (3, 3, "unendlich"),
    (3, 3, "eindeutig", True), (3, 3, "unendlich", True),
    # Vier Unbekannte (18)
    (4, 2, "unendlich"), (4, 2, "keine"),
    (4, 3, "unendlich"), (4, 3, "unendlich"), (4, 3, "keine"),
    (4, 4, "eindeutig"), (4, 4, "eindeutig"), (4, 4, "eindeutig"),
    (4, 4, "keine"), (4, 4, "keine"),
    (4, 4, "unendlich"), (4, 4, "unendlich"),
    (4, 5, "eindeutig"), (4, 5, "keine"), (4, 5, "unendlich"),
    (4, 4, "eindeutig"), (4, 4, "keine"), (4, 4, "unendlich"),
    (4, 4, "eindeutig", True), (4, 4, "unendlich", True),
]


def validate(tasks: list[dict[str, object]]) -> None:
    if len(tasks) != 70:
        raise AssertionError("Es müssen genau 70 Aufgaben sein")
    variable_counts = Counter(task["n_unbekannte"] for task in tasks)
    if variable_counts != Counter({1: 10, 2: 15, 3: 25, 4: 20}):
        raise AssertionError(f"Falsche Verteilung: {variable_counts}")
    homogeneous = [task for task in tasks if task["homogen"]]
    if len(homogeneous) != 8:
        raise AssertionError("Es müssen 8 homogene Aufgaben sein")
    if any(task["systemform"] != "quadratisch" for task in homogeneous):
        raise AssertionError("Homogene Aufgaben müssen quadratisch sein")
    if Counter(task["n_unbekannte"] for task in homogeneous) != Counter({1: 2, 2: 2, 3: 2, 4: 2}):
        raise AssertionError("Pro Kategorie müssen zwei homogene Aufgaben vorhanden sein")
    if len({task["aufgabe"] for task in tasks}) != len(tasks):
        raise AssertionError("Doppelte Aufgaben gefunden")
    for task in tasks:
        if r"\begin{cases}" in str(task):
            raise AssertionError("Geschweifte Systemklammer gefunden")


def main() -> None:
    topics = [
        {"id": "u1", "title": "Eine Unbekannte"},
        {"id": "u2", "title": "Zwei Unbekannte"},
        {"id": "u3", "title": "Drei Unbekannte"},
        {"id": "u4", "title": "Vier Unbekannte"},
    ]
    tasks = []
    for index, spec in enumerate(SPECS, start=1):
        n, m, solution_type = spec[0], spec[1], spec[2]
        homogeneous = bool(spec[3]) if len(spec) > 3 else False
        tasks.append(make_task(index, n, m, solution_type, homogeneous))
    validate(tasks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump({"topics": topics, "tasks": tasks}, handle, ensure_ascii=False, indent=2)
    print(f"{len(tasks)} geprüfte LGS-Aufgaben geschrieben: {OUTPUT}")


if __name__ == "__main__":
    main()
