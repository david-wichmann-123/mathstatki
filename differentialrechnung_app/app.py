"""Interaktive Grundlagen-App zur Differenzialrechnung."""

import re

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from sympy import E, Float, Function, Integer, Rational, Symbol, cos, diff, exp, lambdify, latex, log, pi, sin, sqrt, tan
from sympy.printing.jscode import jscode
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


st.set_page_config(page_title="Differentialrechnung - Grundlagen", page_icon="chart_with_upwards_trend", layout="wide")

X = Symbol("x")
TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)


def parse_function(formula: str):
    """Parse a student-entered formula using a deliberately small symbol set."""
    allowed_names = {
        "x": X, "sin": sin, "cos": cos, "tan": tan, "exp": exp,
        "log": log, "ln": log, "sqrt": sqrt, "pi": pi, "e": E,
    }
    names_in_formula = re.findall(r"[A-Za-z_]\w*", formula)
    if any(name not in allowed_names for name in names_in_formula):
        raise ValueError("Unknown function or variable")

    return parse_expr(
        formula,
        local_dict=allowed_names,
        global_dict={
            "__builtins__": {}, "Integer": Integer, "Float": Float,
            "Rational": Rational, "Symbol": Symbol, "Function": Function,
        },
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def values_for_range(expression, left: float, right: float) -> tuple[np.ndarray, np.ndarray]:
    """Initial server-side sampling; later samples are calculated in the browser."""
    # The browser uses the equivalent JavaScript expression after zooming or panning.
    numeric_function = lambdify(X, expression, modules=["numpy"])
    x_values = np.linspace(left, right, 1200)
    with np.errstate(all="ignore"):
        y_values = np.asarray(numeric_function(x_values), dtype=float)
    if y_values.ndim == 0:
        y_values = np.full_like(x_values, y_values.item(), dtype=float)
    y_values[~np.isfinite(y_values)] = np.nan
    y_values[np.abs(y_values) > 1e6] = np.nan
    return x_values, y_values


def padded_range(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [-1.0, 1.0]
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if low == high:
        low -= 1
        high += 1
    padding = (high - low) * 0.08
    return [low - padding, high + padding]


st.title("Differentialrechnung: Definition der Ableitung")
st.write("Rechts Funktion eingeben.")

plot_column, input_column = st.columns([2, 1], gap="large")

with input_column:
    st.subheader("Funktion eingeben")
    formula_label, formula_input = st.columns([1, 4], gap="small", vertical_alignment="center")
    with formula_label:
        st.markdown("**f(x) =**")
    with formula_input:
        formula = st.text_input("Funktion", value="x^2", placeholder="z. B. sin(x) + x^2", label_visibility="collapsed")
    st.caption("Beispiele: `x^2`, `exp(x)`, `ln(x)`, `log(x)`, `sin(x)`, `cos(x)`, `sqrt(x)`")
    st.caption("Auch moeglich: `2x + 1`, `e^x`, `pi*x`.")
    draw_derivative = st.checkbox("Ableitung zeichnen")
    st.info("Mit dem Mausrad zoomst du. Ziehe den Graphen zum Verschieben oder nutze die Toolbar.")

try:
    expression = parse_function(formula)
    derivative = diff(expression, X)
    x_values, y_values = values_for_range(expression, -6, 6)
    derivative_x_values, derivative_y_values = values_for_range(derivative, -6, 6)
    derivative_y_values[~np.isfinite(y_values)] = np.nan
    shared_y_range = padded_range(np.concatenate((y_values, derivative_y_values)))
    derivative_title = "Von dir gezeichnete Ableitung" if draw_derivative else f"$f'(x) = {latex(derivative)}$"
    displayed_derivative_x = [] if draw_derivative else derivative_x_values
    displayed_derivative_y = [] if draw_derivative else derivative_y_values

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=(f"$f(x) = {latex(expression)}$", derivative_title),
    )
    figure.add_trace(
        go.Scatter(x=x_values, y=y_values, mode="lines", line={"width": 3, "color": "#2563eb"}, name="f"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=displayed_derivative_x,
            y=displayed_derivative_y,
            mode="lines+markers" if draw_derivative else "lines",
            line={"width": 3, "color": "#16a34a"},
            marker={"size": 3} if draw_derivative else {},
            name="f'",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        xaxis={"range": [-6, 6], "showticklabels": True},
        xaxis2={"title": "x", "range": [-6, 6], "showticklabels": True},
        yaxis={"title": "f(x)", "range": shared_y_range},
        yaxis2={"title": "f'(x)", "range": shared_y_range},
        margin={"l": 45, "r": 15, "t": 55, "b": 45},
        height=630,
        showlegend=False,
        dragmode="pan",
    )

    # Runs in the Plotly browser canvas. It samples exactly the x interval currently visible.
    client_script = f"""
    const graph = document.getElementById('{{plot_id}}');
    const evaluate = (x) => {{ try {{ return {jscode(expression)}; }} catch (_) {{ return NaN; }} }};
    const slopeAt = (x) => {{ try {{ return {jscode(derivative)}; }} catch (_) {{ return NaN; }} }};
    const drawDerivative = {str(draw_derivative).lower()};
    function sample(left, right) {{
      const xs = [], ys = [], dys = [], count = 1200;
      for (let index = 0; index < count; index++) {{
        const x = left + (right - left) * index / (count - 1);
        const y = evaluate(x);
        const dy = slopeAt(x);
        xs.push(x);
        const functionIsDefined = Number.isFinite(y) && Math.abs(y) < 1e6;
        ys.push(functionIsDefined ? y : null);
        dys.push(functionIsDefined && Number.isFinite(dy) && Math.abs(dy) < 1e6 ? dy : null);
      }}
      return {{xs, ys, dys}};
    }}
    function yRange(values) {{
      const finite = values.filter(value => value !== null);
      if (!finite.length) return [-1, 1];
      let low = Math.min(...finite), high = Math.max(...finite);
      if (low === high) {{ low -= 1; high += 1; }}
      const padding = (high - low) * 0.08;
      return [low - padding, high + padding];
    }}
    const TRACE_FUNCTION = 0;
    const TRACE_DERIVATIVE = 1;
    const TRACE_TANGENT = 2;
    const TRACE_POINT = 3;
    const TRACE_TRIANGLE = 4;
    const TRACE_DERIVATIVE_POINT = 5;
    const TRACE_DERIVATIVE_GUIDE = 6;
    let activePointX = null;
    let draggingPoint = false;
    let internalRelayout = false;
    let pendingPointX = null;
    let selectionFrame = null;
    let selectionInFlight = false;
    let clearAfterSelection = false;
    let clearingSelection = false;
    let handleFrame = null;
    let lastDataClickAt = -Infinity;
    let lastViewportInteractionAt = -Infinity;
    let lastDrawnDerivativeX = null;
    const drawnDerivativePoints = new Map();
    const baseAnnotations = graph.layout.annotations ? [...graph.layout.annotations] : [];

    const dragHandle = document.createElement('div');
    Object.assign(dragHandle.style, {{
      position: 'absolute',
      width: '16px',
      height: '16px',
      borderRadius: '50%',
      background: '#dc2626',
      border: '2px solid white',
      boxShadow: '0 1px 5px rgba(0,0,0,0.28)',
      cursor: 'grab',
      zIndex: '1000',
      display: 'none',
      touchAction: 'none',
      transform: 'translate(-50%, -50%)',
    }});
    graph.style.position = graph.style.position || 'relative';
    graph.appendChild(dragHandle);

    function currentXRange() {{
      const range = graph._fullLayout.xaxis.range || graph.layout.xaxis.range;
      return [Number(range[0]), Number(range[1])];
    }}
    function xFromPointer(event) {{
      const axis = graph._fullLayout.xaxis;
      const rect = graph.getBoundingClientRect();
      const plotPixel = event.clientX - rect.left - axis._offset;
      const [left, right] = currentXRange();
      const x = axis.p2l(plotPixel);
      return Math.min(Math.max(x, Math.min(left, right)), Math.max(left, right));
    }}
    function selectedPointPixel() {{
      if (activePointX === null || !graph._fullLayout) return null;
      const y = evaluate(activePointX);
      if (!Number.isFinite(y)) return null;
      const xAxis = graph._fullLayout.xaxis, yAxis = graph._fullLayout.yaxis;
      return {{x: xAxis._offset + xAxis.l2p(activePointX), y: yAxis._offset + yAxis.l2p(y)}};
    }}
    function positionDragHandle() {{
      const point = selectedPointPixel();
      const xAxis = graph._fullLayout.xaxis, yAxis = graph._fullLayout.yaxis;
      if (!point || point.x < xAxis._offset || point.x > xAxis._offset + xAxis._length
        || point.y < yAxis._offset || point.y > yAxis._offset + yAxis._length) {{
        dragHandle.style.display = 'none';
        return false;
      }}
      dragHandle.style.display = 'block';
      dragHandle.style.left = point.x + 'px';
      dragHandle.style.top = point.y + 'px';
      return true;
    }}
    function isNearSelectedPoint(event) {{
      const point = selectedPointPixel();
      if (!point) return false;
      const rect = graph.getBoundingClientRect();
      const dx = event.clientX - rect.left - point.x;
      const dy = event.clientY - rect.top - point.y;
      return Math.hypot(dx, dy) <= 20;
    }}
    function isNearFunction(event) {{
      const xAxis = graph._fullLayout.xaxis, yAxis = graph._fullLayout.yaxis;
      const rect = graph.getBoundingClientRect();
      const plotX = event.clientX - rect.left, plotY = event.clientY - rect.top;
      if (plotX < xAxis._offset || plotX > xAxis._offset + xAxis._length
        || plotY < yAxis._offset || plotY > yAxis._offset + yAxis._length) return false;
      const y = evaluate(xFromPointer(event));
      if (!Number.isFinite(y)) return false;
      return Math.abs(plotY - (yAxis._offset + yAxis.l2p(y))) <= 14;
    }}
    function setPlotCursor(cursor, target = null) {{
      graph.style.cursor = cursor;
      graph.querySelectorAll('.nsewdrag, .ewdrag, .nsdrag').forEach(layer => {{
        layer.style.cursor = cursor;
      }});
      if (target && target.style) target.style.cursor = cursor;
    }}
    function recordDerivativePoint(x) {{
      if (!drawDerivative) return;
      const [left, right] = currentXRange();
      const spacing = Math.max(Math.abs(right - left) / 1200, 1e-6);
      const start = lastDrawnDerivativeX;
      const steps = start === null ? 1 : Math.max(1, Math.ceil(Math.abs(x - start) / spacing));
      for (let index = 1; index <= steps; index++) {{
        const pointX = start === null ? x : start + (x - start) * index / steps;
        const functionValue = evaluate(pointX);
        const slope = slopeAt(pointX);
        if (Number.isFinite(functionValue) && Number.isFinite(slope)) {{
          drawnDerivativePoints.set(pointX.toFixed(4), {{x: pointX, y: slope}});
        }}
      }}
      lastDrawnDerivativeX = x;
    }}
    function derivativeHistory() {{
      const points = [...drawnDerivativePoints.values()].sort((left, right) => left.x - right.x);
      return {{x: points.map(point => point.x), y: points.map(point => point.y)}};
    }}
    function scheduleDragHandlePosition() {{
      if (handleFrame !== null) return;
      handleFrame = requestAnimationFrame(() => {{
        handleFrame = null;
        positionDragHandle();
      }});
    }}
    function redrawSelection(x0) {{
      if (activePointX === null) return Promise.resolve();
      const [left, right] = currentXRange();
      const y0 = evaluate(x0);
      const slope = slopeAt(x0);
      if (!Number.isFinite(y0) || !Number.isFinite(slope)) return Promise.resolve();

      const yAxisRange = graph._fullLayout.yaxis.range;
      const triangleOptions = [-1, 1].map(direction => {{
        const xRoom = direction > 0 ? right - x0 : x0 - left;
        const verticalDirection = slope * direction;
        const yRoom = Math.abs(slope) < 1e-12
          ? Infinity
          : verticalDirection >= 0 ? yAxisRange[1] - y0 : y0 - yAxisRange[0];
        return {{direction, size: Math.min(1.25, xRoom * 0.82, yRoom * 0.82 / Math.abs(slope || 1))}};
      }});
      const triangleOption = triangleOptions.reduce((best, option) => option.size > best.size ? option : best);
      const deltaX = triangleOption.direction * Math.max(0, triangleOption.size);
      const deltaY = slope * deltaX;
      const x1 = x0 + deltaX, y1 = y0 + deltaY;
      const tangentX = [left, right];
      const tangentY = tangentX.map(x => y0 + slope * (x - x0));
      const tangent = {{
        x: tangentX, y: tangentY, type: 'scatter', mode: 'lines',
        line: {{color: '#dc2626', width: 2}}, name: 'Tangente',
        hoverinfo: 'skip', showlegend: false
      }};
      const selected = {{
        x: [x0], y: [y0], type: 'scatter', mode: 'markers',
        marker: {{color: '#dc2626', size: 12}},
        name: 'Beruehrpunkt', hovertemplate: 'x = %{{x:.4g}}<br>f(x) = %{{y:.4g}}<extra></extra>',
        showlegend: false, cliponaxis: false
      }};
      const triangle = {{
        x: [x0, x1, x1], y: [y0, y0, y1], type: 'scatter', mode: 'lines+markers',
        line: {{color: '#dc2626', width: 2, dash: 'dot'}}, marker: {{color: '#dc2626', size: 5}},
        customdata: [deltaX, deltaX, deltaX],
        name: 'Steigungsdreieck',
        hovertemplate: '\\u0394x = ' + Math.abs(deltaX).toPrecision(3) + '<br>\\u0394y = ' + deltaY.toPrecision(3) + '<extra></extra>',
        showlegend: false
      }};
      const derivativePoint = {{
        x: [x0], y: [slope], type: 'scatter', mode: 'markers+text',
        marker: {{color: '#dc2626', size: 10}},
        text: [slope.toPrecision(4)], textposition: 'top center',
        textfont: {{color: '#dc2626'}},
        xaxis: 'x2', yaxis: 'y2',
        name: 'Steigung', hovertemplate: "x = %{{x:.4g}}<br>f'(x) = %{{y:.4g}}<extra></extra>",
        showlegend: false, cliponaxis: false
      }};
      const derivativeGuide = {{
        x: [0, x0], y: [slope, slope], type: 'scatter', mode: 'lines',
        line: {{color: '#dc2626', width: 1.5, dash: 'dash'}},
        xaxis: 'x2', yaxis: 'y2', hoverinfo: 'skip', showlegend: false
      }};
      const drawnDerivative = derivativeHistory();
      const deltaXLabelShift = deltaY >= 0 ? -20 : 20;
      const deltaYLabelShift = deltaX >= 0 ? 46 : -46;
      const annotations = baseAnnotations.concat([{{
        x: (x0 + x1) / 2, y: y0, text: '$\\\\Delta x = ' + Math.abs(deltaX).toPrecision(3) + '$',
        showarrow: false, yshift: deltaXLabelShift, bgcolor: '#ffffff', borderpad: 3, font: {{color: '#dc2626'}}
      }}, {{
        x: x1, y: (y0 + y1) / 2, text: '$\\\\Delta y = ' + deltaY.toPrecision(3) + '$',
        showarrow: false, xshift: deltaYLabelShift, bgcolor: '#ffffff', borderpad: 3, font: {{color: '#dc2626'}}
      }}, {{
        xref: 'paper', yref: 'paper', x: 0.98, y: 1, xanchor: 'right', yanchor: 'bottom', yshift: 8,
        text: '$\\\\frac{{\\\\Delta y}}{{\\\\Delta x}} = ' + slope.toPrecision(4) + '$', showarrow: false,
        bgcolor: '#ffffff', borderpad: 3, font: {{color: '#dc2626'}}
      }}]);

      positionDragHandle();
      internalRelayout = true;
      const selectionTraceIndices = drawDerivative
        ? [TRACE_DERIVATIVE, TRACE_TANGENT, TRACE_POINT, TRACE_TRIANGLE, TRACE_DERIVATIVE_POINT, TRACE_DERIVATIVE_GUIDE]
        : [TRACE_TANGENT, TRACE_POINT, TRACE_TRIANGLE, TRACE_DERIVATIVE_POINT, TRACE_DERIVATIVE_GUIDE];
      const selectionX = [tangent.x, selected.x, triangle.x, derivativePoint.x, derivativeGuide.x];
      const selectionY = [tangent.y, selected.y, triangle.y, derivativePoint.y, derivativeGuide.y];
      const selectionText = [null, null, null, derivativePoint.text, null];
      const selectionCustomdata = [null, null, triangle.customdata, null, null];
      if (drawDerivative) {{
        selectionX.unshift(drawnDerivative.x);
        selectionY.unshift(drawnDerivative.y);
        selectionText.unshift(null);
        selectionCustomdata.unshift(null);
      }}
      const update = graph.data.length > TRACE_DERIVATIVE_GUIDE
        ? Plotly.update(graph, {{
          x: selectionX,
          y: selectionY,
          text: selectionText,
          customdata: selectionCustomdata,
        }}, {{annotations}}, selectionTraceIndices)
        : (drawDerivative
          ? Plotly.restyle(graph, {{x: [drawnDerivative.x], y: [drawnDerivative.y]}}, [TRACE_DERIVATIVE])
          : Promise.resolve())
          .then(() => Plotly.addTraces(graph, [tangent, selected, triangle, derivativePoint, derivativeGuide]))
          .then(() => Plotly.relayout(graph, {{annotations}}));
      return update.finally(() => {{
        internalRelayout = false;
        if (activePointX !== null) positionDragHandle();
      }});
    }}

    function scheduleSelection(x0) {{
      activePointX = x0;
      pendingPointX = x0;
      recordDerivativePoint(x0);
      positionDragHandle();
      if (selectionFrame !== null || selectionInFlight || clearingSelection) return;
      selectionFrame = requestAnimationFrame(() => {{
        selectionFrame = null;
        if (pendingPointX === null || activePointX === null) return;
        const nextX = pendingPointX;
        pendingPointX = null;
        selectionInFlight = true;
        redrawSelection(nextX)
          .catch(() => {{}})
          .finally(() => {{
            selectionInFlight = false;
            if (clearAfterSelection || activePointX === null) {{
              clearSelectionTraces();
            }} else if (pendingPointX !== null) {{
              scheduleSelection(pendingPointX);
            }}
          }});
      }});
    }}

    function clearSelectionTraces() {{
      if (clearingSelection || graph.data.length <= TRACE_TANGENT) return;
      clearAfterSelection = false;
      clearingSelection = true;
      internalRelayout = true;
      Plotly.deleteTraces(graph, [TRACE_TANGENT, TRACE_POINT, TRACE_TRIANGLE, TRACE_DERIVATIVE_POINT, TRACE_DERIVATIVE_GUIDE])
        .then(() => Plotly.relayout(graph, {{annotations: baseAnnotations}}))
        .finally(() => {{
          internalRelayout = false;
          clearingSelection = false;
          if (activePointX !== null) scheduleSelection(activePointX);
        }});
    }}

    function clearSelection() {{
      activePointX = null;
      pendingPointX = null;
      lastDrawnDerivativeX = null;
      dragHandle.style.display = 'none';
      if (selectionFrame !== null) {{
        cancelAnimationFrame(selectionFrame);
        selectionFrame = null;
      }}
      if (selectionInFlight) {{
        clearAfterSelection = true;
      }} else {{
        clearSelectionTraces();
      }}
    }}

    graph.on('plotly_relayouting', () => {{
      lastViewportInteractionAt = performance.now();
      scheduleDragHandlePosition();
    }});
    graph.on('plotly_afterplot', scheduleDragHandlePosition);
    graph.on('plotly_relayout', event => {{
      scheduleDragHandlePosition();
      if (internalRelayout) return;
      const topXRange = [Number(event['xaxis.range[0]']), Number(event['xaxis.range[1]'])];
      const bottomXRange = [Number(event['xaxis2.range[0]']), Number(event['xaxis2.range[1]'])];
      const hasTopXRange = Number.isFinite(topXRange[0]) && Number.isFinite(topXRange[1]) && topXRange[0] < topXRange[1];
      const hasBottomXRange = Number.isFinite(bottomXRange[0]) && Number.isFinite(bottomXRange[1]) && bottomXRange[0] < bottomXRange[1];
      const topYRange = [Number(event['yaxis.range[0]']), Number(event['yaxis.range[1]'])];
      const bottomYRange = [Number(event['yaxis2.range[0]']), Number(event['yaxis2.range[1]'])];
      const hasTopYRange = Number.isFinite(topYRange[0]) && Number.isFinite(topYRange[1]) && topYRange[0] < topYRange[1];
      const hasBottomYRange = Number.isFinite(bottomYRange[0]) && Number.isFinite(bottomYRange[1]) && bottomYRange[0] < bottomYRange[1];
      if (hasTopXRange || hasBottomXRange || hasTopYRange || hasBottomYRange) {{
        internalRelayout = true;
        const updates = [];
        const linkedXRange = hasTopXRange ? topXRange : bottomXRange;
        if (hasTopXRange || hasBottomXRange) {{
          const next = sample(linkedXRange[0], linkedXRange[1]);
          updates.push(Plotly.restyle(graph, {{x: [next.xs], y: [next.ys]}}, [TRACE_FUNCTION]));
          if (!drawDerivative) {{
            updates.push(Plotly.restyle(graph, {{x: [next.xs], y: [next.dys]}}, [TRACE_DERIVATIVE]));
          }}
        }}
        const layoutUpdate = {{}};
        if (hasTopXRange || hasBottomXRange) {{
          layoutUpdate['xaxis.range'] = linkedXRange;
          layoutUpdate['xaxis2.range'] = linkedXRange;
        }}
        if (hasTopYRange) layoutUpdate['yaxis.range'] = topYRange;
        if (hasBottomYRange) layoutUpdate['yaxis2.range'] = bottomYRange;
        Promise.all(updates)
          .then(() => Plotly.relayout(graph, layoutUpdate))
          .then(() => {{
            internalRelayout = false;
            if (activePointX !== null) scheduleSelection(activePointX);
          }})
          .catch(() => {{
            internalRelayout = false;
          }});
      }}
    }});
    graph.on('plotly_click', event => {{
      lastDataClickAt = performance.now();
      if (!event.points || !event.points.length || event.points[0].curveNumber !== TRACE_FUNCTION) return;
      const x0 = event.points[0].x;
      scheduleSelection(x0);
    }});
    graph.addEventListener('click', event => {{
      if (draggingPoint || dragHandle.contains(event.target) || event.target.closest('.modebar')) return;
      if (isNearFunction(event)) {{
        lastDataClickAt = performance.now();
        scheduleSelection(xFromPointer(event));
        return;
      }}
      window.setTimeout(() => {{
        const now = performance.now();
        if (now - lastDataClickAt < 120 || now - lastViewportInteractionAt < 120) return;
        clearSelection();
      }}, 50);
    }});
    dragHandle.addEventListener('click', event => event.stopPropagation());
    dragHandle.addEventListener('pointerdown', event => {{
      draggingPoint = true;
      dragHandle.style.cursor = 'grabbing';
      setPlotCursor('grabbing');
      dragHandle.setPointerCapture(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    }});
    graph.addEventListener('pointermove', event => {{
      if (draggingPoint) {{
        scheduleSelection(xFromPointer(event));
        event.preventDefault();
        event.stopPropagation();
        return;
      }}
      setPlotCursor(isNearSelectedPoint(event) || isNearFunction(event) ? 'grab' : '', event.target);
    }}, true);
    window.addEventListener('pointermove', event => {{
      if (!draggingPoint) return;
      scheduleSelection(xFromPointer(event));
      event.preventDefault();
    }}, true);
    window.addEventListener('pointerup', event => {{
      if (!draggingPoint) return;
      draggingPoint = false;
      dragHandle.style.cursor = 'grab';
      setPlotCursor('');
      try {{ dragHandle.releasePointerCapture(event.pointerId); }} catch (_) {{}}
      event.preventDefault();
      event.stopPropagation();
    }}, true);
    graph.addEventListener('pointerleave', () => {{
      if (!draggingPoint) setPlotCursor('');
    }});
    """

    with plot_column:
        chart_html = pio.to_html(
            figure, include_plotlyjs=True, include_mathjax="cdn", full_html=False, post_script=client_script,
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
        st.components.v1.html(chart_html, height=650)
except Exception:
    with plot_column:
        st.error("Diese Eingabe kann ich noch nicht als Funktion von x lesen.")
        st.info("Probiere zum Beispiel: `x^2`, `sin(x)`, `ln(x)` oder `exp(x)`.")
