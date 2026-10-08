// Drag across a chart with a mouse to open a full-screen chart of only the points inside the rectangle.
// The page works without this script; it only adds the zoom where JavaScript runs.
(() => {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  const dialog = document.getElementById("zoom");
  const plotHost = dialog.querySelector(".zoom-plot");
  const SVG = "http://www.w3.org/2000/svg";
  const MINIMUM_DRAG_PIXELS = 8;
  const LABEL_GAP_PIXELS = 21;
  const MARGIN = { left: 60, right: 170, top: 16, bottom: 44 };
  let drag = null;
  let shown = null;

  document.body.classList.add("zoomable");

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  function axesBox(plot) {
    const box = plot.getBoundingClientRect();
    const [left, top, width, height] = plot.dataset.axes.split(",").map(Number);
    return { x: box.left + box.width * left / 100, y: box.top + box.height * top / 100,
             width: box.width * width / 100, height: box.height * height / 100 };
  }

  function toData(metric, axes, clientX, clientY) {
    const [xLow, xHigh] = data.xLimits;
    const [yLow, yHigh] = metric.yRange;
    return {
      level: xLow + clamp((clientX - axes.x) / axes.width, 0, 1) * (xHigh - xLow),
      score: yHigh - clamp((clientY - axes.y) / axes.height, 0, 1) * (yHigh - yLow),
    };
  }

  document.addEventListener("pointerdown", event => {
    const plot = event.target.closest(".plot[data-metric]");  // effort charts only
    if (!plot || event.pointerType !== "mouse" || event.button !== 0) return;
    event.preventDefault();
    const box = document.createElement("div");
    box.className = "selection";
    plot.append(box);
    document.body.classList.add("dragging");
    drag = { plot, box, startX: event.clientX, startY: event.clientY, endX: event.clientX, endY: event.clientY };
  });

  document.addEventListener("pointermove", event => {
    if (!drag) return;
    drag.endX = event.clientX;
    drag.endY = event.clientY;
    const plotBox = drag.plot.getBoundingClientRect();
    Object.assign(drag.box.style, {
      left: `${Math.min(drag.startX, drag.endX) - plotBox.left}px`,
      top: `${Math.min(drag.startY, drag.endY) - plotBox.top}px`,
      width: `${Math.abs(drag.endX - drag.startX)}px`,
      height: `${Math.abs(drag.endY - drag.startY)}px`,
    });
  });

  document.addEventListener("pointerup", () => {
    if (!drag) return;
    const finished = drag;
    drag = null;
    finished.box.remove();
    document.body.classList.remove("dragging");
    const tooSmall = Math.abs(finished.endX - finished.startX) < MINIMUM_DRAG_PIXELS
      || Math.abs(finished.endY - finished.startY) < MINIMUM_DRAG_PIXELS;
    if (tooSmall) return;
    const metric = data.metrics[finished.plot.dataset.metric];
    const axes = axesBox(finished.plot);
    const topLeft = toData(metric, axes, Math.min(finished.startX, finished.endX), Math.min(finished.startY, finished.endY));
    const bottomRight = toData(metric, axes, Math.max(finished.startX, finished.endX), Math.max(finished.startY, finished.endY));
    const points = selectedPoints(metric, topLeft, bottomRight);
    if (!points.length) return;
    shown = { metric, points };
    dialog.showModal();
    dialog.focus();  // the close button would otherwise take focus and show its ring
    render();
  });

  function selectedPoints(metric, topLeft, bottomRight) {
    const visible = data.models.filter(model =>
      model.slug in metric.scores && document.getElementById(`show-${model.slug}`).checked);
    return visible.flatMap(model => metric.scores[model.slug]
      .map((score, level) => ({ model, level, score }))
      .filter(point => point.score !== null
        && point.level >= topLeft.level && point.level <= bottomRight.level
        && point.score <= topLeft.score && point.score >= bottomRight.score));
  }

  // Round tick values that cover the scores with a little headroom, about five steps.
  function niceTicks(low, high) {
    const padding = Math.max((high - low) * 0.12, Math.abs(high) * 0.01, 0.5);
    const paddedLow = low - padding;
    const paddedHigh = high + padding;
    const rawStep = (paddedHigh - paddedLow) / 5;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 2.5, 5, 10].map(factor => factor * magnitude).find(candidate => candidate >= rawStep);
    const first = Math.floor(paddedLow / step) * step;
    const last = Math.ceil(paddedHigh / step) * step;
    const decimals = (String(Number(step.toFixed(10))).split(".")[1] || "").length;
    const values = [];
    for (let value = first; value <= last + step / 2; value += step) values.push(value);
    return { low: first, high: last, values, decimals };
  }

  // Spread sorted positions so neighbors sit at least minimumGap apart, each crowded group centered on its mean.
  function spread(positions, minimumGap) {
    const layout = group => {
      const center = group.reduce((sum, value) => sum + value, 0) / group.length;
      return group.map((_, offset) => center + (offset - (group.length - 1) / 2) * minimumGap);
    };
    const groups = [];
    for (const position of [...positions].sort((a, b) => a - b)) {
      groups.push([position]);
      while (groups.length > 1 && layout(groups.at(-1))[0] - layout(groups.at(-2)).at(-1) < minimumGap) {
        groups.splice(-2, 2, [...groups.at(-2), ...groups.at(-1)]);
      }
    }
    return groups.flatMap(layout);
  }

  function element(parent, name, attributes, style = {}) {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    Object.assign(node.style, style);
    parent.append(node);
    return node;
  }

  function render() {
    const { metric, points } = shown;
    const levels = points.map(point => point.level);
    const firstLevel = Math.min(...levels);
    const lastLevel = Math.max(...levels);
    const scores = points.map(point => point.score);
    const ticks = niceTicks(Math.min(...scores), Math.max(...scores));
    const models = data.models.filter(model => points.some(point => point.model === model));
    const format = score => score.toFixed(metric.decimals);

    dialog.querySelector("h3").textContent = metric.name;
    dialog.querySelector(".zoom-unit").textContent = metric.unit;
    dialog.querySelector(".zoom-range").textContent =
      `${data.levels[firstLevel]} to ${data.levels[lastLevel]} · scores ${format(Math.min(...scores))} to ${format(Math.max(...scores))}`;
    dialog.querySelector(".zoom-legend").replaceChildren(...models.map(model => {
      const key = document.createElement("span");
      key.style.setProperty("--series", `var(--${model.color})`);
      key.textContent = model.name;
      return key;
    }));

    plotHost.replaceChildren();
    const width = plotHost.clientWidth;
    const height = plotHost.clientHeight;
    const svg = element(plotHost, "svg", { viewBox: `0 0 ${width} ${height}` });
    const x = level => MARGIN.left + (level - firstLevel + 0.5) / (lastLevel - firstLevel + 1) * (width - MARGIN.left - MARGIN.right);
    const y = score => MARGIN.top + (ticks.high - score) / (ticks.high - ticks.low) * (height - MARGIN.top - MARGIN.bottom);
    const text = { fontFamily: "inherit", fontSize: "15px" };

    for (const value of ticks.values) {
      element(svg, "line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: y(value), y2: y(value) },
        { stroke: "var(--grid)", strokeWidth: "1" });
      element(svg, "text", { x: MARGIN.left - 12, y: y(value), "text-anchor": "end", "dominant-baseline": "middle" },
        { ...text, fill: "var(--ink-3)" }).textContent = value.toFixed(ticks.decimals);
    }
    element(svg, "line", { x1: x(firstLevel - 0.35), x2: x(lastLevel + 0.35), y1: y(ticks.low), y2: y(ticks.low) },
      { stroke: "var(--baseline)", strokeWidth: "1" });
    for (let level = firstLevel; level <= lastLevel; level += 1) {
      element(svg, "text", { x: x(level), y: height - MARGIN.bottom + 28, "text-anchor": "middle" },
        { ...text, fill: "var(--ink-3)" }).textContent = data.levels[level];
    }

    for (const model of models) {
      const own = points.filter(point => point.model === model);
      for (const [from, to] of own.slice(0, -1).map((point, index) => [point, own[index + 1]])) {
        if (to.level !== from.level + 1) continue;
        element(svg, "line", { x1: x(from.level), y1: y(from.score), x2: x(to.level), y2: y(to.score) },
          { stroke: `var(--${model.color})`, strokeWidth: "3", strokeLinecap: "round" });
      }
    }
    for (const point of points) {
      const estimated = metric.estimated.some(([slug, level]) => slug === point.model.slug && level === point.level);
      element(svg, "circle", { cx: x(point.level), cy: y(point.score), r: 6.5 }, {
        fill: estimated ? "var(--surface)" : `var(--${point.model.color})`,
        stroke: estimated ? `var(--${point.model.color})` : "var(--surface)",
        strokeWidth: "2",
      });
    }

    // Each level's labels sit right of their points, spread apart vertically; each model's last point also names it.
    for (let level = firstLevel; level <= lastLevel; level += 1) {
      const column = points.filter(point => point.level === level).sort((a, b) => y(a.score) - y(b.score));
      const labelYs = spread(column.map(point => y(point.score)), LABEL_GAP_PIXELS);
      column.forEach((point, index) => {
        const pointX = x(point.level);
        const pointY = y(point.score);
        const labelY = labelYs[index];
        if (Math.abs(labelY - pointY) > 0.5) {
          element(svg, "line", { x1: pointX + 6, y1: pointY, x2: pointX + 14, y2: labelY },
            { stroke: "var(--baseline)", strokeWidth: "1" });
        }
        const label = element(svg, "text", { x: pointX + 16, y: labelY, "dominant-baseline": "middle" }, {
          ...text, fontWeight: "600", fill: "var(--ink-1)",
          stroke: "var(--surface)", strokeWidth: "4px", paintOrder: "stroke", strokeLinejoin: "round",
        });
        label.textContent = format(point.score);
        const isLast = !points.some(other => other.model === point.model && other.level > point.level);
        if (isLast) {
          const name = element(label, "tspan", { dx: "8" }, {
            fontFamily: "Menlo, monospace", fontSize: "12.5px", fontWeight: "400", fill: "var(--ink-3)",
          });
          name.textContent = point.model.name;
        }
      });
    }
  }

  dialog.addEventListener("close", () => { shown = null; });
  window.addEventListener("resize", () => { if (shown) render(); });
})();
