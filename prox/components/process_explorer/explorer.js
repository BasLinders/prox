// Interactive process explorer: renders a directly-follows graph with
// Cytoscape.js and reports node/edge clicks back to Python as the
// `selection` state value. `cytoscape` is provided by the vendor bundle that
// __init__.py prepends to this file.

function cssVar(el, name, fallback) {
    const value = getComputedStyle(el).getPropertyValue(name).trim();
    return value || fallback;
}

// Maps a frequency onto an edge width in px, on a log scale so one very
// common transition doesn't flatten every other edge to a hairline.
function widthScale(values, minPx, maxPx) {
    const logs = values.map((v) => Math.log1p(v));
    const lo = Math.min(...logs);
    const hi = Math.max(...logs);
    return (v) => {
        if (hi === lo) return (minPx + maxPx) / 2;
        return minPx + ((Math.log1p(v) - lo) / (hi - lo)) * (maxPx - minPx);
    };
}

// Node boxes are sized from their label; Cytoscape's own `width: "label"` is
// deprecated, so the text is measured on an off-screen canvas instead.
const measureCtx = document.createElement("canvas").getContext("2d");
function labelWidth(label) {
    measureCtx.font = "12px sans-serif";
    return Math.min(measureCtx.measureText(String(label)).width, 140);
}

const UNIT_LABELS = { seconds: "s", minutes: "min", hours: "h", days: "d" };

function formatTime(value, unit) {
    if (value === null || value === undefined) return "-";
    const digits = Math.abs(value) < 10 ? 1 : 0;
    return `${value.toFixed(digits)} ${UNIT_LABELS[unit] || unit}`;
}

// Linear blend of two "#rrggbb" colours, t in [0, 1]. Falls back to `to` for
// anything that isn't a 6-digit hex (e.g. a theme variable given as rgb()).
function mix(from, to, t) {
    const parse = (c) => /^#[0-9a-f]{6}$/i.test(c) ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) : null;
    const a = parse(from);
    const b = parse(to);
    if (!a || !b) return to;
    return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

// Streamlit calls the component function again when its data changes (a slider,
// a new run) on the same parentElement, without necessarily running the
// previous call's cleanup first. Remember each mount's cleanup here so the next
// call can tear the old graph down instead of stacking a second one under it.
const activeMounts = new WeakMap();

export default function (component) {
    const { data, parentElement, setStateValue } = component;

    const previous = activeMounts.get(parentElement);
    if (previous) previous();
    const graph = data.graph;
    const metric = data.metric === "time" ? "time" : "frequency";

    const container = document.createElement("div");
    container.className = "pe-canvas";
    container.style.height = `${data.height || 600}px`;
    parentElement.appendChild(container);

    const text = cssVar(container, "--st-text-color", "#262730");
    const background = cssVar(container, "--st-background-color", "#ffffff");
    const surface = cssVar(container, "--st-secondary-background-color", "#f0f2f6");
    const accent = cssVar(container, "--st-primary-color", "#ff4b4b");
    const muted = cssVar(container, "--st-gray-color", "#808495");

    const edgeWidth = widthScale(graph.edges.map((e) => e.frequency), 1.5, 9);

    // Time view: colour edges from cool (fast) to hot (slow) on the mean
    // transition time. Start/end edges have no time and stay neutral.
    const times = graph.edges.map((e) => e.mean_time).filter((t) => t !== null && t !== undefined);
    const maxTime = times.length ? Math.max(...times) : 0;
    const edgeColor = (e) => {
        if (metric !== "time" || e.mean_time === null || e.mean_time === undefined || maxTime <= 0) return muted;
        return mix("#4f8fd6", "#d9534f", Math.sqrt(e.mean_time / maxTime));
    };
    const edgeLabel = (e) => (
        metric === "time" && e.mean_time !== null && e.mean_time !== undefined
            ? formatTime(e.mean_time, graph.time_unit)
            : String(e.frequency)
    );

    const elements = [
        ...graph.nodes.map((n) => ({
            group: "nodes",
            data: { id: n.id, label: n.label, kind: n.kind, cases: n.cases, events: n.events },
        })),
        ...graph.edges.map((e, i) => ({
            group: "edges",
            data: {
                id: `e${i}`,
                source: e.source,
                target: e.target,
                frequency: e.frequency,
                cases: e.cases,
                mean_time: e.mean_time,
                median_time: e.median_time,
                happy: e.happy,
                width: edgeWidth(e.frequency),
                color: edgeColor(e),
                label: edgeLabel(e),
            },
        })),
    ];

    const cy = cytoscape({
        container,
        elements,
        minZoom: 0.1,
        maxZoom: 3,
        style: [
            {
                selector: "node",
                style: {
                    label: "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "text-wrap": "wrap",
                    "text-max-width": 140,
                    "font-size": 12,
                    color: text,
                    "background-color": surface,
                    "border-width": 1.5,
                    "border-color": muted,
                    shape: "round-rectangle",
                    width: (n) => labelWidth(n.data("label")) + 24,
                    height: 34,
                },
            },
            {
                selector: "node[kind = 'start'], node[kind = 'end']",
                style: { shape: "ellipse", "background-color": accent, color: background, "border-width": 0, "font-weight": "bold" },
            },
            { selector: "edge", style: {
                width: "data(width)",
                "line-color": "data(color)",
                "target-arrow-color": "data(color)",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                opacity: 0.75,
                label: "data(label)",
                "font-size": 10,
                color: text,
                "text-background-color": background,
                "text-background-opacity": 0.85,
                "text-background-padding": 2,
            } },
            { selector: "edge[?happy]", style: metric === "time"
                ? { opacity: 1, "underlay-color": accent, "underlay-opacity": 0.25, "underlay-padding": 3 }
                : { "line-color": accent, "target-arrow-color": accent, opacity: 1 } },
            { selector: "node:selected, edge:selected", style: { "border-color": accent, "border-width": 3, "line-color": accent, "target-arrow-color": accent } },
        ],
        layout: { name: "preset" },
    });

    cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 40, rankSep: 70, edgeSep: 20, animate: false }).run();
    cy.fit(undefined, 30);

    // Hover tooltip. Built with textContent only: activity names come from the
    // uploaded log and must never be interpreted as markup.
    const tip = document.createElement("div");
    tip.className = "pe-tooltip";
    tip.hidden = true;
    container.appendChild(tip);

    const showTip = (evt, lines) => {
        tip.replaceChildren(...lines.map((line, i) => {
            const row = document.createElement("div");
            row.textContent = line;
            if (i === 0) row.className = "pe-tooltip-title";
            return row;
        }));
        const p = evt.renderedPosition;
        tip.style.left = `${Math.min(p.x + 14, container.clientWidth - 230)}px`;
        tip.style.top = `${Math.min(p.y + 14, container.clientHeight - 100)}px`;
        tip.hidden = false;
    };
    cy.on("mouseover", "node", (evt) => {
        const d = evt.target.data();
        showTip(evt, d.kind === "activity"
            ? [d.label, `${d.cases} cases`, `${d.events} events`]
            : [d.label, `${d.cases} cases`]);
    });
    cy.on("mouseover", "edge", (evt) => {
        const d = evt.target.data();
        const lines = [`${evt.target.source().data("label")} \u2192 ${evt.target.target().data("label")}`,
            `${d.frequency} times, in ${d.cases} cases`];
        if (d.mean_time !== null && d.mean_time !== undefined) {
            lines.push(`mean ${formatTime(d.mean_time, graph.time_unit)}`, `median ${formatTime(d.median_time, graph.time_unit)}`);
        }
        showTip(evt, lines);
    });
    cy.on("mouseout", "node, edge", () => { tip.hidden = true; });
    cy.on("pan zoom drag", () => { tip.hidden = true; });

    cy.on("tap", "node", (evt) => {
        const d = evt.target.data();
        setStateValue("selection", {
            type: "node", id: d.id, label: d.label, kind: d.kind, cases: d.cases, events: d.events,
        });
    });
    cy.on("tap", "edge", (evt) => {
        const d = evt.target.data();
        setStateValue("selection", {
            type: "edge", source: d.source, target: d.target, frequency: d.frequency, cases: d.cases,
            mean_time: d.mean_time, median_time: d.median_time,
        });
    });
    cy.on("tap", (evt) => {
        if (evt.target === cy) setStateValue("selection", null);
    });

    // Cytoscape caches the canvas's position on the page and only refreshes it
    // on window scroll/resize. Streamlit scrolls an inner container instead,
    // and the sidebar can resize the canvas without a window resize, so after
    // either one clicks would land offset from the nodes. Scroll events don't
    // bubble, hence the capture listener.
    let refreshQueued = false;
    const refresh = () => {
        if (refreshQueued) return;
        refreshQueued = true;
        requestAnimationFrame(() => {
            refreshQueued = false;
            cy.resize();
        });
    };
    window.addEventListener("scroll", refresh, true);
    const resizeObserver = new ResizeObserver(refresh);
    resizeObserver.observe(container);

    const cleanup = () => {
        activeMounts.delete(parentElement);
        window.removeEventListener("scroll", refresh, true);
        resizeObserver.disconnect();
        cy.destroy();
        container.remove();
    };
    activeMounts.set(parentElement, cleanup);
    return cleanup;
}
