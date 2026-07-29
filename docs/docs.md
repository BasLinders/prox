# Process Mining Documentation

## Core Concepts
* **The Process Model:** Your **Ideal Journey**. This is the mapped-out, efficient path you want users to take (e.g., *Homepage → Product Page → Add to Cart → Checkout → Purchase*).
* **The Event Log:** The **Real World**. The raw, messy data of what all your users actually did.

---

## Key Performance Metrics

### 1. Fitness (Recall)
> *"Can the model replay the reality?"*

* **Definition:** The percentage of traces in the Event Log that can be perfectly replayed by the Process Model from start to finish.
* **The Analogy:** Think of the Event Log as **Road Trips** and the Process Model as a **Map**.
    * **100% Fitness:** Every single road trip taken exists on the map.
    * **Low Fitness:** Visitors are driving "off-road" because the map doesn't show the paths they actually used.

### 2. Precision (Specificity)
> *"Is the model too vague?"*

* **Definition:** Measures if the model allows for "ghost" behaviors—paths that are theoretically possible in the model but never actually happen in reality.
* **The "Flower Model" Trap:** A model where every activity connects to everything else. You get 100% Fitness (anything is possible) but near 0% Precision (it's just chaos).
* **The Analogy:**
    * **High Precision:** The map shows only the roads people actually use.
    * **Low Precision:** The map is a giant paved parking lot covering the whole world. You can drive anywhere, but it doesn't tell you where the specific lanes are.

### 3. Alignment
* **Definition:** A step-by-step score for how much a single user journey deviates from the ideal path.
* **Note:** Every deviation is assigned a **"cost"** to help quantify the friction. In PRoX, per-trace deviations (skipped and unsolicited activities) are available when running the **State Equation A\*** conformance method, shown in the Conformance tab.

---

## Petri Net Legend

* **Places (Circles):** A state or condition that must be met before a transition can occur. These hold **Tokens** (resources).
* **Transitions (Rectangles):** Represent the actual events or actions taking place.

---

## Conformance Methods in PRoX

| Method | Speed | Output |
| :--- | :--- | :--- |
| **Token Replay** | Fast | Overall fitness and precision scores. |
| **State Equation A\*** | Slower | Exact per-trace deviations (skipped and unsolicited activities), in addition to overall scores. |

---

## Model Status

| Component | Status | Description |
| :--- | :--- | :--- |
| **Process Discovery** | Active | Inductive Miner or Heuristics Miner generates the process model and "Happy Path" BPMN. |
| **Conformance Checking** | Active | Token Replay (fast) or State Equation A\* (per-trace deviations). |
| **Bottleneck Analysis** | Active | Activity/transition durations ranked by impact score, plus an overall process health score. |
| **Business Insights** | Active | Repeat purchase rate, inter-purchase timing, and revenue multiplier. |

PRoX is pure Python — there is no Cython/compiled component to build or maintain.
