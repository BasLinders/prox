**Process mining documentation**

- The Process Model is your ideal customer journey map. The perfect, most efficient path you want users to take to convert (e.g., Homepage → Product Page → Add to Cart → Checkout → Purchase).
- The Event Log is the messy, real-world data of what all your users actually did.

Metrics
- Fitness: Can the model duplicate traces it finds in the log?
- Precision: Does the ideal map allow for crazy, nonsensical journeys that never happen?
- Alignment: A step by step score for how much a single user journey deviates from the ideal path (Deviations have a 'cost').

1. Fitness (Recall)
"Can the model replay the reality?"
What it measures: The percentage of traces in the Event Log that can be perfectly replayed by the Process Model from start to finish without forcing any token moves.

The Analogy:
Think of the Event Log as a set of Road Trips your visitors took, and the Process Model as a Map.
 - 100% Fitness: Every single road trip taken by a visitor exists on the map.
 - Low Fitness: Visitors are driving off-road, through fields and rivers, because the map doesn't show the roads they actually used.

2. Precision (Specificity)
"Is the model too vague?"

What it measures: Does the model allow for behavior that never actually happens?

The Analogy:
 - High Precision: The map shows only the roads people actually use.
 - Low Precision: The map shows a giant paved parking lot covering the whole world. Sure, you can drive anywhere (High Fitness!), but the map is useless because it doesn't tell you where the specific lanes are.

The "Flower Model" Trap: If you have a model where every activity connects to every other activity, you will have 100% Fitness (because anything is possible), but near 0% Precision (because it implies chaos).
