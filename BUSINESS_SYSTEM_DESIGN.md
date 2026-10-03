# Business System Design

## Problem Statement

City traffic congestion, inefficient signal timing, and slow incident response raise commute times, fuel use, and emergency-response delays. Static, rule-based signal control cannot adapt to real-time conditions or anticipate buildup before it happens. The city needs a system that predicts traffic conditions ahead of time and recommends signal adjustments accordingly — one that can start small (a pilot district) and scale city-wide without being rebuilt at each stage.

## Stakeholders

| Stakeholder | Interest |
|---|---|
| City Department of Transportation (DOT) | Owns traffic outcomes; approves signal timing policy changes |
| Traffic Management Center operators | Day-to-day users who monitor and act on system recommendations |
| City IT / infrastructure team | Owns deployment, uptime, and integration with existing systems |
| Emergency services | Benefit from faster corridor clearance and incident-aware signal changes |
| Residents / commuters | End beneficiaries of reduced congestion and commute time |
| Hardware/software vendors (legacy signal controllers) | Integration partners for applying recommendations to physical signals |

## Business Goals

- Reduce average commute time and intersection wait time
- Detect and respond to congestion-causing incidents faster
- Reduce fuel consumption and emissions associated with idling/congestion
- Provide data-driven evidence for future infrastructure investment
- Avoid re-architecting the system as it scales from a pilot to city-wide, and eventually regional, coverage

## Success Metrics (KPIs)

- % reduction in average commute time / intersection wait time, pilot vs. baseline
- Forecast accuracy (MAE) for traffic volume/speed/occupancy
- Time-to-detect for incident-driven congestion
- System uptime for the serving layer
- Operator adoption rate (recommendations accepted vs. overridden)

## Business-Level System View

The system is organized into three layers, each independently scalable:

1. **Data Collection Layer** — traffic cameras, GPS/connected-vehicle feeds, and historical records, aggregated per intersection. (Currently simulated with realistic synthetic data until live feeds are connected — see [REQUIREMENTS.md](REQUIREMENTS.md).)
2. **Intelligence Layer** — forecasts near-term traffic conditions per intersection and converts forecasts into a recommended signal timing, within safety bounds.
3. **Action Layer** — existing traffic management systems and signal controllers that receive and apply (or an operator reviews and applies) the recommendation.

```text
[Cameras / GPS / Historical records]
            │
            ▼
   Data Collection Layer
            │
            ▼
   Intelligence Layer  (forecast + recommendation)
            │
            ▼
   Action Layer  (traffic mgmt center / signal controllers)
```

## Integration Points

- **Traffic Management Center software** — consumes recommendations, presents them to operators
- **Legacy signal controllers** — ultimately receive timing adjustments (protocol-specific integration is a future phase; see [REQUIREMENTS.md](REQUIREMENTS.md#out-of-scope-current-phase))
- **City GIS / asset systems** — intersection metadata (location, lane configuration)
- **Public dashboards** (optional, future) — aggregate congestion information for residents

## Rollout Strategy

1. **Pilot** — single district, read-only recommendations (operator-reviewed, not auto-applied)
2. **District-wide** — expand sensor coverage and intersections; begin measuring KPI impact
3. **City-wide** — scale training/serving infrastructure horizontally (cloud GPU/TPU) as data volume grows
4. **Regional** (stretch) — extend to neighboring municipalities' networks where coordination affects shared corridors

Each phase should meet its KPI targets before the next phase begins.

## Key Risks

| Risk | Mitigation |
|---|---|
| Legacy sensor/controller data quality or availability | Start with synthetic data for development; validate against real feeds incrementally per intersection |
| Integration risk with proprietary signal hardware | Treat as a scoped, separate integration phase rather than a blocking dependency for model development |
| Model drift during atypical events (construction, major events, holidays) | Automated drift monitoring and retraining (see [SKILLS.md](SKILLS.md#mlops--production-ml)) |
| Privacy/compliance concerns with camera or vehicle data | Operate on aggregate metrics (volume/speed/occupancy), not vehicle-level identifiers |
| Infrastructure cost at city/regional scale | Horizontal scaling is opt-in per phase, not provisioned upfront |

## Out of Scope (Business Level)

- Automatically applying signal changes without operator review during the pilot phase
- Multi-modal prioritization (transit, pedestrians, cyclists) — candidate for a later phase
