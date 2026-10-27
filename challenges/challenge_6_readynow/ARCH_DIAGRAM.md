# ReadyNow! — Architecture

```mermaid
flowchart TB
    User([User]) --> Guard

    subgraph Workflow["Workflow: readynow  (fixed order)"]
        direction TB
        Guard["before_model_callback<br/>validate_user_input → log_user_prompt"]
        Guard -->|rejected| Refusal([Refusal message])
        Guard -->|allowed| Coord

        Coord["coordinator  (LlmAgent, root)<br/>states mission · picks sub-agents · drafts reply<br/>output_key = draft_answer"]
        Coord -. single_turn tool call .-> Weather["weather_agent<br/>NWS forecast + hazards"]
        Coord -. single_turn tool call .-> Search["search_agent<br/>google_search: news, alerts, shelters"]
        Coord -. single_turn tool call .-> Routes["routes_agent<br/>OSRM driving directions"]
        Coord -. single_turn tool call .-> QA["qa_agent<br/>preparedness & safety Q&A"]

        Coord --> Critique["critique  (LlmAgent)<br/>reads {draft_answer}<br/>output_key = critique_notes"]
        Critique --> Refine["refine  (LlmAgent, terminal)<br/>reads {draft_answer} {critique_notes}<br/>output_key = final_answer"]
    end

    Refine --> Answer([Final answer to user])

    Weather --> NWS[(api.weather.gov)]
    Weather --> Geo[(Open-Meteo geocoding)]
    Routes --> Geo
    Routes --> OSRM[(router.project-osrm.org)]
    Search --> GS[(Google Search grounding)]

    State[("Session state<br/>draft_answer · critique_notes · final_answer")]
    Coord -.writes.-> State
    State -.read.-> Critique
    State -.read.-> Refine

    Log[/"Interaction log (logging)<br/>USER » · MODEL » · REJECTED »"/]
    Guard -.-> Log
    Coord -.-> Log
    Critique -.-> Log
    Refine -.-> Log
```

## Request flow

1. **Validate:** two Gemini classifiers on the raw message: malicious? off-mission? Either → refusal, no sub-agent runs. Fail open on classifier error.
2. **Coordinate:** root `LlmAgent` states capabilities or calls the sub-agent(s) needed; writes a combined draft to state.
3. **Critique:** reviews the draft for urgency ordering, clarity, missing safety steps.
4. **Refine:** rewrites using the critique. Terminal node, so its reply is what the user receives.

Stages hand off through session state: each writes its reply under its `output_key`; the next stage reads it via `{placeholder}` in its instruction. Every stage logs its output; the coordinator also logs the user prompt and any rejection.

## Components

| Agent           | Type          | Role                                                                |
| --------------- | ------------- | ------------------------------------------------------------------- |
| `readynow`      | `Workflow`    | Runs coordinator > critique > refine.                               |
| `coordinator`   | `LlmAgent`    | Mission, delegation, draft. Carries validation + logging callbacks. |
| `weather_agent` | `single_turn` | NWS forecast and hazards for a US place.                            |
| `search_agent`  | `single_turn` | News, alerts, shelters via Google Search.                           |
| `routes_agent`  | `single_turn` | Driving directions between two US places via OSRM.                  |
| `qa_agent`      | `single_turn` | Preparedness and safety guidance.                                   |
| `critique`      | `LlmAgent`    | Reviews the draft.                                                  |
| `refine`        | `LlmAgent`    | Final user-facing reply.                                            |

Deployed to Vertex AI Agent Engine by `deploy.py`; optional Model Armor screening via `MODEL_ARMOR_TEMPLATE`.
