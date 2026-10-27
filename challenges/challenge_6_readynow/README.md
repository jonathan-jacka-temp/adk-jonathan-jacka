# ReadyNow! FEMA Emergency Preparedness Assistant - Challenge 6

Multi-agent ADK system that helps people during a disaster know what's
going on, where to go, and how to stay safe.

**Architecture diagram: [`ARCH_DIAGRAM.md`](ARCH_DIAGRAM.md)**

## Requirements and implementation

| Requirement                                                 | Where                                                                   |
| ----------------------------------------------------------- | ----------------------------------------------------------------------- |
| Architecture diagram                                        | `ARCH_DIAGRAM.md`                                                       |
| Root agent describing capabilities, coordinating sub-agents | `coordinator_agent.py`                                                  |
| Weather / search / routes / Q&A agents                      | `weather_agent.py`, `search_agent.py`, `routes_agent.py`, `qa_agent.py` |
| Sequential validate-and-refine workflow                     | `agent.py` — `Workflow(START → coordinator → critique → refine)`        |
| Log all user–agent interactions                             | `callbacks.py`                                                          |
| Validate input, refuse off-mission requests                 | `callbacks.py` — `is_safe_input`, `is_on_mission`                       |
| Deploy to Agent Platform                                    | `deploy.py` — run log in `deploy_output.log` (engine `reasoningEngines/2841087743516213248`, us-central1) |
| Test code                                                   | `test_callbacks.py` (13), `test_workflow.py` (6) - no API key needed    |
| Working deployed solution                                   | `deploy_output.log` step 3 — three prompts answered by the *remote* engine, incl. multi-agent delegation and off-mission refusal |
| UI evidence                                                 | `screenshots/` — `adk web` sessions: capabilities, multi-agent trace, critique/refine stages, refusal |

Supporting modules: `models.py` (model factory), `tools.py` (geocoding, NWS, OSRM).

## Running

```bash
uv run pytest challenges/challenge_6_readynow/          # tests, no credentials
uv run adk web                                          # chat UI; pick "challenge_6_readynow"
uv run python -m challenges.challenge_6_readynow.agent  # scripted demo
PROJECT_ID=... STAGING_BUCKET=gs://... uv run python -m challenges.challenge_6_readynow.deploy
```

Optional Model Armor screening: set `MODEL_ARMOR_TEMPLATE` before `deploy.py`. The classifier
callbacks are the required-path validation either way.
