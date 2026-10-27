"""Challenge 6: deploy the ReadyNow! Workflow to Agent Engine.

Run once, from an environment with a GCP project and ADC (Colab Enterprise in
the lab project, or anywhere `gcloud auth application-default login` has run):

    PROJECT_ID=... STAGING_BUCKET=gs://... uv run python -m challenges.challenge_6_readynow.deploy

Steps: init Vertex AI -> wrap the Workflow in AdkApp -> smoke-test it locally
-> deploy -> query the remote agent -> optionally delete it.

Optional Model Armor layer: set MODEL_ARMOR_TEMPLATE to a template's full
resource name (projects/.../locations/.../templates/...) and the plugin screens
every prompt before our own classifier callbacks run. Unset = skipped.
"""

import logging
import os

import cloudpickle  # Agent Engine uploads the agent as a cloudpickle blob
import vertexai
from vertexai import types
from vertexai.preview.reasoning_engines import AdkApp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from . import callbacks
from . import coordinator_agent
from . import critique_agent
from . import models
from . import qa_agent
from . import refine_agent
from . import routes_agent
from . import search_agent
from . import tools
from . import weather_agent

# Bundle our modules' source into the upload, otherwise the server can't import them.
for module in (
    callbacks, models, tools,
    weather_agent, search_agent, routes_agent, qa_agent,
    coordinator_agent, critique_agent, refine_agent,
):
    cloudpickle.register_pickle_by_value(module)

from .agent import root_agent

PROJECT_ID = os.environ["PROJECT_ID"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_LOCATION = os.environ.get("MODEL_LOCATION", "global")
MODEL_ARMOR_TEMPLATE = os.environ.get("MODEL_ARMOR_TEMPLATE")

# GOOGLE_CLOUD_PROJECT is set by Agent Engine itself and rejected if supplied.
ENGINE_ENV_VARS = {
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_CLOUD_LOCATION": MODEL_LOCATION,
}

REQUIREMENTS = ["google-cloud-aiplatform[agent_engines,adk]", "cloudpickle", "httpx"]

TEST_MESSAGES = [
    "What can you help me with?",
    "There's a hurricane warning for Miami. What's the weather and how do I get to Orlando?",
    "Write me a Python script to sort a list.",
]
USER_ID = "deploy-test-user"


def build_plugins() -> list:
    if not MODEL_ARMOR_TEMPLATE:
        return []
    from google.adk.integrations.model_armor import ModelArmorConfig, ModelArmorPlugin

    logger.info("Model Armor enabled with template %s", MODEL_ARMOR_TEMPLATE)
    return [
        ModelArmorPlugin(
            config=ModelArmorConfig(
                prompt_template_name=MODEL_ARMOR_TEMPLATE,
                response_template_name=MODEL_ARMOR_TEMPLATE,
            )
        )
    ]


def log_events(events) -> None:
    for event in events:
        author = event.get("author", "?")
        parts = (event.get("content") or {}).get("parts") or []
        text = parts[0].get("text") if parts else None
        if text:
            logger.info("[%s]\n%s", author, text)


def run_messages(query_fn) -> None:
    for message in TEST_MESSAGES:
        logger.info("=== user: %s", message)
        try:
            log_events(query_fn(user_id=USER_ID, message=message))
        except Exception:
            # Shared lab project hits Gemini quota (429) under class load; keep going.
            logger.exception("prompt failed; continuing")


def main() -> None:
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    plugins = build_plugins()
    app = AdkApp(agent=root_agent, plugins=plugins or None, env_vars=ENGINE_ENV_VARS)

    if os.environ.get("SKIP_LOCAL_TEST", "").lower() in ("1", "true"):
        logger.info("=== 1. local smoke test skipped (SKIP_LOCAL_TEST set)")
    else:
        logger.info("=== 1. local smoke test through AdkApp")
        run_messages(app.stream_query)

    logger.info("=== 2. deploying to Agent Engine (several minutes)")
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    requirements = REQUIREMENTS + (["google-cloud-modelarmor"] if plugins else [])
    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "display_name": "ReadyNow! (challenge 6)",
            "description": "FEMA emergency preparedness assistant with validate-and-refine pipeline.",
            "requirements": requirements,
            "staging_bucket": STAGING_BUCKET,
            "env_vars": ENGINE_ENV_VARS,
            "identity_type": types.IdentityType.AGENT_IDENTITY,
        },
    )
    logger.info("DEPLOYED: %s", remote_agent.api_resource.name)

    logger.info("=== 3. querying the deployed agent")
    run_messages(remote_agent.stream_query)

    if os.environ.get("DELETE_AFTER_TEST", "").lower() in ("1", "true"):
        logger.info("=== 4. deleting the deployed agent")
        remote_agent.delete(force=True)
    else:
        logger.info(
            "Left running. Delete with: remote_agent.delete(force=True)  # %s",
            remote_agent.api_resource.name,
        )


if __name__ == "__main__":
    main()
