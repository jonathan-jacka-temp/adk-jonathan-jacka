"""Challenge 5: deploy the challenge-4 answer_team Workflow to Agent Engine.

Run once, from an environment with a GCP project and ADC (Colab Enterprise in
the lab project, or anywhere `gcloud auth application-default login` has run):

    PROJECT_ID=... STAGING_BUCKET=gs://... uv run python -m challenges.challenge_5_deploy.deploy

Steps: init Vertex AI -> wrap the Workflow in AdkApp -> smoke-test it locally
-> deploy -> query the remote agent -> optionally delete it.

Nothing here runs under `adk run`; agent.py stays the ADK entrypoint.
"""

import logging
import os

import cloudpickle  # See below re: bundling our modules' source into the upload
import vertexai
from vertexai import types
from vertexai.preview.reasoning_engines import AdkApp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from . import callbacks
from . import critique_agent
from . import greeter_agent
from . import models
from . import refine_agent
from . import search_agent

# Bundle our modules' source into the upload, otherwise the server can't import them.
# Alternatively could have flattened out but wanted to keep modules separate.
for module in (callbacks, models, greeter_agent, search_agent, critique_agent, refine_agent):
    cloudpickle.register_pickle_by_value(module)

from .agent import root_agent

PROJECT_ID = os.environ["PROJECT_ID"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_LOCATION = os.environ.get("MODEL_LOCATION", "global")

# GOOGLE_CLOUD_PROJECT is set by Agent Engine itself and rejected if supplied.
ENGINE_ENV_VARS = {
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_CLOUD_LOCATION": MODEL_LOCATION,
}

TEST_QUESTION = "Explain what the Google ADK is and when I'd use it."
USER_ID = "deploy-test-user"


def log_events(events) -> None:
    for event in events:
        author = event.get("author", "?")
        parts = (event.get("content") or {}).get("parts") or []
        text = parts[0].get("text") if parts else None
        if text:
            logger.info("[%s]\n%s", author, text)


def main() -> None:
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    app = AdkApp(agent=root_agent, env_vars=ENGINE_ENV_VARS)

    if os.environ.get("SKIP_LOCAL_TEST", "").lower() in ("1", "true"):
        logger.info("=== 1. local smoke test skipped (SKIP_LOCAL_TEST set)")
    else:
        logger.info("=== 1. local smoke test through AdkApp")
        log_events(app.stream_query(user_id=USER_ID, message=TEST_QUESTION))

    logger.info("=== 2. deploying to Agent Engine (several minutes)")
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "display_name": "answer_team (challenge 4 workflow)",
            "description": "Greeter -> Search -> Critique -> Refine pipeline.",
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]", "cloudpickle"],
            "staging_bucket": STAGING_BUCKET,
            "env_vars": ENGINE_ENV_VARS,
            "identity_type": types.IdentityType.AGENT_IDENTITY,
        },
    )
    logger.info("DEPLOYED: %s", remote_agent.api_resource.name)

    logger.info("=== 3. querying the deployed agent")
    log_events(remote_agent.stream_query(user_id=USER_ID, message=TEST_QUESTION))

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
