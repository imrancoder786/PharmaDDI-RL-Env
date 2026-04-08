# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the PharmaDDI Environment.

Endpoints:
    - POST /reset: Reset the environment with a new patient scenario
    - POST /step: Submit drug interaction analysis for grading
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from models import PharmaDDIAction, PharmaDDIObservation
    from .PharmaDDIEnv_environment import PharmaDDIEnvironment
except ModuleNotFoundError:
    from models import PharmaDDIAction, PharmaDDIObservation
    from server.PharmaDDIEnv_environment import PharmaDDIEnvironment


# Initialize a single global instance of the environment
# This ensures state is preserved across stateless HTTP calls
global_env = PharmaDDIEnvironment()

# Create the app with web interface and README integration
app = create_app(
    lambda: global_env,
    PharmaDDIAction,
    PharmaDDIObservation,
    env_name="PharmaDDIEnv",
    max_concurrent_envs=1,
)


def main():
    """Entry point for the environment server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
