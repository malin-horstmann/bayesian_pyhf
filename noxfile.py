import os
from pathlib import Path

import nox

# Default sessions to run if no session handles are passed
nox.options.sessions = ["lock"]


DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lock(session: nox.Session) -> None:
    """
    Build a lock file with conda-lock

    Examples:

        $ nox --session lock
    """
    image = "mambaorg/micromamba:1.3.0"
    session.run("docker", "pull", image, external=True)
    install_conda_lock_command = (
        "micromamba --quiet --yes install --channel conda-forge conda-lock"
    )
    build_lock_file_command = (
        "conda-lock lock --micromamba --file environment.yml --kind lock"
    )
    session.run(
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{DIR}:/build",
        "--workdir",
        "/build",
        image,
        "/bin/bash",
        "-c",
        f"{install_conda_lock_command} && {build_lock_file_command}",
        external=True,
    )
