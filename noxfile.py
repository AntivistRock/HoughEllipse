import nox
import tempfile

from nox.sessions import Session
from typing import Any

locations = "src", "tests", "noxfile.py", "docs/conf.py"
nox.options.sessions = "lint", "pytype", "tests"


@nox.session(python=["3.9"])
def tests(session: Session) -> None:
    """Run tests."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", "--cov")


def install_with_constraints(session, *args: str, **kwargs: Any):
    """Install necessary packages."""
    with tempfile.NamedTemporaryFile() as constraints:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=constraints.txt",
            "--without-hashes",
            f"--output={constraints.name}",
            external=True,
        )
        session.install(f"--constraint={constraints.name}", *args, **kwargs)


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run Black."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python=["3.9"])
def lint(session: Session) -> None:
    """Lint project."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.9")
def safety(session: Session) -> None:
    """Check project safety."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        install_with_constraints(session, "safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python="3.9")
def pytype(session: Session) -> None:
    """Run the static type checker."""
    args = session.posargs or ["--disable=import-error", *locations]
    install_with_constraints(session, "pytype")
    session.run("pytype", *args)


@nox.session(python="3.9")
def docs(session: Session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("poetry", "run", "sphinx-build", "docs", "docs/_build")
