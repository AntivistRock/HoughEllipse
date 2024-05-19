import click.testing
import pytest

from hough.main import main


@pytest.fixture
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()


def test_main_succeeds(runner: click.testing.CliRunner):
    result = runner.invoke(main)
    assert result.exit_code == 0
