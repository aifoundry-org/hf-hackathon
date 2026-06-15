#!/usr/bin/env python3

from pathlib import Path

import pytest

from test_support import DebugApi


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ep-library", action="store", required=True, help="Path to the GGONNX EP shared library")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: slower end-to-end tests that may download external models",
    )


@pytest.fixture(scope="session")
def ep_library(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.getoption("ep_library")).resolve()


@pytest.fixture(scope="session")
def debug_api(ep_library: Path) -> DebugApi:
    return DebugApi(ep_library)


@pytest.fixture(scope="session")
def suite_tmpdir(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp("ggonnx-test")
