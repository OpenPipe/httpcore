import pathlib
import runpy

import pytest


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "from httpcore._async.connection_pool import AsyncPoolRequest\n",
            "from httpcore._sync.connection_pool import PoolRequest\n",
        ),
        (
            "from httpcore._async.http_proxy import AsyncTunnelHTTPConnection\n",
            "from httpcore._sync.http_proxy import TunnelHTTPConnection\n",
        ),
    ],
)
def test_unasync_internal_import(source, expected):
    script = pathlib.Path(__file__).parents[1] / "scripts" / "unasync.py"
    assert runpy.run_path(str(script))["unasync_line"](source) == expected
