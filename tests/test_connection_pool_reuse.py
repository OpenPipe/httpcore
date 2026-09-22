from __future__ import annotations

import asyncio

import httpcore


def test_warm_http11_burst_does_not_retry_reserved_connections(monkeypatch):
    attempts = 0
    original = httpcore.AsyncHTTP11Connection.handle_async_request

    async def counted_request(
        self: httpcore.AsyncHTTP11Connection, request: httpcore.Request
    ) -> httpcore.Response:
        nonlocal attempts
        attempts += 1
        return await original(self, request)

    class CountingPool(httpcore.AsyncConnectionPool):
        assignment_passes = 0

        def _assign_requests_to_connections(
            self,
        ) -> list[httpcore.AsyncConnectionInterface]:
            self.assignment_passes += 1
            return super()._assign_requests_to_connections()

    monkeypatch.setattr(
        httpcore.AsyncHTTP11Connection, "handle_async_request", counted_request
    )

    async def run() -> None:
        nonlocal attempts
        count = 16
        backend = httpcore.AsyncMockBackend(
            [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"] * (2 * count)
        )
        pool = CountingPool(
            max_connections=100_000,
            max_keepalive_connections=100_000,
            network_backend=backend,
        )
        async with pool:

            async def fetch() -> None:
                response = await pool.request("GET", "http://example.com/")
                assert response.status == 200
                assert response.content == b"OK"

            await asyncio.gather(*(fetch() for _ in range(count)))
            assert len(pool.connections) == count
            assert all(connection.is_idle() for connection in pool.connections)

            attempts = pool.assignment_passes = 0
            await asyncio.gather(*(fetch() for _ in range(count)))

            assert attempts == count
            assert pool.assignment_passes == 2 * count
            assert len(pool.connections) == count
            assert pool._requests == []
            assert pool._request_connections == {}
        assert pool.connections == []

    asyncio.run(run())
