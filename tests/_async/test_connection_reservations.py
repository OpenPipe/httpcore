from __future__ import annotations

from unittest.mock import Mock

import pytest

import httpcore
from httpcore._async.connection_pool import AsyncPoolRequest
from httpcore._async.http_proxy import (
    AsyncForwardHTTPConnection,
    AsyncTunnelHTTPConnection,
)
from httpcore._async.socks_proxy import AsyncSocks5Connection

ORIGIN = httpcore.Origin(b"https", b"example.com", 443)
RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"


async def idle_connection() -> httpcore.AsyncHTTP11Connection:
    connection = httpcore.AsyncHTTP11Connection(
        ORIGIN, httpcore.AsyncMockStream([RESPONSE] * 4)
    )
    response = await connection.request("GET", "https://example.com/")
    assert response.content == b"OK"
    return connection


def pool_request(url: str = "https://example.com/") -> AsyncPoolRequest:
    return AsyncPoolRequest(httpcore.Request("GET", url))


@pytest.mark.anyio
@pytest.mark.parametrize("max_connections", [2, 100_000])
async def test_reserves_idle_http11_within_pass(max_connections):
    async with httpcore.AsyncConnectionPool(max_connections=max_connections) as pool:
        connections: list[httpcore.AsyncConnectionInterface] = [
            await idle_connection(),
            await idle_connection(),
        ]
        pool._connections = connections.copy()
        pool._requests = [pool_request(), pool_request()]

        assert pool._assign_requests_to_connections() == []
        assert [request.connection for request in pool._requests] == connections
        assert pool.connections == connections


@pytest.mark.anyio
async def test_reserves_idle_http11_across_passes_and_releases():
    async with httpcore.AsyncConnectionPool(max_connections=1) as pool:
        connection = await idle_connection()
        pool._connections = [connection]
        owner, waiting = pool_request(), pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        pool._requests.append(waiting)
        pool._assign_requests_to_connections()
        assert owner.connection is connection
        assert waiting.connection is None

        # Requeueing the owner releases the reservation but preserves FIFO order.
        pool._release_request_connection(owner)
        owner.clear_connection()
        pool._assign_requests_to_connections()
        assert owner.connection is connection
        assert waiting.connection is None

        # Error/cancellation/response closure all remove the owner from this list.
        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        pool._assign_requests_to_connections()
        assert waiting.connection is connection


@pytest.mark.anyio
async def test_reserved_idle_connection_is_not_evicted():
    async with httpcore.AsyncConnectionPool(max_connections=1) as pool:
        connection = await idle_connection()
        pool._connections = [connection]
        owner = pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        waiting = pool_request("https://other.example/")
        pool._requests.append(waiting)
        pool._max_keepalive_connections = 0

        assert pool._assign_requests_to_connections() == []
        assert owner.connection is connection
        assert waiting.connection is None
        assert pool.connections == [connection]

        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        assert pool._assign_requests_to_connections() == [connection]
        await connection.aclose()
        assert waiting.connection is pool.connections[0]
        assert waiting.connection is not connection


@pytest.mark.anyio
async def test_mixed_pool_evicts_only_unreserved_idle_surplus():
    async with httpcore.AsyncConnectionPool(max_connections=3) as pool:
        reserved, surplus, retained = [await idle_connection() for _ in range(3)]
        pool._connections = [reserved, surplus, retained]
        owner = pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        pool._max_keepalive_connections = 1

        assert pool._assign_requests_to_connections() == [surplus]
        await surplus.aclose()
        assert pool.connections == [reserved, retained]
        assert owner.connection is reserved
        assert pool._request_connections == {reserved: 1}

        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        assert pool._assign_requests_to_connections() == [reserved]
        await reserved.aclose()
        assert pool.connections == [retained]
        response = await pool.request("GET", "https://example.com/")
        assert response.status == 200 and response.content == b"OK"
        assert pool._requests == [] and pool._request_connections == {}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error_type", [httpcore.ReadError, httpcore.ConnectionNotAvailable]
)
async def test_reservation_released_after_handler_error(error_type):
    class RejectOnce(httpcore.AsyncHTTP11Connection):
        reject = False

        async def handle_async_request(self, request):
            if self.reject:
                self.reject = False
                raise error_type()
            return await super().handle_async_request(request)

    connection = RejectOnce(ORIGIN, httpcore.AsyncMockStream([RESPONSE] * 4))
    await connection.request("GET", "https://example.com/")
    connection.reject = True
    async with httpcore.AsyncConnectionPool(max_connections=1) as pool:
        pool._connections = [connection]
        if error_type is httpcore.ReadError:
            with pytest.raises(httpcore.ReadError):
                await pool.request("GET", "https://example.com/")
        else:
            response = await pool.request("GET", "https://example.com/")
            assert response.content == b"OK"
        assert pool._requests == []
        assert pool._request_connections == {}
        response = await pool.request("GET", "https://example.com/")
        assert response.status == 200 and response.content == b"OK"
        assert pool._requests == []
        assert pool._request_connections == {}


@pytest.mark.anyio
@pytest.mark.parametrize("http2", [False, True])
async def test_speculative_multiplexing_and_http11_fallback(http2):
    async with httpcore.AsyncConnectionPool(max_connections=2, http2=http2) as pool:
        pool._requests = [pool_request(), pool_request()]
        pool._assign_requests_to_connections()
        assert len(pool.connections) == (1 if http2 else 2)
        connection = pool.connections[0]
        assert isinstance(connection, httpcore.AsyncHTTPConnection)
        assert connection._is_multiplexable() is http2

        # After a speculative H2 connection negotiates H1, reserve it singly.
        connection._connection = await idle_connection()
        assert not connection._is_multiplexable()
        extra = pool_request()
        pool._requests.append(extra)
        pool._assign_requests_to_connections()
        assert extra.connection is not connection


@pytest.mark.anyio
async def test_existing_http2_connection_remains_shared():
    async with httpcore.AsyncConnectionPool(max_connections=1, http2=True) as pool:
        connection = httpcore.AsyncHTTP2Connection(ORIGIN, httpcore.AsyncMockStream([]))
        pool._connections = [connection]
        owner, waiting = pool_request(), pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        pool._requests.append(waiting)
        pool._assign_requests_to_connections()
        assert owner.connection is waiting.connection is connection
        assert pool._request_connections == {connection: 2}
        pool._max_keepalive_connections = 0
        assert pool._assign_requests_to_connections() == []
        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        assert pool._request_connections == {connection: 1}
        assert pool._assign_requests_to_connections() == []
        pool._release_request_connection(waiting)
        pool._requests.remove(waiting)
        assert pool._request_connections == {}
        assert pool._assign_requests_to_connections() == [connection]
        await connection.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "wrapper_type", [AsyncForwardHTTPConnection, AsyncTunnelHTTPConnection]
)
async def test_proxy_reservation_capability_delegates(wrapper_type):
    wrapper = wrapper_type(
        proxy_origin=httpcore.Origin(b"http", b"proxy.example", 8080),
        remote_origin=ORIGIN,
    )
    assert not wrapper._is_multiplexable()
    wrapper._connection = await idle_connection()
    assert not wrapper._is_multiplexable()
    await wrapper.aclose()
    wrapper._connection = httpcore.AsyncHTTP2Connection(
        ORIGIN, httpcore.AsyncMockStream([])
    )
    assert wrapper._is_multiplexable()
    await wrapper.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("http2", [False, True])
async def test_socks_reservation_capability_delegates(http2):
    wrapper = AsyncSocks5Connection(
        proxy_origin=httpcore.Origin(b"http", b"proxy.example", 1080),
        remote_origin=ORIGIN,
        http2=http2,
    )
    assert wrapper._is_multiplexable() is http2
    wrapper._connection = await idle_connection()
    assert not wrapper._is_multiplexable()
    await wrapper.aclose()
    wrapper._connection = httpcore.AsyncHTTP2Connection(
        ORIGIN, httpcore.AsyncMockStream([])
    )
    assert wrapper._is_multiplexable()
    await wrapper.aclose()


@pytest.mark.anyio
async def test_reserved_connection_skips_expiry_until_release(monkeypatch):
    async with httpcore.AsyncConnectionPool(max_connections=1) as pool:
        connection = await idle_connection()
        pool._connections = [connection]
        owner = pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        has_expired = Mock(return_value=True)
        monkeypatch.setattr(connection, "has_expired", has_expired)
        assert pool._assign_requests_to_connections() == []
        has_expired.assert_not_called()
        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        assert pool._assign_requests_to_connections() == [connection]
        has_expired.assert_called_once()
        await connection.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("transition", ["availability", "assignment"])
async def test_reserved_connection_survives_state_transition(transition, monkeypatch):
    async with httpcore.AsyncConnectionPool(max_connections=1) as pool:
        connection = await idle_connection()
        pool._connections = [connection]
        owner, waiting = pool_request(), pool_request()
        pool._requests = [owner, waiting]

        def activate() -> bool:
            monkeypatch.setattr(connection, "is_idle", lambda: False)
            return True

        if transition == "availability":
            pool._reserve_connection(owner, connection)
            monkeypatch.setattr(connection, "is_available", activate)
        else:
            original_assign = owner.assign_to_connection

            def assign(connection: httpcore.AsyncConnectionInterface | None) -> None:
                original_assign(connection)
                activate()

            monkeypatch.setattr(owner, "assign_to_connection", assign)
        pool._assign_requests_to_connections()
        assert owner.connection is connection
        assert waiting.connection is None
        assert pool._request_connections == {connection: 1}
