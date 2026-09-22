from __future__ import annotations

from unittest.mock import Mock

import hpack
import hyperframe.frame
import pytest

import httpcore
from httpcore._sync.connection_pool import PoolRequest
from httpcore._sync.http_proxy import (
    ForwardHTTPConnection,
    TunnelHTTPConnection,
)
from httpcore._sync.socks_proxy import Socks5Connection

ORIGIN = httpcore.Origin(b"https", b"example.com", 443)
RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"


def idle_connection() -> httpcore.HTTP11Connection:
    connection = httpcore.HTTP11Connection(
        ORIGIN, httpcore.MockStream([RESPONSE] * 4)
    )
    response = connection.request("GET", "https://example.com/")
    assert response.content == b"OK"
    return connection


def pool_request(url: str = "https://example.com/") -> PoolRequest:
    return PoolRequest(httpcore.Request("GET", url))



@pytest.mark.parametrize("max_connections", [2, 100_000])
def test_reserves_idle_http11_within_pass(max_connections):
    with httpcore.ConnectionPool(max_connections=max_connections) as pool:
        connections: list[httpcore.ConnectionInterface] = [
            idle_connection(),
            idle_connection(),
        ]
        pool._connections = connections.copy()
        pool._requests = [pool_request(), pool_request()]

        assert pool._assign_requests_to_connections() == []
        assert [request.connection for request in pool._requests] == connections
        assert pool.connections == connections



def test_reserves_idle_http11_across_passes_and_releases():
    with httpcore.ConnectionPool(max_connections=1) as pool:
        connection = idle_connection()
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



def test_reserved_idle_connection_is_not_evicted():
    with httpcore.ConnectionPool(max_connections=1) as pool:
        connection = idle_connection()
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
        connection.close()
        assert waiting.connection is pool.connections[0]
        assert waiting.connection is not connection



def test_mixed_pool_evicts_only_unreserved_idle_surplus():
    with httpcore.ConnectionPool(max_connections=3) as pool:
        reserved, surplus, retained = [idle_connection() for _ in range(3)]
        pool._connections = [reserved, surplus, retained]
        owner = pool_request()
        pool._requests = [owner]
        pool._assign_requests_to_connections()
        pool._max_keepalive_connections = 1

        assert pool._assign_requests_to_connections() == [surplus]
        surplus.close()
        assert pool.connections == [reserved, retained]
        assert owner.connection is reserved
        assert pool._request_connections == {reserved: 1}

        pool._release_request_connection(owner)
        pool._requests.remove(owner)
        assert pool._assign_requests_to_connections() == [reserved]
        reserved.close()
        assert pool.connections == [retained]
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200 and response.content == b"OK"
        assert pool._requests == [] and pool._request_connections == {}



@pytest.mark.parametrize(
    "error_type", [httpcore.ReadError, httpcore.ConnectionNotAvailable]
)
def test_reservation_released_after_handler_error(error_type):
    class RejectOnce(httpcore.HTTP11Connection):
        reject = False

        def handle_request(self, request):
            if self.reject:
                self.reject = False
                raise error_type()
            return super().handle_request(request)

    connection = RejectOnce(ORIGIN, httpcore.MockStream([RESPONSE] * 4))
    connection.request("GET", "https://example.com/")
    connection.reject = True
    with httpcore.ConnectionPool(max_connections=1) as pool:
        pool._connections = [connection]
        if error_type is httpcore.ReadError:
            with pytest.raises(httpcore.ReadError):
                pool.request("GET", "https://example.com/")
        else:
            response = pool.request("GET", "https://example.com/")
            assert response.content == b"OK"
        assert pool._requests == []
        assert pool._request_connections == {}
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200 and response.content == b"OK"
        assert pool._requests == []
        assert pool._request_connections == {}



@pytest.mark.parametrize("http2", [False, True])
def test_speculative_multiplexing_and_http11_fallback(http2):
    with httpcore.ConnectionPool(max_connections=2, http2=http2) as pool:
        pool._requests = [pool_request(), pool_request()]
        pool._assign_requests_to_connections()
        assert len(pool.connections) == (1 if http2 else 2)
        connection = pool.connections[0]
        assert isinstance(connection, httpcore.HTTPConnection)
        assert connection._is_multiplexable() is http2

        # After a speculative H2 connection negotiates H1, reserve it singly.
        connection._connection = idle_connection()
        assert not connection._is_multiplexable()
        extra = pool_request()
        pool._requests.append(extra)
        pool._assign_requests_to_connections()
        assert extra.connection is not connection



def test_existing_http2_connection_remains_shared():
    with httpcore.ConnectionPool(max_connections=1, http2=True) as pool:
        connection = httpcore.HTTP2Connection(ORIGIN, httpcore.MockStream([]))
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
        # Releasing the same owner again must not consume the other reservation.
        pool._release_request_connection(owner)
        assert pool._request_connections == {connection: 1}
        assert pool._assign_requests_to_connections() == []
        pool._release_request_connection(waiting)
        pool._requests.remove(waiting)
        assert pool._request_connections == {}
        assert pool._assign_requests_to_connections() == [connection]
        connection.close()



@pytest.mark.parametrize(
    "wrapper_type", [ForwardHTTPConnection, TunnelHTTPConnection]
)
def test_proxy_reservation_capability_delegates(wrapper_type):
    wrapper = wrapper_type(
        proxy_origin=httpcore.Origin(b"http", b"proxy.example", 8080),
        remote_origin=ORIGIN,
    )
    assert not wrapper._is_multiplexable()
    wrapper._connection = idle_connection()
    assert not wrapper._is_multiplexable()
    wrapper.close()
    wrapper._connection = httpcore.HTTP2Connection(
        ORIGIN, httpcore.MockStream([])
    )
    assert wrapper._is_multiplexable()
    wrapper.close()



@pytest.mark.parametrize("http2", [False, True])
def test_socks_reservation_capability_delegates(http2):
    wrapper = Socks5Connection(
        proxy_origin=httpcore.Origin(b"http", b"proxy.example", 1080),
        remote_origin=ORIGIN,
        http2=http2,
    )
    assert wrapper._is_multiplexable() is http2
    wrapper._connection = idle_connection()
    assert not wrapper._is_multiplexable()
    wrapper.close()
    wrapper._connection = httpcore.HTTP2Connection(
        ORIGIN, httpcore.MockStream([])
    )
    assert wrapper._is_multiplexable()
    wrapper.close()



def test_reserved_connection_skips_expiry_until_release(monkeypatch):
    with httpcore.ConnectionPool(max_connections=1) as pool:
        connection = idle_connection()
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
        connection.close()



@pytest.mark.parametrize("transition", ["availability", "assignment"])
def test_reserved_connection_survives_state_transition(transition, monkeypatch):
    with httpcore.ConnectionPool(max_connections=1) as pool:
        connection = idle_connection()
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

            def assign(connection: httpcore.ConnectionInterface | None) -> None:
                original_assign(connection)
                activate()

            monkeypatch.setattr(owner, "assign_to_connection", assign)
        pool._assign_requests_to_connections()
        assert owner.connection is connection
        assert waiting.connection is None
        assert pool._request_connections == {connection: 1}



def test_reservation_released_when_response_close_trace_raises():
    error = RuntimeError("close callback failed")

    def trace(name: str, info: dict[str, object]) -> None:
        if name == "http11.response_closed.complete":
            raise error

    backend = httpcore.MockBackend([RESPONSE] * 2)
    with httpcore.ConnectionPool(
        max_connections=1, network_backend=backend
    ) as pool:
        response = pool.handle_request(
            httpcore.Request(
                "GET",
                "https://example.com/",
                headers={"Host": "example.com"},
                extensions={"trace": trace},
            )
        )
        assert response.status == 200 and response.read() == b"OK"
        with pytest.raises(RuntimeError) as caught:
            response.close()
        assert caught.value is error
        assert pool._requests == [] and pool._request_connections == {}
        assert pool.connections[0].is_idle()
        response.close()
        response = pool.request(
            "GET", "https://example.com/", extensions={"timeout": {"pool": 0}}
        )
        assert response.status == 200 and response.content == b"OK"
        assert pool._requests == [] and pool._request_connections == {}



def test_duplicate_http2_response_close_preserves_other_reservation():
    encoder = hpack.Encoder()
    buffer = [
        hyperframe.frame.SettingsFrame(
            settings={hyperframe.frame.SettingsFrame.MAX_CONCURRENT_STREAMS: 2}
        ).serialize()
    ]
    for stream_id in (1, 3):
        buffer.extend(
            [
                hyperframe.frame.HeadersFrame(
                    stream_id=stream_id,
                    data=encoder.encode([(b":status", b"200")]),
                    flags=["END_HEADERS"],
                ).serialize(),
                hyperframe.frame.DataFrame(
                    stream_id=stream_id, data=b"OK", flags=["END_STREAM"]
                ).serialize(),
            ]
        )
    backend = httpcore.MockBackend(buffer, http2=True)
    with httpcore.ConnectionPool(
        max_connections=1,
        max_keepalive_connections=0,
        http2=True,
        network_backend=backend,
    ) as pool:
        first = pool.handle_request(
            httpcore.Request(
                "GET", "https://example.com/", headers={"Host": "example.com"}
            )
        )
        assert first.status == 200 and first.read() == b"OK"
        second = pool.handle_request(
            httpcore.Request(
                "GET", "https://example.com/", headers={"Host": "example.com"}
            )
        )
        (connection,) = pool.connections
        assert pool._request_connections == {connection: 2}
        first.close()
        first.close()
        assert pool._request_connections == {connection: 1}
        assert pool.connections == [connection]
        assert not connection.is_closed()
        assert second.status == 200 and second.read() == b"OK"
        second.close()
        assert pool._requests == [] and pool._request_connections == {}
        assert connection.is_closed() and pool.connections == []
