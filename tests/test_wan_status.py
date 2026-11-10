"""WAN selection through the real client and a loopback-only HTTP gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from aiounifigw import ApiKeyAuth, GatewayClient
from aiounifigw.exceptions import GwApiError, GwAuthError, GwCapabilityError, GwConnectionError
from aiounifigw.models import Device

SITE = "branch"
DEVICE_PATH = f"/proxy/network/api/s/{SITE}/stat/device"
STATUS_PATH = f"/proxy/network/v2/api/site/{SITE}/wan/load-balancing/status"


class Gateway:
    """Synthetic UDM-Pro: the GRE backup has no legacy is_uplink flag."""

    def __init__(self) -> None:
        self.device: dict[str, Any] = {
            "type": "udm",
            "model": "UDMPRO",
            "wan1": {"ifname": "eth9", "up": True, "is_uplink": True},
            "wan3": {"ifname": "gre1", "up": True},
        }
        self.payload: Any = self.status("WAN")
        self.http_status = 200
        self.mode = "json"
        self.requests: list[tuple[str, str]] = []

    @staticmethod
    def status(active: str) -> dict[str, Any]:
        return {
            "wan_interfaces": [
                {
                    "name": "Example 5G",
                    "state": "ACTIVE" if active == "WAN3" else "BACKUP",
                    "wan_networkgroup": "WAN3",
                },
                {"name": "Example standby", "state": "BACKUP", "wan_networkgroup": "WAN2"},
                {
                    "name": "Example wired",
                    "state": "ACTIVE" if active == "WAN" else "BACKUP",
                    "wan_networkgroup": "WAN",
                },
            ]
        }

    async def handle(self, request: web.Request) -> web.Response:
        self.requests.append((request.method, request.path))
        if request.path == DEVICE_PATH:
            return web.json_response({"data": [self.device]})
        if request.path == STATUS_PATH:
            if self.mode == "disconnect":
                assert request.transport is not None
                request.transport.close()
            if self.mode == "html":
                return web.Response(text="<html>Starting</html>", content_type="text/html")
            if self.mode == "invalid_json":
                return web.Response(text="{", content_type="application/json")
            if self.http_status == 302:
                raise web.HTTPFound("/manage")
            return web.json_response(self.payload, status=self.http_status)
        raise web.HTTPNotFound


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[Gateway, GatewayClient]]:
    fake = Gateway()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            yield (
                fake,
                GatewayClient(
                    session,
                    "127.0.0.1",
                    ApiKeyAuth("synthetic-test-key"),
                    site=SITE,
                    port=port,
                    use_ssl=False,
                ),
            )
    finally:
        await runner.cleanup()


async def test_5g_failover_and_recovery(gateway: tuple[Gateway, GatewayClient]) -> None:
    fake, client = gateway
    for active in ("WAN", "WAN3", "WAN"):
        fake.payload = fake.status(active)
        device = await client.get_device()
        assert device.active_wan is not None
        assert device.active_wan.id == active
        assert device.failover_active is (active != "WAN")
        assert {w.id: w.is_active for w in device.wans} == {
            "WAN": active == "WAN",
            "WAN3": active == "WAN3",
        }
        assert device.wans[1].ifname == "gre1"
    assert fake.requests == [("GET", p) for _ in range(3) for p in (DEVICE_PATH, STATUS_PATH)]


async def test_explicit_primary_does_not_need_a_legacy_flag(
    gateway: tuple[Gateway, GatewayClient],
) -> None:
    fake, client = gateway
    fake.device["wan1"].pop("is_uplink")
    device = await client.get_device()
    assert device.active_wan is not None
    assert device.active_wan.id == "WAN"
    assert device.failover_active is False


async def test_multiple_active_groups_preserve_first_wan_semantics(
    gateway: tuple[Gateway, GatewayClient],
) -> None:
    fake, client = gateway
    fake.payload["wan_interfaces"][0]["state"] = "ACTIVE"
    device = await client.get_device()
    assert [w.id for w in device.wans if w.is_active] == ["WAN", "WAN3"]
    assert device.active_wan is not None
    assert device.active_wan.id == "WAN"
    assert device.failover_active is False


@pytest.mark.parametrize("http_status", [404, 405])
async def test_unsupported_status_preserves_legacy_selection(
    gateway: tuple[Gateway, GatewayClient],
    http_status: int,
) -> None:
    fake, client = gateway
    fake.http_status = http_status
    device = await client.get_device()
    assert device.active_wan is not None
    assert device.active_wan.id == "WAN"
    assert device.failover_active is False
    # Re-probe next poll: a transient 404 during boot must not disable it forever.
    fake.http_status = 200
    fake.payload = fake.status("WAN3")
    assert (await client.get_device()).failover_active is True


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"wan_interfaces": None},
        {"wan_interfaces": {}},
        {"wan_interfaces": [None]},
        {"wan_interfaces": [{}]},
        {"wan_interfaces": [{"state": "ACTIVE"}]},
        {"wan_interfaces": [{"wan_networkgroup": "WAN"}]},
        {"wan_interfaces": [{"wan_networkgroup": "WAN", "state": None}]},
        {"wan_interfaces": [{"wan_networkgroup": "WAN", "state": ""}]},
        {"wan_interfaces": [{"wan_networkgroup": ["WAN"], "state": "ACTIVE"}]},
        {"wan_interfaces": [{"wan_networkgroup": "WAN1", "state": "ACTIVE"}]},
        {"wan_interfaces": [{"wan_networkgroup": "WAN2", "state": "ACTIVE"}]},
        {
            "wan_interfaces": [
                {"wan_networkgroup": "WAN", "state": "ACTIVE"},
                {"wan_networkgroup": "WAN", "state": "BACKUP"},
            ]
        },
    ],
)
async def test_malformed_status_is_not_healthy_legacy_data(
    gateway: tuple[Gateway, GatewayClient],
    payload: Any,
) -> None:
    fake, client = gateway
    fake.payload = payload
    with pytest.raises(GwApiError, match="WAN status"):
        await client.get_device()


@pytest.mark.parametrize("payload", [{"wan_interfaces": []}, Gateway.status("")])
async def test_no_active_is_not_primary_healthy(
    gateway: tuple[Gateway, GatewayClient],
    payload: Any,
) -> None:
    fake, client = gateway
    fake.payload = payload
    device = await client.get_device()
    assert device.active_wan is None
    assert not any(w.is_active for w in device.wans)
    assert device.failover_active is None


def test_legacy_missing_selection_is_unknown() -> None:
    device = Device.from_api({"wan1": {"up": True}, "wan3": {"up": True}})
    assert device.active_wan is None
    assert device.failover_active is None


@pytest.mark.parametrize(
    ("http_status", "mode", "error"),
    [
        (401, "json", GwAuthError),
        (403, "json", GwCapabilityError),
        (429, "json", GwApiError),
        (500, "json", GwApiError),
        (302, "json", GwApiError),
        (200, "html", GwApiError),
        (200, "invalid_json", GwApiError),
        (200, "disconnect", GwConnectionError),
    ],
)
async def test_status_failure_never_falls_back_to_healthy_legacy(
    gateway: tuple[Gateway, GatewayClient],
    http_status: int,
    mode: str,
    error: type[Exception],
) -> None:
    fake, client = gateway
    fake.http_status, fake.mode = http_status, mode
    with pytest.raises(error):
        await client.get_device()
    # No stale/cached selection survives a failed poll.
    fake.http_status, fake.mode, fake.payload = 200, "json", fake.status("WAN3")
    assert (await client.get_device()).failover_active is True
