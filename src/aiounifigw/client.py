"""High-level UniFi gateway client: endpoint methods returning typed models."""

from __future__ import annotations

from typing import Any

import aiohttp

from .auth import AbstractAuth
from .const import (
    DEFAULT_PORT,
    DEFAULT_SITE,
    DEFAULT_TIMEOUT,
    GATEWAY_TYPES,
    PATH_STAT_DEVICE,
    PATH_STAT_HEALTH,
    PATH_STAT_SYSINFO,
    PATH_SYSTEM,
    PATH_WAN_STATUS,
)
from .exceptions import GwApiError
from .models import Device, Health, SysInfo, SystemIdentity
from .transport import GatewayTransport


def _find_gateway(devices: list[Any]) -> dict[str, Any] | None:
    """Pick the gateway/console from a /stat/device device list.

    Prefers a device of a known gateway ``type``; falls back to any device that
    reports a ``wan1`` uplink (covers model codes not in the static type set).
    """
    fallback: dict[str, Any] | None = None
    for d in devices:
        if not isinstance(d, dict):
            continue
        if _str(d.get("type")) in GATEWAY_TYPES:
            return d
        if fallback is None and ("wan1" in d or "wan2" in d):
            fallback = d
    return fallback


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


class GatewayClient:
    """Read-only client for the UniFi Network local REST API.

    The caller owns *session*; the client never closes it.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        auth: AbstractAuth,
        *,
        site: str = DEFAULT_SITE,
        port: int = DEFAULT_PORT,
        use_ssl: bool = True,
        ssl: bool | aiohttp.Fingerprint = True,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._site = site
        self._transport = GatewayTransport(
            session,
            host,
            auth,
            port=port,
            use_ssl=use_ssl,
            ssl=ssl,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    @property
    def site(self) -> str:
        return self._site

    async def async_prepare(self) -> None:
        """Run the auth handshake once (optional; done lazily otherwise)."""
        await self._transport.async_prepare()

    async def get_identity(self) -> SystemIdentity:
        """Console identity (MAC / name / model) from /api/system."""
        return SystemIdentity.from_api(await self._transport.get_json(PATH_SYSTEM))

    async def get_device(self) -> Device:
        """Device telemetry with authoritative WAN selection when supported."""
        payload = await self._transport.get_json(PATH_STAT_DEVICE.format(site=self._site))
        devices = payload.get("data") if isinstance(payload, dict) else payload
        gateway = _find_gateway(devices if isinstance(devices, list) else [])
        if gateway is None:
            raise GwApiError("no gateway device found in /stat/device response")
        try:
            status = await self._transport.get_json(PATH_WAN_STATUS.format(site=self._site))
        except ValueError:
            raise GwApiError("invalid WAN status: malformed JSON") from None
        except GwApiError as err:
            if err.status not in (404, 405):
                raise
            # Older Network versions do not expose this endpoint. Do not cache
            # absence: a boot-time 404 or an upgrade can change it next poll.
            return Device.from_api(gateway)
        rows = status.get("wan_interfaces") if isinstance(status, dict) else None
        if not isinstance(rows, list):
            raise GwApiError("invalid WAN status: expected wan_interfaces list")
        seen: set[str] = set()
        active: set[str] = set()
        wan_keys = {"WAN": "wan1", **{f"WAN{i}": f"wan{i}" for i in range(2, 7)}}
        for row in rows:
            if not isinstance(row, dict):
                raise GwApiError("invalid WAN status: expected interface object")
            group, state = row.get("wan_networkgroup"), row.get("state")
            if (
                not isinstance(group, str)
                or group not in wan_keys
                or group in seen
                or not isinstance(state, str)
                or not state.strip()
            ):
                raise GwApiError("invalid WAN status: missing or ambiguous interface state")
            seen.add(group)
            if state == "ACTIVE":
                if not isinstance(gateway.get(wan_keys[group]), dict):
                    raise GwApiError(
                        "invalid WAN status: active interface absent from /stat/device"
                    )
                active.add(group)
        return Device.from_api(gateway, active_wans=frozenset(active))

    async def get_health(self) -> Health:
        """Subsystem health (WAN/ISP/www/vpn/lan/wlan) from /stat/health."""
        payload = await self._transport.get_json(PATH_STAT_HEALTH.format(site=self._site))
        data = payload.get("data") if isinstance(payload, dict) else payload
        return Health.from_api(data)

    async def get_sysinfo(self) -> SysInfo:
        """Controller/console info (Network version, updates) from /stat/sysinfo."""
        payload = await self._transport.get_json(PATH_STAT_SYSINFO.format(site=self._site))
        data = payload.get("data") if isinstance(payload, dict) else payload
        return SysInfo.from_api(data)

    async def close(self) -> None:
        """Release client resources.

        A no-op: the ``aiohttp`` session is owned by the caller and is never
        closed here. Present so consumers (CLI/MCP) can call it unconditionally.
        """
