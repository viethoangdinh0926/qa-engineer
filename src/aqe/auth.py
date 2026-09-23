"""Client-credentials HTTP clients for an OpenAI-compatible gateway."""

import httpx

from aqe.settings import get_settings


def build_http_clients(
    client_id: str,
    client_secret: str,
    verify: bool = True,
) -> tuple[httpx.Client, httpx.AsyncClient]:
    """Return HTTP clients that send a bearer token from the gateway."""
    settings = get_settings()
    token_url = settings.aia_gateway_token_url
    if not token_url:
        base = (settings.aia_gateway_base_url or "").rstrip("/")
        token_url = f"{base}/oauth/token"
    response = httpx.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        verify=verify,
        timeout=30,
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    return (
        httpx.Client(verify=verify, headers=headers, timeout=120),
        httpx.AsyncClient(verify=verify, headers=headers, timeout=120),
    )
