from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxInfo:
    sandbox_id: str
    user_id: str
    generation: str
    container_id: str
    proxy_id: str
    network_id: str
    egress_network_id: str
    sandbox_url: str
    relay_token: str = field(repr=False)
