import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxInfo:
    sandbox_id: str
    sandbox_url: str
    container_name: str | None = None
    container_id: str | None = None
    created_at: float = field(default_factory=time.time)
    request_headers: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    requires_replacement: bool = field(default=False, repr=False, compare=False)
    user_id: str = ""
    generation: str = ""
    proxy_id: str = ""
    network_id: str = ""
    egress_network_id: str = ""
    relay_token: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict:
        return {"sandbox_id": self.sandbox_id, "sandbox_url": self.sandbox_url,
                "container_name": self.container_name, "container_id": self.container_id,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(sandbox_id=data["sandbox_id"], sandbox_url=data.get("sandbox_url", data.get("base_url", "")),
                   container_name=data.get("container_name"), container_id=data.get("container_id"),
                   created_at=data.get("created_at", time.time()))
