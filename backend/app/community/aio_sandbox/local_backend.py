import csv
import hashlib
import io
import ipaddress
import json
import os
import re
import secrets
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.community.aio_sandbox.backend import SandboxBackend
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.config import DEFAULT_SANDBOX_ALLOW_DOMAINS
from app.user_context import current_user


AIO_IMAGE = "enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox@sha256:6328d7fd2f0ff0b4c147c3d05b3df1ce331f4a482eb6e550ecd64ed1fcf906e7"
PROXY_IMAGE = "ghcr.io/bytedance/deer-flow-sandbox-network-proxy@sha256:ab8e86073cfef47f72b5b07ca3e92a8c9f8ff5b50328c16436f21d35ab359a40"
_LABEL = "openworkbot.sandbox"
_DESKTOP_PROXY = """import runpy,socket,sys
original=socket.getaddrinfo
def ipv4(host,port,family=0,type=0,proto=0,flags=0):
    return original(host,port,socket.AF_INET if family==socket.AF_UNSPEC else family,type,proto,flags)
socket.getaddrinfo=ipv4
sys.argv=['/opt/deerflow-network-proxy.py','serve']
runpy.run_path(sys.argv[0],run_name='__main__')
"""


def normalize_allow_domains(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    normalized = []
    for item in value.split(","):
        raw = item.strip().lower().rstrip(".")
        suffix = raw[2:] if raw.startswith("*.") else raw
        try:
            domain = suffix.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("沙箱 allowlist 域名无效") from exc
        labels = domain.split(".")
        if (len(labels) < 2 or len(domain) > 253 or any(not label or len(label) > 63
                or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in labels)):
            raise ValueError("沙箱 allowlist 域名无效")
        try:
            ipaddress.ip_address(domain)
        except ValueError:
            pass
        else:
            raise ValueError("沙箱 allowlist 不能包含 IP 地址")
        rule = ("*." if raw.startswith("*.") else "") + domain
        if rule not in normalized:
            normalized.append(rule)
    return tuple(normalized)


class DockerTimeout(RuntimeError):
    pass


class SandboxCleanupError(RuntimeError):
    def __init__(self, sandbox_id: str, user_id: str, generation: str) -> None:
        super().__init__("沙箱资源状态无法确认，已停止复用")
        self.sandbox_id = sandbox_id
        self.user_id = user_id
        self.generation = generation


class LocalContainerBackend(SandboxBackend):
    def __init__(self, data_dir: Path, workspaces_dir: Path, skills_dir: Path,
                 allow_domains: str = DEFAULT_SANDBOX_ALLOW_DOMAINS) -> None:
        root = os.path.normcase(str(data_dir.resolve()))
        self.instance = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
        self.workspaces_dir = workspaces_dir.resolve()
        self.skills_dir = skills_dir.resolve()
        self.allow_domains = normalize_allow_domains(allow_domains)

    def _docker(self, *args: str, timeout: int = 30, env: dict | None = None, input_text: str | None = None) -> str:
        try:
            result = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=timeout, env=env, input=input_text)
        except subprocess.TimeoutExpired as exc:
            raise DockerTimeout(f"Docker 操作超时：{args[0]}") from exc
        except OSError as exc:
            raise RuntimeError("Docker 不可用") from exc
        if result.returncode:
            raise RuntimeError(f"Docker 操作失败（{args[0]}）：{result.stderr.strip()[:500]}")
        return result.stdout.strip()

    def _inspect(self, kind: str, resource: str) -> dict | None:
        try:
            raw = self._docker(kind, "inspect", resource)
        except RuntimeError as exc:
            if not isinstance(exc, DockerTimeout) and re.search(
                    r"no such (object|container|network|image)|network \S+ not found", str(exc), re.I):
                return None
            raise
        result = json.loads(raw)
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
            raise RuntimeError("Docker 返回无效资源信息")
        return result[0]

    def _labels(self, sandbox_id: str, user_id: str, generation: str, role: str) -> dict[str, str]:
        return {f"{_LABEL}.instance": self.instance, f"{_LABEL}.id": sandbox_id,
                f"{_LABEL}.user": user_id, f"{_LABEL}.generation": generation,
                f"{_LABEL}.role": role}

    @staticmethod
    def _label_args(labels: dict[str, str]) -> list[str]:
        return [part for key, value in labels.items() for part in ("--label", f"{key}={value}")]

    @staticmethod
    def _mount(source: Path, target: str, readonly: bool = False) -> str:
        output = io.StringIO(newline="")
        parts = ["type=bind", f"src={source}", f"dst={target}"]
        if readonly:
            parts.append("readonly")
        csv.writer(output, lineterminator="").writerow(parts)
        return output.getvalue()

    def _workspace(self, user_id: str) -> Path:
        if not re.fullmatch(r"(?:[0-9a-f]{32}|default)", user_id):
            raise ValueError("用户身份无效")
        target = self.workspaces_dir / user_id
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise ValueError("用户工作区不能是链接")
        if target.resolve().parent != self.workspaces_dir:
            raise ValueError("用户工作区路径无效")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _check_engine(self) -> bool:
        version = self._docker("version", "--format", "{{.Server.Version}}")
        try:
            if int(version.split(".", 1)[0]) < 28:
                raise RuntimeError("沙箱网络需要 Docker Engine 28 或更新版本")
        except ValueError as exc:
            raise RuntimeError("无法确认 Docker Engine 版本") from exc
        system = self._docker("info", "--format", "{{.OperatingSystem}}")
        return "docker desktop" in system.lower()

    def _ensure_image(self, image: str) -> None:
        if self._inspect("image", image) is None:
            self._docker("pull", image, timeout=600)

    def create(self, thread_id: str | None, sandbox_id: str, extra_mounts=None,
               *, user_id: str | None = None) -> SandboxInfo:
        user_id = user_id or current_user() or "default"
        if (current_user() and user_id != current_user()) or (not current_user() and user_id != "default"):
            raise PermissionError("沙箱不属于当前用户")
        if not re.fullmatch(r"[0-9a-f]{32}", sandbox_id):
            raise ValueError("沙箱身份无效")
        if extra_mounts:
            raise ValueError("不支持额外挂载")
        workspace = self._workspace(user_id)
        if not self.skills_dir.is_dir():
            raise RuntimeError("技能目录不存在")
        process_helper = Path(__file__).with_name("process_cleanup.py").resolve()
        if not process_helper.is_file():
            raise RuntimeError("沙箱受控程序不存在")
        desktop = self._check_engine()
        self._ensure_image(AIO_IMAGE)
        self._ensure_image(PROXY_IMAGE)
        generation = uuid.uuid4().hex
        prefix = f"owb-{self.instance[:8]}-{sandbox_id[:12]}-{generation[:8]}"
        names = {"sandbox": prefix, "proxy": f"{prefix}-proxy",
                 "network": f"{prefix}-internal", "egress": f"{prefix}-egress"}
        labels = {role: self._labels(sandbox_id, user_id, generation, role) for role in names}
        token = secrets.token_urlsafe(32)
        env = dict(os.environ, DEERFLOW_RELAY_TOKEN=token)
        proxy_url = f"http://{names['proxy']}:3128"
        try:
            network = self._docker("network", "create", "--driver", "bridge", "--internal", "--opt",
                                   "com.docker.network.bridge.gateway_mode_ipv4=isolated", "--opt",
                                   "com.docker.network.bridge.gateway_mode_ipv6=isolated",
                                   *self._label_args(labels["network"]), names["network"])
            egress = self._docker("network", "create", "--driver", "bridge", "--opt",
                                  "com.docker.network.bridge.enable_icc=false",
                                  *self._label_args(labels["egress"]), names["egress"])
            proxy = self._docker("create", "--rm", "--name", names["proxy"],
                                 *self._label_args(labels["proxy"]), "--network", names["egress"],
                                 "-p", "127.0.0.1::8080", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
                                 "--memory", "256m", "--cpus", "1", "--pids-limit", "128", "--read-only",
                                 "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--user", "65532:65532",
                                 "-e", "DEERFLOW_NETWORK_MODE=allowlist", "-e",
                                 f"DEERFLOW_ALLOW_DOMAINS_JSON={json.dumps(self.allow_domains)}", "-e",
                                 f"DEERFLOW_SANDBOX_TARGET={names['sandbox']}:8080", "-e",
                                 f"DEERFLOW_ALLOW_SYNTHETIC_DNS={int(desktop)}", "-e", "DEERFLOW_RECORD_DENIALS=0",
                                 "-e", "DEERFLOW_RELAY_TOKEN", PROXY_IMAGE,
                                 *(["python", "-c", _DESKTOP_PROXY] if desktop else []), env=env)
            self._docker("network", "connect", names["network"], proxy)
            self._docker("start", proxy)
            port = self._docker("port", proxy, "8080/tcp")
            match = re.fullmatch(r"127\.0\.0\.1:(\d+)", port)
            if not match or not 0 < int(match[1]) < 65536:
                raise RuntimeError("代理端口不是本机回环地址")
            environment = {key: proxy_url for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                                                         "http_proxy", "https_proxy", "all_proxy")}
            environment.update({key: "localhost,127.0.0.1,::1" for key in ("NO_PROXY", "no_proxy", "PROXY_EXCLUDE")})
            environment["PROXY_SERVER"] = f"{names['proxy']}:3128"
            environment["MAX_SHELL_SESSIONS"] = "12"
            environment["AIO_BASH_MAX_SESSIONS"] = "12"
            sandbox = self._docker("create", "--rm", "--name", names["sandbox"],
                                   *self._label_args(labels["sandbox"]), "--network", names["network"],
                                   "--cap-drop=ALL", "--cap-add=CHOWN", "--cap-add=FOWNER", "--cap-add=SETUID",
                                   "--cap-add=SETGID", "--cap-add=DAC_OVERRIDE", "--security-opt", "no-new-privileges",
                                   "--security-opt", "seccomp=unconfined", "--memory", "1g", "--cpus", "1",
                                   "--pids-limit", "512", "--mount", self._mount(workspace, "/mnt/user-data/workspace"),
                                   "--mount", self._mount(self.skills_dir, "/mnt/skills", True),
                                   "--mount", self._mount(process_helper, "/opt/openworkbot-process-cleanup.py", True),
                                   *(part for key, value in environment.items() for part in ("-e", f"{key}={value}")),
                                   AIO_IMAGE)
            self._docker("start", sandbox)
            info = SandboxInfo(sandbox_id=sandbox_id, sandbox_url=f"http://127.0.0.1:{match[1]}",
                               container_name=names["sandbox"], container_id=sandbox, user_id=user_id,
                               generation=generation, proxy_id=proxy, network_id=network, egress_network_id=egress,
                               relay_token=token, request_headers={"X-DeerFlow-Relay-Token": token})
            deadline = time.monotonic() + 60
            with httpx.Client(trust_env=False, headers={"X-DeerFlow-Relay-Token": token}) as client:
                while time.monotonic() < deadline:
                    try:
                        if client.get(f"{info.sandbox_url}/v1/sandbox", timeout=3).status_code == 200:
                            return info
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
            raise RuntimeError("AIO 沙箱启动超时")
        except BaseException as exc:
            try:
                self._cleanup(names, labels)
            except Exception as cleanup_error:
                raise SandboxCleanupError(sandbox_id, user_id, generation) from cleanup_error
            if isinstance(exc, DockerTimeout):
                raise SandboxCleanupError(sandbox_id, user_id, generation) from exc
            raise

    def _cleanup(self, resources: dict[str, str], labels: dict[str, dict[str, str]]) -> None:
        for role in ("sandbox", "proxy", "network", "egress"):
            kind = "container" if role in {"sandbox", "proxy"} else "network"
            resource = self._inspect(kind, resources[role])
            if resource is None:
                continue
            actual = (resource.get("Config") or {}).get("Labels") if kind == "container" else resource.get("Labels")
            if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in labels[role].items()):
                raise RuntimeError("Docker 资源归属无法确认，停止清理")
            resource_id = resource.get("Id") or resource.get("ID")
            if not isinstance(resource_id, str) or not re.fullmatch(r"[0-9a-f]{64}", resource_id):
                raise RuntimeError("Docker 资源 ID 无效")
            self._docker(kind, "rm", *( ["-f"] if kind == "container" else []), resource_id)
            if self._inspect(kind, resource_id) is not None:
                raise RuntimeError("Docker 资源未被清理")

    def destroy(self, info: SandboxInfo) -> None:
        resources = {"sandbox": info.container_id, "proxy": info.proxy_id,
                     "network": info.network_id, "egress": info.egress_network_id}
        labels = {role: self._labels(info.sandbox_id, info.user_id, info.generation, role) for role in resources}
        self._cleanup(resources, labels)

    def is_alive(self, info: SandboxInfo) -> bool:
        containers = {}
        for role, identity in (("sandbox", info.container_id), ("proxy", info.proxy_id)):
            resource = self._inspect("container", identity)
            labels = (resource or {}).get("Config", {}).get("Labels") or {}
            expected = self._labels(info.sandbox_id, info.user_id, info.generation, role)
            if not resource or not resource.get("State", {}).get("Running") or any(
                    labels.get(key) != value for key, value in expected.items()):
                return False
            containers[role] = resource
        for role, identity in (("network", info.network_id), ("egress", info.egress_network_id)):
            resource = self._inspect("network", identity)
            labels = (resource or {}).get("Labels") or {}
            expected = self._labels(info.sandbox_id, info.user_id, info.generation, role)
            if (not resource or resource.get("Driver") != "bridge"
                    or any(labels.get(key) != value for key, value in expected.items())):
                return False
            options = resource.get("Options") or {}
            if role == "network" and (not resource.get("Internal") or
                    options.get("com.docker.network.bridge.gateway_mode_ipv4") != "isolated" or
                    options.get("com.docker.network.bridge.gateway_mode_ipv6") != "isolated"):
                return False
            if role == "egress" and (resource.get("Internal") or
                    options.get("com.docker.network.bridge.enable_icc") != "false"):
                return False
        sandbox_networks = (containers["sandbox"].get("NetworkSettings") or {}).get("Networks") or {}
        proxy_networks = (containers["proxy"].get("NetworkSettings") or {}).get("Networks") or {}
        if {value.get("NetworkID") for value in sandbox_networks.values()} != {info.network_id}:
            return False
        if {value.get("NetworkID") for value in proxy_networks.values()} != {info.network_id, info.egress_network_id}:
            return False
        ports = (containers["proxy"].get("NetworkSettings") or {}).get("Ports") or {}
        bindings = ports.get("8080/tcp") or []
        target = urlparse(info.sandbox_url)
        if target.hostname != "127.0.0.1" or len(bindings) != 1 or bindings[0] != {
                "HostIp": "127.0.0.1", "HostPort": str(target.port)}:
            return False
        return True

    def _resources(self, sandbox_id: str) -> dict[str, dict]:
        if not re.fullmatch(r"[0-9a-f]{32}", sandbox_id):
            raise ValueError("沙箱身份无效")
        filters = ("--filter", f"label={_LABEL}.instance={self.instance}",
                   "--filter", f"label={_LABEL}.id={sandbox_id}")
        groups = (("container", self._docker("ps", "-aq", *filters).splitlines()),
                  ("network", self._docker("network", "ls", "-q", *filters).splitlines()))
        resources = {}
        for kind, identities in groups:
            for identity in identities:
                resource = self._inspect(kind, identity)
                if resource is None:
                    raise RuntimeError("Docker 资源状态正在变化")
                labels = (resource.get("Config") or {}).get("Labels") if kind == "container" else resource.get("Labels")
                role = (labels or {}).get(f"{_LABEL}.role")
                allowed = {"sandbox", "proxy"} if kind == "container" else {"network", "egress"}
                user = (labels or {}).get(f"{_LABEL}.user", "")
                generation = (labels or {}).get(f"{_LABEL}.generation", "")
                if (role not in allowed or role in resources or not re.fullmatch(r"(?:[0-9a-f]{32}|default)", user)
                        or not re.fullmatch(r"[0-9a-f]{32}", generation)
                        or any(labels.get(key) != value for key, value in
                               self._labels(sandbox_id, user, generation, role).items())):
                    raise RuntimeError("Docker 资源归属无法确认")
                resources[role] = resource
        return resources

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        resources = self._resources(sandbox_id)
        sandbox = resources.get("sandbox")
        if not sandbox or not sandbox.get("State", {}).get("Running"):
            return None
        if set(resources) != {"sandbox", "proxy", "network", "egress"}:
            raise RuntimeError("沙箱资源不完整")
        labels = sandbox["Config"]["Labels"]
        user, generation = labels[f"{_LABEL}.user"], labels[f"{_LABEL}.generation"]
        for role, resource in resources.items():
            actual = (resource.get("Config") or {}).get("Labels") if role in {"sandbox", "proxy"} else resource.get("Labels")
            if any(actual.get(key) != value for key, value in self._labels(sandbox_id, user, generation, role).items()):
                raise RuntimeError("沙箱资源归属不一致")
        proxy = resources["proxy"]
        ports = proxy.get("NetworkSettings", {}).get("Ports", {}).get("8080/tcp") or []
        env = (proxy.get("Config") or {}).get("Env") or []
        tokens = [item.split("=", 1)[1] for item in env if item.startswith("DEERFLOW_RELAY_TOKEN=")]
        if (len(ports) != 1 or ports[0].get("HostIp") != "127.0.0.1" or not ports[0].get("HostPort")
                or len(tokens) != 1 or not tokens[0]):
            raise RuntimeError("沙箱连接信息无法确认")
        info = SandboxInfo(
            sandbox_id=sandbox_id, sandbox_url=f"http://127.0.0.1:{ports[0]['HostPort']}",
            container_name=(sandbox.get("Name") or "").lstrip("/"), container_id=sandbox.get("Id"),
            user_id=user, generation=generation, proxy_id=proxy.get("Id"),
            network_id=resources["network"].get("Id"), egress_network_id=resources["egress"].get("Id"),
            relay_token=tokens[0], request_headers={"X-DeerFlow-Relay-Token": tokens[0]})
        if not self.is_alive(info):
            raise RuntimeError("沙箱资源状态异常")
        return info

    def is_absent(self, sandbox_id: str) -> bool:
        return not self._resources(sandbox_id)

    def inspect_runtime(self, sandbox_id: str) -> SandboxInfo | None:
        info = self.discover(sandbox_id)
        if info is not None:
            return info
        resources = self._resources(sandbox_id)
        if not resources:
            return None
        sandbox = resources.get("sandbox")
        if not sandbox:
            raise RuntimeError("沙箱资源不完整")
        labels = sandbox["Config"]["Labels"]
        return SandboxInfo(sandbox_id, "", container_id=sandbox.get("Id"),
                           user_id=labels[f"{_LABEL}.user"], generation=labels[f"{_LABEL}.generation"],
                           requires_replacement=True)

    def list_running(self) -> list[SandboxInfo]:
        selector = ("--filter", f"label={_LABEL}.instance={self.instance}",
                    "--filter", f"label={_LABEL}.role=sandbox")
        result = []
        for identity in self._docker("ps", "-q", *selector).splitlines():
            resource = self._inspect("container", identity)
            if resource is None:
                raise RuntimeError("Docker 资源状态正在变化")
            sandbox_id = (resource.get("Config", {}).get("Labels") or {}).get(f"{_LABEL}.id", "")
            info = self.discover(sandbox_id)
            if info:
                result.append(info)
        return result

    def reconcile(self) -> None:
        selector = f"label={_LABEL}.instance={self.instance}"
        groups = (("container", self._docker("ps", "-aq", "--filter", selector).splitlines()),
                  ("network", self._docker("network", "ls", "-q", "--filter", selector).splitlines()))
        for kind, identities in groups:
            for identity in identities:
                resource = self._inspect(kind, identity)
                if resource is None:
                    continue
                labels = (resource.get("Config") or {}).get("Labels") if kind == "container" else resource.get("Labels")
                role = (labels or {}).get(f"{_LABEL}.role")
                user_id = (labels or {}).get(f"{_LABEL}.user", "")
                sandbox_id = (labels or {}).get(f"{_LABEL}.id", "")
                generation = (labels or {}).get(f"{_LABEL}.generation", "")
                allowed = {"sandbox", "proxy"} if kind == "container" else {"network", "egress"}
                if (role not in allowed or not re.fullmatch(r"(?:[0-9a-f]{32}|default)", user_id)
                        or not re.fullmatch(r"[0-9a-f]{32}", sandbox_id)
                        or not re.fullmatch(r"[0-9a-f]{32}", generation)
                        or any(labels.get(key) != value for key, value in
                               self._labels(sandbox_id, user_id, generation, role).items())):
                    raise RuntimeError("Docker 旧资源归属无法确认，停止恢复")
                resource_id = resource.get("Id") or resource.get("ID")
                if not isinstance(resource_id, str) or not re.fullmatch(r"[0-9a-f]{64}", resource_id):
                    raise RuntimeError("Docker 旧资源 ID 无效")
                self._docker(kind, "rm", *(["-f"] if kind == "container" else []), resource_id)
                if self._inspect(kind, resource_id) is not None:
                    raise RuntimeError("Docker 旧资源未被清理")
        if self._docker("ps", "-aq", "--filter", selector) or self._docker("network", "ls", "-q", "--filter", selector):
            raise RuntimeError("仍有未清理的 Docker 旧资源")
