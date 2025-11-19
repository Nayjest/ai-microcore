import asyncio
import logging
import datetime
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

import httpx
import requests
import mcp.types

from .utils import ExtendedString, ConvertableToMessage
from .ai_func import AiFuncSyntax
from . import ui
from .types import BadAIAnswer, BadAIJsonAnswer
from .wrappers.llm_response_wrapper import LLMResponse
from ._env import env
from .file_storage import storage
from .json_parsing import parse_json

# Temporary mitigate dotenv warnings
# because of issue in FastMCP: https://github.com/jlowin/fastmcp/issues/2018
logging.getLogger("dotenv.main").setLevel(logging.CRITICAL)
if TYPE_CHECKING:
    from fastmcp.client import Client
    from fastmcp.client.progress import ProgressHandler


class WrongMcpUsage(BadAIAnswer):
    ...


class HeaderAuth(httpx.Auth):
    def __init__(self, header_name_or_dict: str | dict, header_value: str | None = None):
        if isinstance(header_name_or_dict, dict):
            if header_value is not None:
                raise ValueError(
                    "HeaderAuth: If a dictionary is provided in first argument, "
                    "header_value should not be set."
                )
            self.headers = header_name_or_dict
        else:
            if header_value is None:
                raise ValueError(
                    "HeaderAuth: header_value must be provided when header_name is a string"
                )
            self.headers = {str(header_name_or_dict): str(header_value)}

    def auth_flow(self, request):
        for k, v in self.headers.items():
            request.headers[k] = v
        yield request


class ToolsCache:
    FILE = "mcp/tools.json"

    @staticmethod
    def read(mcp_url: str) -> Optional["Tools"]:
        logging.info(f"Checking MCP tools cache for {ui.green(mcp_url)}...")
        tools = storage.read_json(ToolsCache.FILE, default={}).get(mcp_url, None)
        if tools is not None:
            return Tools.from_list([Tool(**raw) for raw in tools.values()])
        return None

    @staticmethod
    def write(mcp_url: str, tools: "Tools"):
        logging.info(f"Storing MCP tools cache for {ui.green(mcp_url)}...")
        cached_tools = storage.read_json(ToolsCache.FILE, default={})
        cached_tools[mcp_url] = tools
        storage.write_json(ToolsCache.FILE, cached_tools)

    @staticmethod
    def clear():
        logging.info("Clearing MCP tools cache...")
        storage.delete(ToolsCache.FILE)


class MCPAnswer(ExtendedString, ConvertableToMessage):
    ...


class McpTransport(str, Enum):
    SSE: str = "sse"
    STREAMABLE_HTTP: str = "streamable_http"
    WS: str = "ws"
    STDIO: str = "stdio"

    def __str__(self):
        return self.value


@dataclass
class MCPConnection:
    url: str = None
    transport: McpTransport = field(default=McpTransport.STREAMABLE_HTTP)
    tools: Optional["Tools"] = field(default=None, init=False)
    _client: Optional["Client"] = field(default=None, init=False)

    @staticmethod
    async def init(
        url: str,
        transport: McpTransport,
        fetch_tools: bool = True,
        use_cache: bool = True,
        connect_timeout: float = 10,
        auth: httpx.Auth = None,
    ) -> "MCPConnection":
        # Temporary mitigate FastMCP errors in some environments
        # https://github.com/jlowin/fastmcp/issues/2018
        from fastmcp.client import Client
        con: MCPConnection = MCPConnection(url=url, transport=transport)
        con._client = Client(  # pylint: disable=W0212
            url,
            timeout=connect_timeout,
            auth=auth,
        )
        await con._client.__aenter__()  # pylint: disable=E1101,W0212,C2801
        if fetch_tools:
            await con.fetch_tools(use_cache=use_cache)
        return con

    async def close(self):
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
        else:
            logging.error(f"Trying to close MCP connection that is not opened ({self.url})")

    async def fetch_tools(self, use_cache: bool = True) -> "Tools":
        if self.tools is not None:
            return self.tools

        if use_cache and (cached_tools := ToolsCache.read(self.url)):
            logging.info("Using MCP tools from cache for %s", ui.green(self.url))
            self.tools = cached_tools
            return self.tools

        logging.info("Fetching tools from MCP %s", ui.green(self.url))
        mcp_tools = await self._client.list_tools()
        self.tools = Tools.from_list([Tool.from_mcp(tool) for tool in mcp_tools])
        if use_cache:
            self.update_tools_cache()
        return self.tools

    def update_tools_cache(self):
        if self.tools is None:
            raise RuntimeError("Tools are not fetched yet. Call fetch_tools() first.")
        ToolsCache.write(self.url, self.tools)

    async def call(
        self,
        name: str,
        timeout: datetime.timedelta | float | int | None = None,
        progress_handler: Optional["ProgressHandler"] = None,
        **kwargs
    ):
        assert env().config.AI_SYNTAX_FUNCTION_NAME_FIELD not in kwargs
        params = dict(kwargs)
        params[env().config.AI_SYNTAX_FUNCTION_NAME_FIELD] = name
        return await self.exec(params, timeout=timeout, progress_handler=progress_handler)

    async def exec(
        self,
        params: dict | LLMResponse,
        timeout: datetime.timedelta | float | int | None = None,
        progress_handler: Optional["ProgressHandler"] = None,
    ):
        tool_field = env().config.AI_SYNTAX_FUNCTION_NAME_FIELD
        if isinstance(params, LLMResponse):
            try:
                params = params.parse_json(
                    raise_errors=True,
                    required_fields=[tool_field],
                )
            except BadAIJsonAnswer as e:
                raise WrongMcpUsage(str(e)) from e
        elif isinstance(params, str):
            try:
                params = parse_json(
                    params,
                    raise_errors=True,
                    required_fields=[tool_field]
                )
            except BadAIJsonAnswer as e:
                raise WrongMcpUsage(str(e)) from e
        params = dict(params)
        if not (name := params.pop(tool_field, "")):
            raise WrongMcpUsage(
                f"Wrong MCP tool usage:\n"
                f"Tool name should be passed in '{tool_field}' field."
            )
        logging.info(f"Calling MCP tool {ui.green(name)} with {params}...")
        call_tool_result: mcp.types.CallToolResult = await self._client.call_tool(
            name=name,
            arguments=params,
            timeout=timeout,
            progress_handler=progress_handler
        )
        if (
            call_tool_result
            and len(call_tool_result.content) == 1
            and call_tool_result.content[0].type == "text"
        ):
            return MCPAnswer(call_tool_result.content[0].text, dict(response=call_tool_result))
        return call_tool_result

    # alias
    execute = exec


@dataclass
class Tool:
    @dataclass
    class Arg:
        name: str = field()
        description: str = field(default="")
        type: str = field(default="string")
        required: bool = field(default=...)
        default: any = field(default=...)

        def __post_init__(self):
            if self.required is ...:
                self.required = self.default is ...
            if self.default is ...:
                self.default = None

    name: str = field()
    description: str = field(default="")
    args: dict[str, Arg | dict] = field(default_factory=dict)

    def __post_init__(self):
        for key in self.args.keys():  # pylint: disable=C0201, C0206
            if isinstance(self.args[key], dict):
                self.args[key] = Tool.Arg(**self.args[key])

    @staticmethod
    def from_mcp(tool: mcp.types.Tool) -> "Tool":
        t = Tool(
            name=tool.name,
            description=tool.description,
        )
        for param_name, data in tool.inputSchema.get("properties", {}).items():
            param = Tool.Arg(
                name=param_name,
                description=data.get("title", ""),
                type=data.get("type", "string"),
                required=param_name in tool.inputSchema.get("required", []),
                default="",
            )
            t.args[param_name] = param
        return t

    def describe(self, syntax: AiFuncSyntax = None) -> str:
        syntax = syntax or env().config.DEFAULT_AI_FUNCTION_SYNTAX
        tpl_file = f"ai-func.{syntax}.j2" if syntax in [
            str(e) for e in AiFuncSyntax
        ] else syntax
        metadata = self._get_metadata()
        return env().tpl_function(tpl_file, **metadata)

    def _get_metadata(self):
        return dict(
            name=self.name,
            description=self.description,
            args={
                arg.name: dict(
                    default="NOT_SET" if arg.required else arg.default,
                    type=arg.type,
                    comment=arg.description,  # @todo rename it to description
                )
                for arg in self.args.values()
            }
        )

    def __str__(self):
        return self.describe(syntax=env().config.DEFAULT_AI_FUNCTION_SYNTAX)


class Tools(dict[str, Tool]):
    def __str__(self):
        return "\n".join([str(tool) for tool in self.values()])

    @staticmethod
    def from_list(tools: list[Tool]) -> "Tools":
        return Tools({tool.name: tool for tool in tools})


@dataclass
class MCPServer:
    url: str
    name: str = field(default="")
    tools: Tools = field(default_factory=Tools)
    transport: McpTransport = field(default=None)
    auth: httpx.Auth | dict | None = field(default=None)

    @staticmethod
    def _try_sse_or_streamable_http(url) -> tuple[McpTransport, str]:
        if url.endswith("/"):
            test_sse_url = f"{url}sse"
        else:
            test_sse_url = f"{url}/sse"
        try:
            response = requests.request(method="HEAD", url=test_sse_url, timeout=5)
            if response.status_code == 200:
                return McpTransport.SSE, test_sse_url
        except requests.RequestException:
            # not a SSE endpoint or not reachable
            pass
        return McpTransport.STREAMABLE_HTTP, f"{url}/mcp"

    @staticmethod
    def _guess_transport_type_by_url(url: str) -> Optional[McpTransport]:
        if url.startswith("ws://") or url.startswith("wss://"):
            return McpTransport.WS
        if url.endswith("/mcp") or url.endswith("/mcp/"):
            return McpTransport.STREAMABLE_HTTP
        if url.endswith("/sse") or url.endswith("/sse/"):
            return McpTransport.SSE
        return None

    @staticmethod
    def name_from_url(url: str) -> str:
        """Domain name from URL."""
        return url.split("//")[-1].split("/")[0]

    def __post_init__(self):
        if not self.name:
            self.name = MCPServer.name_from_url(self.url)
        if not self.transport:
            self.transport = self._guess_transport_type_by_url(self.url)
        if self.auth and isinstance(self.auth, dict):
            self.auth = HeaderAuth(self.auth)

    async def connect(
        self,
        fetch_tools: bool = True,
        use_cache: bool = True,
        connect_timeout: float = 10,
    ) -> MCPConnection:
        if self.transport:
            transport, url = self.transport, self.url
        else:
            transport, url = self._try_sse_or_streamable_http(self.url)
        return await MCPConnection.init(
            url=url,
            transport=transport,
            fetch_tools=fetch_tools,
            use_cache=use_cache,
            connect_timeout=connect_timeout,
            auth=self.auth
        )

    def get_tools_cache(self) -> Tools | None:
        return ToolsCache.read(self.url)


class MCPRegistry(dict[str, MCPServer]):
    def __init__(self, server_configs: list[dict | str]):
        super().__init__()
        for server_config in server_configs:
            if isinstance(server_config, str):
                server_config = {
                    "name": MCPServer.name_from_url(server_config),
                    "url": server_config
                }
            self[server_config["name"]] = MCPServer(**server_config)

    def get(self, server_name: str) -> MCPServer:
        if server_name not in self:
            raise ValueError(f"MCP server '{server_name}' not found in registry")
        return self[server_name]

    async def precache_tools(
        self,
        raise_errors: bool = False,
        connect_timeout: int = 10,
    ):
        async def precache_server_tools(server_name):
            conn = None
            try:
                conn = await self.get(server_name).connect(
                    fetch_tools=True,
                    use_cache=False,
                    connect_timeout=connect_timeout,
                )
                conn.update_tools_cache()
            except Exception as e:  # pylint: disable=W0718
                logging.error("Failed to precache tools for MCP server %s: %s", server_name, e)
                if raise_errors:
                    raise
            finally:
                if conn is not None:
                    await conn.close()

        await asyncio.gather(*[precache_server_tools(srv) for srv in self.keys()])

    async def connect_to(
        self,
        server_name: str,
        fetch_tools: bool = True,
        use_cache: bool = True,
    ) -> MCPConnection:
        mcp_server = self.get(server_name)
        return await mcp_server.connect(fetch_tools=fetch_tools, use_cache=use_cache)


def server(name: str) -> MCPServer:
    """
    Returns MCP server by name from the registry.

    Args:
        name (str): The name of the MCP server.

    Returns:
        MCPServer: The MCP server instance.

    Raises:
        ValueError: If the server with the given name is not found in the registry.
    """
    return env().mcp_registry.get(name)
