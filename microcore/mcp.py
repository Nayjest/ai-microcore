import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, types

from .utils import ExtendedString, ConvertableToMessage
from .ai_func import AiFuncSyntax
from . import ui
from .types import BadAIAnswer, BadAIJsonAnswer
from .wrappers.llm_response_wrapper import LLMResponse
from ._env import env
from .file_storage import storage


class WrongMcpUsage(BadAIAnswer):
    ...


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


@dataclass
class MCPConnection:
    url: str = None
    read_stream: any = None
    write_stream: any = None
    connection: any = None
    context_manager: any = None
    session: ClientSession = field(default=None, init=False)
    tools: Optional["Tools"] = field(default=None, init=False)
    _lifecycle_task: asyncio.Task = field(default=None, init=False)
    _del_event: asyncio.Event = field(default=None, init=False)

    @staticmethod
    async def init(
        url: str,
        fetch_tools: bool = True,
        use_cache: bool = True,
        connect_timeout: float = 7,
    ) -> "MCPConnection":

        del_event = asyncio.Event()
        opened_event = asyncio.Event()
        con: MCPConnection = MCPConnection()
        con._del_event = del_event  # pylint: disable=W0212

        # That's a bit of a hack for closing the async context managers
        async def lifecycle():
            try:
                logging.info(f"Connecting to MCP {url}...")
                context_manager = streamablehttp_client(url)
                (
                    read_stream,
                    write_stream,
                    connection
                ) = await context_manager.__aenter__()  # pylint: disable=E1101
                con.url = url
                con.read_stream = read_stream
                con.write_stream = write_stream
                con.connection = connection
                con.context_manager = context_manager
                await con.init_session()
                if fetch_tools:
                    await con.fetch_tools(use_cache=use_cache)
                opened_event.set()

            finally:
                await del_event.wait()
                await con._close()  # pylint: disable=W0212

        con._lifecycle_task = asyncio.create_task(lifecycle())  # pylint: disable=W0212
        try:
            await asyncio.wait_for(opened_event.wait(), timeout=connect_timeout)
        except Exception as e:
            logging.warning(f"Failed to connect to MCP {url}: {e}")
            try:
                await con._close()  # pylint: disable=W0212
            except:  # noqa: E722 # pylint: disable=W0702
                pass
            raise
        return con

    async def close(self):
        self._del_event.set()
        if self._lifecycle_task:
            await self._lifecycle_task
        else:
            logging.error(f"Trying to close MCP connection that is not opened ({self.url})")

    async def _close(self):
        logging.info(f"Closing MCP session ({self.url})...")
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                del self.session
                self.session = None
        finally:
            logging.info(f"Closing MCP connection ({self.url})...")
            try:
                if self.context_manager:
                    await self.context_manager.__aexit__(None, None, None)
                    del self.context_manager
                    self.context_manager = None
            finally:
                logging.info(f"Closed CTX ({self.url})")

    async def init_session(self):
        logging.info(f"Initializing MCP session ({self.url})")
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()  # pylint: disable=unnecessary-dunder-call
        await self.session.initialize()
        return self.session

    async def fetch_tools(self, use_cache: bool = True) -> "Tools":
        if self.tools is not None:
            return self.tools

        if use_cache and (cached_tools := ToolsCache.read(self.url)):
            logging.info("Using MCP tools from cache for %s", ui.green(self.url))
            self.tools = cached_tools
            return self.tools

        logging.info("Fetching tools from MCP %s", ui.green(self.url))
        mcp_tools = await self.session.list_tools()
        self.tools = Tools.from_list([Tool.from_mcp(tool) for tool in mcp_tools.tools])
        if use_cache:
            self.update_tools_cache()
        return self.tools

    def update_tools_cache(self):
        if self.tools is None:
            raise RuntimeError("Tools are not fetched yet. Call fetch_tools() first.")
        ToolsCache.write(self.url, self.tools)

    def __del__(self):
        self._del_event.set()

    async def exec(self, params: dict | LLMResponse):
        if isinstance(params, LLMResponse):
            try:
                params = params.parse_json(
                    raise_errors=True,
                    required_fields=[env().config.AI_SYNTAX_FUNCTION_NAME_FIELD],
                )
            except BadAIJsonAnswer as e:
                raise WrongMcpUsage(str(e)) from e
        params = dict(params)
        name = params.pop(env().config.AI_SYNTAX_FUNCTION_NAME_FIELD)
        if not name:
            raise WrongMcpUsage(
                f"Tool name should be passed in {env().config.AI_SYNTAX_FUNCTION_NAME_FIELD} field"
            )
        logging.info(f"Calling MCP tool {ui.green(name)} with {params}...")
        result = await self.session.call_tool(name, params)
        content = result.content
        if content and len(content) == 1 and content[0].type == "text":
            return MCPAnswer(content[0].text, result.__dict__)
        return result


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
    def from_mcp(tool: types.Tool) -> "Tool":
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
        syntax = syntax or AiFuncSyntax.DEFAULT
        tpl_file = f"ai-func.{syntax}.j2" if syntax in AiFuncSyntax else syntax
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
        return self.describe(syntax=AiFuncSyntax.DEFAULT)


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

    @staticmethod
    def name_from_url(url: str) -> str:
        """Domain name from URL."""
        return url.split("//")[-1].split("/")[0]

    def __post_init__(self):
        if not self.name:
            self.name = MCPServer.name_from_url(self.url)

    async def connect(
        self,
        fetch_tools: bool = True,
        use_cache: bool = True,
        connect_timeout: float = 7,
    ) -> MCPConnection:
        return await MCPConnection.init(
            self.url,
            fetch_tools=fetch_tools,
            use_cache=use_cache,
            connect_timeout=connect_timeout,
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
        connect_timeout: int = 7,
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
