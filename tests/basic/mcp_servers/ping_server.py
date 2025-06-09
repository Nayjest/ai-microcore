import argparse
from mcp.server.fastmcp import FastMCP


parser = argparse.ArgumentParser(description='MCP Server')
parser.add_argument('--port', type=int, default=8881)
parser.add_argument('--transport', type=str, default='sse', help='sse|strio|streamable-http)')
args = parser.parse_args()

mcp = FastMCP(debug=True, host='0.0.0.0', port=args.port)

@mcp.tool()
def ping(message: str): return " ".join(["pong", message])

if __name__ == "__main__":
    mcp.run(args.transport)