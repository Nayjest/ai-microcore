"""
Temporary mitigate dotenv warnings
because of issue in FastMCP: https://github.com/jlowin/fastmcp/issues/2018
"""
import logging


logging.getLogger("dotenv.main").setLevel(logging.CRITICAL)
