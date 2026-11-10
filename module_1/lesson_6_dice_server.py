import random
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="dice", log_level="WARNING")

@mcp.tool()
def roll_dice(sides: int = 6) -> int:
    """Roll a die with the given number of sides and return the result."""
    return random.randint(1, sides)

if __name__ == "__main__":
    mcp.run()