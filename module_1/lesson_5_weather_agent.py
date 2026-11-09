from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool
import requests

SYSTEM_PROMPT = """
<system_prompt>
    <role>You are a weather assistant.</role>
    <domain>Your domain is strictly: weather data.</domain>
    <rules>
        <rule>
            Use get_current_temperature to retrieve the current temperature 
            for a location.
        </rule>
        <rule>
            If an error arises when calling get_current_temperature, explain 
            the error to the user.
        </rule>
    </rules>
</system_prompt>
"""

class LocationNotFoundError(LookupError):
    pass

class CoordinatesNotFoundError(LookupError):
    pass

def get_geo_location(location: str) -> dict:
    """Get the geographical coordinates (latitude and longitude) for a given 
    location.
    
    Args:
        location: name of the location for which to retrieve the geographical 
        coordinates.

    Returns:
        A dict containing the name, latitude and longitude of the location.
    Raises:
        requests.exceptions.RequestException: If the request to the geocoding 
        API fails.
        LocationNotFoundError: If the location is not found.
    """
    response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1")
    response.raise_for_status()
    data = response.json()
    results = data.get("results", None)
    if results:
        geo_loc = results[0]
        if "latitude" not in geo_loc or "longitude" not in geo_loc:
            raise CoordinatesNotFoundError(f"Coordinates not found for {location}.")
        return geo_loc
    raise LocationNotFoundError(f"No geo location found for {location}.")

@tool
def get_current_temperature(location: str) -> str:
    """Get the current temperature for the given location.
    
    Args:
        location: name of the location for which to retrieve the current 
        temperature.
    Returns:
        A string containing the current temperature in Celsius for the given
        location.
    """
    try:
        geo_loc = get_geo_location(location)
        response = requests.get(url=f"https://api.open-meteo.com/v1/forecast?latitude={geo_loc['latitude']}&longitude={geo_loc['longitude']}&timezone=auto&current=temperature_2m")
        response.raise_for_status()
        data = response.json()
        return f"Current temperature in {location} is {data['current']['temperature_2m']}°C"
    except Exception as e:
        return f"Error: {e}"


agent = create_deep_agent(
    name="Weather_Agent",
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_current_temperature]
)

def main():
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Moskau?"
            }
        ]
    })

    print(result["messages"][-1].text)

if __name__ == "__main__":
    main()