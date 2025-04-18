# services/geocoding.py
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from functools import lru_cache
import time

# Import settings ONLY to get the user agent
try:
    from config.settings import settings
    GEOCODING_USER_AGENT = settings.geocoding_user_agent
except ImportError:
    print("WARNING: Could not import settings for geocoding user agent. Using default.")
    GEOCODING_USER_AGENT = "EidoAgentPOC_Default/1.0"
except AttributeError:
     print("WARNING: 'geocoding_user_agent' not found in settings. Using default.")
     GEOCODING_USER_AGENT = "EidoAgentPOC_Default/1.0"


# Get a logger for this module
logger = logging.getLogger(__name__)

# --- REMOVED logging.basicConfig FROM HERE ---

# Initialize geolocator (do this once)
try:
    geolocator = Nominatim(user_agent=GEOCODING_USER_AGENT)
    logger.info(f"Nominatim geolocator initialized with user agent: {GEOCODING_USER_AGENT}")
except Exception as e:
    logger.error(f"Failed to initialize Nominatim geolocator: {e}", exc_info=True)
    # Handle inability to initialize geolocator if critical, maybe raise error or use a dummy
    geolocator = None

# Cache results to avoid repeated API calls for the same address
@lru_cache(maxsize=512)
def get_coordinates_cached(address: str):
    """Internal cached function for geocoding."""
    if not geolocator:
        logger.error("Geolocator not available. Cannot geocode.")
        return None
    if not address or not isinstance(address, str) or len(address.strip()) < 3:
         logger.debug(f"Skipping geocoding for invalid or short address: '{address}'")
         return None

    logger.debug(f"Geocoding address (cache miss): '{address}'")
    try:
        # Increased timeout, addressdetails=True might not be needed just for coords
        location = geolocator.geocode(address, timeout=10)
        time.sleep(1) # Add delay to respect Nominatim usage policy (1 req/sec)

        if location:
            coords = (location.latitude, location.longitude)
            logger.info(f"Geocoded '{address}' to {coords}")
            return coords
        else:
            logger.warning(f"Geocoding failed for address: '{address}'. No location found.")
            return None
    except GeocoderTimedOut:
        logger.error(f"Geocoding timed out for address: '{address}'")
        return None
    except GeocoderServiceError as e:
        logger.error(f"Geocoding service error for address '{address}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during geocoding for address '{address}': {e}", exc_info=True)
        return None

def get_coordinates(address: str):
    """
    Public function to get coordinates for an address, using the cache.
    Returns a tuple (latitude, longitude) or None if geocoding fails.
    """
    return get_coordinates_cached(address)

# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    # Configure logging ONLY when running this script directly for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    test_address = "1600 Amphitheatre Parkway, Mountain View, CA"
    coords = get_coordinates(test_address)
    print(f"Coordinates for '{test_address}': {coords}")

    # Test cache
    start_time = time.time()
    coords_cached = get_coordinates(test_address)
    end_time = time.time()
    print(f"Cached coordinates for '{test_address}': {coords_cached} (took {end_time - start_time:.6f}s)")

    coords_fail = get_coordinates("an invalid address string that likely won't resolve")
    print(f"Coordinates for invalid address: {coords_fail}")