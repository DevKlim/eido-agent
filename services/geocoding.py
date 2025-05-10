import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from functools import lru_cache
import time
import sys
from typing import Optional
import os

# Ensure config is importable
# Assuming geocoding.py is in services/ and config/ is a sibling
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config.settings import settings
    GEOCODING_USER_AGENT = settings.geocoding_user_agent
except ImportError as e:
    print(f"WARNING: Could not import settings for geocoding user agent: {e}. Using default.")
    GEOCODING_USER_AGENT = "EidoSentinelApp_Default/0.8 (contact: default@example.com)"
except AttributeError:
     print("WARNING: 'geocoding_user_agent' not found in settings. Using default.")
     GEOCODING_USER_AGENT = "EidoSentinelApp_Default/0.8 (contact: default@example.com)"

logger = logging.getLogger(__name__)
# Configure logger if not already configured by a higher-level basicConfig
if not logger.hasHandlers():
    log_level_attr = getattr(settings if 'settings' in locals() else object(), 'log_level', 'INFO')
    logging.basicConfig(level=log_level_attr.upper(), format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


geolocator: Optional[Nominatim] = None
try:
    if not GEOCODING_USER_AGENT or 'example.com' in GEOCODING_USER_AGENT or '@' not in GEOCODING_USER_AGENT: # Stricter check
        logger.critical(f"Nominatim User-Agent is not configured or uses default/invalid format: '{GEOCODING_USER_AGENT}'. Geocoding will likely fail. Please set GEOCODING_USER_AGENT correctly in .env")
        # Keep geolocator as None or proceed and let Nominatim fail
        # geolocator = None # Option to disable if invalid
    if geolocator is not None or (GEOCODING_USER_AGENT and '@' in GEOCODING_USER_AGENT): # Attempt init only if valid agent or not explicitly disabled
        geolocator = Nominatim(user_agent=GEOCODING_USER_AGENT)
        logger.info(f"Nominatim geolocator initialized with user agent: {GEOCODING_USER_AGENT}")
except Exception as e:
    logger.error(f"Failed to initialize Nominatim geolocator: {e}", exc_info=True)
    geolocator = None


@lru_cache(maxsize=1024)
def get_coordinates_cached(address: str) -> Optional[tuple[float, float]]: # Type hint uses Optional now
    if not geolocator:
        logger.error("Geolocator not available or not initialized properly. Cannot geocode.")
        return None
    if not address or not isinstance(address, str) or len(address.strip()) < 3:
         # logger.debug(f"Skipping geocoding for invalid or too short address: '{address}'")
         return None

    # logger.debug(f"Geocoding address (cache miss or first call): '{address}'")
    try:
        location = geolocator.geocode(address, timeout=10)
        time.sleep(1.1) # Adhere to 1 req/sec policy

        if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            # Ensure coordinates are floats before returning
            try:
                lat = float(location.latitude)
                lon = float(location.longitude)
                coords = (lat, lon)
                logger.info(f"Geocoded '{address}' to {coords}")
                return coords
            except (ValueError, TypeError):
                 logger.warning(f"Geocoded location for '{address}' has non-float coordinates: lat={location.latitude}, lon={location.longitude}")
                 return None
        else:
            logger.warning(f"Geocoding failed for address: '{address}'. No location found or location object malformed.")
            return None
    except GeocoderTimedOut:
        logger.error(f"Geocoding timed out for address: '{address}'. Try increasing timeout or check network.")
        return None
    except GeocoderUnavailable:
        logger.error(f"Geocoding service (Nominatim) unavailable for address '{address}'. Service might be down or blocking requests.")
        return None
    except GeocoderServiceError as e:
        logger.error(f"Geocoding service error for address '{address}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during geocoding for address '{address}': {e}", exc_info=True)
        return None

def get_coordinates(address: str) -> Optional[tuple[float, float]]: # Type hint uses Optional now
    """
    Public function to get coordinates for an address, using the cache.
    Returns a tuple (latitude, longitude) or None if geocoding fails.
    """
    return get_coordinates_cached(address)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', force=True)

    if not geolocator:
        print("Geolocator not initialized. Check GEOCODING_USER_AGENT and logs.")
    else:
        print(f"Testing geocoding with User-Agent: {GEOCODING_USER_AGENT}")
        test_addresses = [
            "1600 Amphitheatre Parkway, Mountain View, CA 94043", # Added ZIP
            "Eiffel Tower, Paris, France",
            "Invalid Address String Which Should Not Resolve 123456789",
            "Buckingham Palace, London, UK", # Test cache
            "1600 Amphitheatre Parkway, Mountain View, CA 94043", # Test cache
            "", # Test empty string
            "   ", # Test whitespace only
            None, # Test None
        ]
        for addr in test_addresses:
            print(f"\nRequesting coordinates for: '{addr}'")
            start_time = time.time()
            coords = get_coordinates(addr) # type: ignore # Ignore None type check for test
            duration = time.time() - start_time
            if coords:
                print(f"Coordinates: {coords} (Took {duration:.3f}s)")
            else:
                print(f"Failed to get coordinates or invalid input. (Took {duration:.3f}s)")