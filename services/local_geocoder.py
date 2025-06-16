import json
import os
import logging
from typing import Optional, Dict, Tuple, Any
# FIX: Import the datetime class from the datetime module
from datetime import datetime, timezone

# Ensure llm_interface is importable if needed from agent package
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

try:
    from agent.llm_interface import geocode_address_with_llm
except ImportError:
    # Define a dummy if it fails, so app can at least load and show error
    def geocode_address_with_llm(address_text: str) -> Optional[Tuple[float, float]]:
        logging.getLogger(__name__).error(
            "CRITICAL: llm_interface.geocode_address_with_llm could not be imported in local_geocoder.")
        return None


logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'data'))
LOCATIONS_FILE = os.path.join(DATA_DIR, 'geocoded_locations.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def _load_locations() -> Dict[str, Dict[str, Any]]:
    """Loads known locations from the JSON file."""
    if not os.path.exists(LOCATIONS_FILE):
        # If file doesn't exist, create it with an empty JSON object
        with open(LOCATIONS_FILE, 'w', encoding='utf-8') as f:
            f.write('{}')
        logger.info(f"Locations file created at {LOCATIONS_FILE}.")
        return {}
    try:
        # FIX: Handle empty file gracefully before trying to load
        if os.path.getsize(LOCATIONS_FILE) == 0:
            logger.warning(
                f"Locations file {LOCATIONS_FILE} is empty. Treating as empty dict.")
            return {}
        with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(
                    f"Locations file {LOCATIONS_FILE} does not contain a valid dictionary. Returning empty.")
                return {}
            return data
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from {LOCATIONS_FILE}. Returning empty dict.")
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error loading locations from {LOCATIONS_FILE}: {e}", exc_info=True)
        return {}


def _save_locations(locations_data: Dict[str, Dict[str, Any]]):
    """Saves locations data to the JSON file."""
    try:
        with open(LOCATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(locations_data, f, indent=2)
        logger.debug(f"Saved locations data to {LOCATIONS_FILE}")
    except Exception as e:
        logger.error(
            f"Error saving locations data to {LOCATIONS_FILE}: {e}", exc_info=True)


def get_all_known_locations() -> Dict[str, Dict[str, Any]]:
    """Returns all currently known locations."""
    return _load_locations()


def get_coordinates_from_local_store(location_name: str, use_llm_fallback: bool = False) -> Optional[Tuple[float, float]]:
    """
    Retrieves coordinates for a location name.
    First checks local store, then optionally uses LLM as fallback.
    """
    if not location_name or not isinstance(location_name, str):
        return None

    locations_data = _load_locations()
    normalized_location_name = location_name.strip().lower()  # Normalize for lookup

    # Exact match first
    for key, data in locations_data.items():
        if key.strip().lower() == normalized_location_name:
            if 'lat' in data and 'lon' in data:
                try:
                    lat, lon = float(data['lat']), float(data['lon'])
                    logger.info(
                        f"Found '{location_name}' in local store: ({lat}, {lon})")
                    return lat, lon
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid coordinates for '{location_name}' in local store: {data}")
            else:
                logger.warning(
                    f"Entry for '{location_name}' in local store missing lat/lon.")
            # If found but invalid/incomplete, don't proceed to LLM unless explicitly told for this entry
            return None

    if use_llm_fallback:
        logger.info(
            f"'{location_name}' not in local store or invalid. Attempting LLM geocoding.")
        # This calls the function from llm_interface
        llm_coords = geocode_address_with_llm(location_name)
        if llm_coords:
            update_known_location(
                location_name, llm_coords[0], llm_coords[1], source="llm_geocoded", notes="Geocoded by LLM")
            return llm_coords
        else:
            logger.warning(f"LLM geocoding failed for '{location_name}'.")
            return None

    logger.debug(
        f"'{location_name}' not found in local store. LLM fallback not used or failed.")
    return None


def update_known_location(location_name: str, lat: float, lon: float, source: str = "manual", notes: str = "") -> bool:
    """Adds or updates a location in the local store."""
    if not location_name or not isinstance(location_name, str):
        logger.error("Invalid location name for update.")
        return False
    try:
        lat_f, lon_f = float(lat), float(lon)
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            logger.error(f"Invalid coordinates for update: ({lat_f}, {lon_f})")
            return False
    except (ValueError, TypeError):
        logger.error(
            f"Coordinates must be numbers for update: lat={lat}, lon={lon}")
        return False

    locations_data = _load_locations()
    # Use the original casing for the key, but check for existing normalized key to avoid near-duplicates
    normalized_new_name = location_name.strip().lower()
    existing_key_to_update = None
    for key in locations_data.keys():
        if key.strip().lower() == normalized_new_name:
            existing_key_to_update = key
            break

    key_to_use = existing_key_to_update if existing_key_to_update else location_name.strip()

    locations_data[key_to_use] = {
        "lat": lat_f,
        "lon": lon_f,
        "source": source,
        "notes": notes.strip(),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    _save_locations(locations_data)
    logger.info(
        f"Updated/Added '{key_to_use}' in local geocode store: ({lat_f}, {lon_f}), Source: {source}")
    return True


def remove_known_location(location_name: str) -> bool:
    """Removes a location from the local store."""
    if not location_name or not isinstance(location_name, str):
        return False

    locations_data = _load_locations()
    normalized_location_name = location_name.strip().lower()
    key_to_remove = None
    for key in locations_data.keys():
        if key.strip().lower() == normalized_location_name:
            key_to_remove = key
            break

    if key_to_remove and key_to_remove in locations_data:
        del locations_data[key_to_remove]
        _save_locations(locations_data)
        logger.info(f"Removed '{key_to_remove}' from local geocode store.")
        return True
    logger.warning(
        f"Location '{location_name}' not found in local store for removal.")
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Local Geocoder Test")

    # Test data
    test_loc_name = "Test Location Alpha"
    test_lat, test_lon = 32.12345, -117.54321

    print(f"\nAttempting to update/add: {test_loc_name}")
    update_known_location(test_loc_name, test_lat, test_lon,
                          source="test_manual", notes="A test entry")

    print(f"\nAttempting to retrieve: {test_loc_name}")
    coords = get_coordinates_from_local_store(test_loc_name)
    if coords:
        print(f"Retrieved: {coords}")
        assert coords == (test_lat, test_lon)
    else:
        print("Not found or error.")

    print(
        f"\nAttempting to retrieve with LLM fallback (will mock LLM for this test): {test_loc_name} (should be found locally)")
    coords_llm_test_local = get_coordinates_from_local_store(
        test_loc_name, use_llm_fallback=True)
    print(f"Retrieved for LLM fallback (local hit): {coords_llm_test_local}")

    unseen_loc_name = "Unseen Location Omega"
    print(
        f"\nAttempting to retrieve unseen location with LLM fallback: {unseen_loc_name}")
    # Mocking the LLM call for this test
    original_geocode_llm = geocode_address_with_llm

    def mock_llm_geocode(address: str):
        if address == unseen_loc_name:
            print(
                f"[MOCK LLM geocode_address_with_llm] called for {address}, returning (33.0, -117.0)")
            return (33.0, -117.0)
        return None
    geocode_address_with_llm = mock_llm_geocode

    coords_llm = get_coordinates_from_local_store(
        unseen_loc_name, use_llm_fallback=True)
    if coords_llm:
        print(f"Retrieved via LLM fallback: {coords_llm}")
        assert coords_llm == (33.0, -117.0)
        # Check if it was saved
        all_locs = _load_locations()
        assert unseen_loc_name in all_locs
        assert all_locs[unseen_loc_name]['source'] == "llm_geocoded"
    else:
        print("LLM fallback did not return coordinates.")
    geocode_address_with_llm = original_geocode_llm  # Restore original

    print(
        f"\nAll known locations: {json.dumps(get_all_known_locations(), indent=2)}")

    print(f"\nAttempting to remove: {test_loc_name}")
    remove_known_location(test_loc_name)
    coords_after_remove = get_coordinates_from_local_store(test_loc_name)
    assert coords_after_remove is None
    print(f"Retrieved after removal: {coords_after_remove}")

    print(f"\nAttempting to remove: {unseen_loc_name}")
    remove_known_location(unseen_loc_name)
    all_locs_after_remove = _load_locations()
    assert unseen_loc_name not in all_locs_after_remove
    print(
        f"All known locations after removals: {json.dumps(all_locs_after_remove, indent=2)}")

    print("\nTest finished.")
