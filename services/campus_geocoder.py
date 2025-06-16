import logging
from typing import Optional, Tuple, Dict
import re

logger = logging.getLogger(__name__)

# Approximate coordinates for UCSD locations (for demo purposes)
# In a real system, this might come from a GIS database or a more detailed GeoJSON file.
UCSD_NAMED_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "revelle college": (32.8790, -117.2370),
    "muir college": (32.8800, -117.2400),
    "john muir college": (32.8800, -117.2400),  # Alias
    "marshall college": (32.8830, -117.2410),
    "thurgood marshall college": (32.8830, -117.2410),  # Alias
    "warren college": (32.8822, -117.2345),
    "earl warren college": (32.8822, -117.2345),  # Alias
    "roosevelt college": (32.8850, -117.2380),
    "eleanor roosevelt college": (32.8850, -117.2380),  # Alias
    "sixth college": (32.8830, -117.2330),
    "seventh college": (32.8870, -117.2350),
    "eighth college": (32.8880, -117.2390),
    "geisel library": (32.8811, -117.2376),
    "price center": (32.8798, -117.2359),
    "rimac": (32.8886, -117.2397),  # RIMAC Arena / Field
    "liontree arena": (32.8886, -117.2397),  # Alias for RIMAC
    "scripps institution of oceanography": (32.8661, -117.2542),
    "sio": (32.8661, -117.2542),  # Alias
    "ucsd medical center hillcrest": (32.7557, -117.1602),
    "jacobs medical center": (32.8765, -117.2303),
    "pepper canyon apartments": (32.877, -117.233),
    "matthews apartments": (32.876, -117.240),
    "central campus trolley station": (32.8788, -117.2355),
    "library walk": (32.8805, -117.2370),  # Approximate midpoint
    "sun god lawn": (32.8800, -117.2360),
}

# Add more specific buildings or areas as needed
UCSD_NAMED_LOCATIONS.update({
    "applied physics and mathematics building": (32.8818, -117.2348),
    "ap&m": (32.8818, -117.2348),
    "cognitive science building": (32.8825, -117.2365),
    "computer science and engineering building": (32.8829, -117.2339),
    "cse building": (32.8829, -117.2339),
    "student health services": (32.8780, -117.2395),
    "galbraith hall": (32.8795, -117.2385),
    "york hall": (32.8785, -117.2380),
    "mandeville auditorium": (32.8788, -117.2405),
})


class CampusGeocoder:
    def __init__(self, known_locations: Dict[str, Tuple[float, float]]):
        self.known_locations = {name.lower().strip(
        ): coords for name, coords in known_locations.items()}
        logger.info(
            f"Campus Geocoder initialized with {len(self.known_locations)} known locations.")

    def get_coordinates_for_named_place(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Attempts to find coordinates for a known named place.
        Performs a case-insensitive lookup.
        """
        if not place_name or not isinstance(place_name, str):
            return None

        normalized_place_name = place_name.lower().strip()

        # Direct match
        if normalized_place_name in self.known_locations:
            coords = self.known_locations[normalized_place_name]
            logger.info(
                f"Found direct match for named place '{place_name}': {coords}")
            return coords

        # Improved partial match using whole-word boundaries to avoid false positives
        # This prevents 'de' from matching inside 'student'.
        for known_name, coords in self.known_locations.items():
            # Check if the place_name is a whole word within the known_name
            try:
                if re.search(r'\b' + re.escape(normalized_place_name) + r'\b', known_name):
                    # Avoid overly broad matches for very short strings unless it's an exact match
                    if len(normalized_place_name) > 2:
                        logger.info(
                            f"Found partial word match for '{place_name}' with known place '{known_name}': {coords}")
                        return coords
            except re.error as e:
                logger.warning(
                    f"Regex error while matching '{normalized_place_name}' in '{known_name}': {e}")

        logger.debug(f"No match found for named place: '{place_name}'")
        return None


# Initialize a global instance for UCSD
ucsd_geocoder = CampusGeocoder(UCSD_NAMED_LOCATIONS)


def get_ucsd_coordinates(place_name: str) -> Optional[Tuple[float, float]]:
    """Convenience function to use the global UCSD geocoder instance."""
    return ucsd_geocoder.get_coordinates_for_named_place(place_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_places = [
        "Geisel Library",
        "revelle college",
        "Price Center Food Court",  # This might not match directly, tests partial matching
        "Sixth College Apartments",
        "SIO Pier",  # Should match "Scripps Institution of Oceanography" or "sio"
        "NonExistentPlace UCSD"
    ]

    for place in test_places:
        coords = get_ucsd_coordinates(place)
        if coords:
            print(f"Coordinates for '{place}': {coords}")
        else:
            print(f"Could not find coordinates for '{place}'.")

    # Test direct alias
    print(f"Coordinates for 'muir': {get_ucsd_coordinates('muir')}")
