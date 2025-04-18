# utils/helpers.py
import xml.etree.ElementTree as ET
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Define common namespaces found in the sample EIDO XML snippets
NAMESPACES = {
    'pidf': 'urn:ietf:params:xml:ns:pidf',
    'gp': 'urn:ietf:params:xml:ns:pidf:geopriv10',
    'ca': 'urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr',
    'dm': 'urn:ietf:params:xml:ns:pidf:data-model',
    'com': 'urn:ietf:params:xml:ns:EmergencyCallData:Comment'
}

def parse_civic_address_from_pidf(xml_string: str) -> Optional[Dict[str, str]]:
    """
    Parses a PIDF-LO XML string to extract civic address components.
    Returns a dictionary of address components or None if parsing fails.
    """
    if not xml_string:
        return None
    try:
        root = ET.fromstring(xml_string)
        # Find the civicAddress element using namespaces
        civic_address_element = root.find('.//ca:civicAddress', NAMESPACES)

        if civic_address_element is None:
            logger.debug("No civicAddress element found in PIDF XML.")
            return None

        address_components = {}
        # Extract common civic address fields (adjust tags based on standard/needs)
        # See RFC 5139 and RFC 4119 for common civic address elements
        tags_to_extract = ['country', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', # Admin units (State, County, City, etc.)
                           'PRD', 'POD', # Primary/Post Directional
                           'RD', 'STS', # Street Name, Street Suffix
                           'HNO', 'HNS', # House Number, House Number Suffix
                           'LMK', 'LOC', # Landmark, Additional Location Info
                           'NAM', 'PC', # Name (Building), Postal Code
                           'BLD', 'UNIT', 'FLR', 'ROOM', 'SEAT', # Building details
                           'PLC'] # Place Type

        for tag in tags_to_extract:
            element = civic_address_element.find(f'ca:{tag}', NAMESPACES)
            if element is not None and element.text:
                address_components[tag] = element.text.strip()

        logger.debug(f"Parsed civic address components: {address_components}")
        return address_components

    except ET.ParseError as e:
        logger.warning(f"Failed to parse PIDF XML for civic address: {e}\nXML: {xml_string[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing PIDF XML: {e}", exc_info=True)
        return None

def format_address_from_components(addr_components: Dict[str, str]) -> Optional[str]:
    """
    Formats a string address from parsed civic address components.
    (This is a basic example, real formatting can be complex)
    """
    if not addr_components:
        return None

    parts = []
    # House number and street
    hno = addr_components.get('HNO', '')
    prd = addr_components.get('PRD', '')
    rd = addr_components.get('RD', '')
    sts = addr_components.get('STS', '')
    street_part = f"{hno} {prd} {rd} {sts}".strip().replace("  ", " ")
    if street_part:
        parts.append(street_part)

    # Building/Unit info
    bld = addr_components.get('BLD', '')
    unit = addr_components.get('UNIT', '')
    flr = addr_components.get('FLR', '')
    room = addr_components.get('ROOM', '')
    if bld: parts.append(f"Bldg {bld}")
    if unit: parts.append(f"Unit {unit}")
    if flr: parts.append(f"Flr {flr}")
    if room: parts.append(f"Room {room}")

    # City, State, Postal Code
    city = addr_components.get('A3', '') # Assuming A3 is City
    state = addr_components.get('A1', '') # Assuming A1 is State
    pc = addr_components.get('PC', '')
    city_state_zip = f"{city}, {state} {pc}".strip().replace(" ,", ",").replace("  ", " ")
    if city_state_zip != ',': # Avoid adding just separators
        parts.append(city_state_zip)

    # Landmark or Additional Info
    lmk = addr_components.get('LMK', '')
    loc = addr_components.get('LOC', '')
    if lmk: parts.append(f"({lmk})")
    if loc: parts.append(f"({loc})")

    if not parts:
        return None

    full_address = ", ".join(filter(None, parts))
    logger.debug(f"Formatted address: {full_address}")
    return full_address


def parse_comment_from_emergency_data(xml_string: str) -> Optional[str]:
    """
    Parses EmergencyCallData.Comment XML to extract the comment text.
    """
    if not xml_string:
        return None
    try:
        root = ET.fromstring(xml_string)
        # Find the Comment element using namespaces
        comment_element = root.find('.//com:Comment', NAMESPACES)

        if comment_element is not None and comment_element.text:
            comment = comment_element.text.strip()
            logger.debug(f"Parsed comment: {comment}")
            return comment
        else:
            logger.debug("No Comment element found or comment is empty.")
            return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse Comment XML: {e}\nXML: {xml_string[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing Comment XML: {e}", exc_info=True)
        return None