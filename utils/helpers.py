import xml.etree.ElementTree as ET
import logging
from typing import Optional, Dict, List # Added List

logger = logging.getLogger(__name__)

# Define common namespaces found in the sample EIDO XML snippets
NAMESPACES = {
    'pidf': 'urn:ietf:params:xml:ns:pidf',
    'gp': 'urn:ietf:params:xml:ns:pidf:geopriv10',
    'ca': 'urn:ietf:params:xml:ns:pidf:geopriv10:civicAddr',
    'dm': 'urn:ietf:params:xml:ns:pidf:data-model',
    'com': 'urn:ietf:params:xml:ns:EmergencyCallData:Comment',
    'gml': 'http://www.opengis.net/gml' # Added GML
}

def parse_civic_address_from_pidf(xml_string: str) -> Optional[Dict[str, str]]:
    """
    Parses a PIDF-LO XML string (or similar structures containing civicAddress)
    to extract civic address components.
    Returns a dictionary of address components or None if parsing fails.
    """
    if not xml_string or not isinstance(xml_string, str):
        return None
    try:
        # Attempt to parse the XML string
        # Need to handle potential encoding issues or non-XML content gracefully
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
             # Check if it's wrapped in a simple <location> tag from alert_parser
             if xml_string.strip().startswith('<location>') and xml_string.strip().endswith('</location>'):
                  inner_xml = xml_string.strip()[len('<location>'):-len('</location>')].strip()
                  try:
                      root = ET.fromstring(inner_xml)
                      logger.debug("Parsed XML after removing <location> wrapper.")
                  except ET.ParseError as inner_e:
                      logger.warning(f"Failed to parse inner XML after removing wrapper: {inner_e}\nInner XML: {inner_xml[:200]}...")
                      return None
             else:
                  logger.warning(f"Failed to parse XML for civic address: {e}\nXML: {xml_string[:200]}...")
                  return None

        # Find the civicAddress element using namespaces, searching from the parsed root
        civic_address_element = root.find('.//ca:civicAddress', NAMESPACES)

        if civic_address_element is None:
            logger.debug("No <ca:civicAddress> element found in XML.")
            # Fallback: check for simple text address if no structured address
            civic_text_node = root.find('.//civicAddressText') # Common fallback tag
            if civic_text_node is not None and civic_text_node.text:
                 logger.debug("Found <civicAddressText> instead of structured address.")
                 return {'civicAddressText': civic_text_node.text.strip()} # Return as a special key
            return None # No structured or simple text address found

        address_components = {}
        tags_to_extract = ['country', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                           'PRD', 'POD', 'RD', 'STS', 'HNO', 'HNS',
                           'LMK', 'LOC', 'NAM', 'PC', 'BLD', 'UNIT', 'FLR', 'ROOM', 'SEAT', 'PLC']

        for tag in tags_to_extract:
            element = civic_address_element.find(f'ca:{tag}', NAMESPACES)
            if element is not None and element.text:
                address_components[tag] = element.text.strip()

        # If primary components are missing, check the fallback text node within civicAddress too
        if not any(k in address_components for k in ['RD', 'A3', 'PC']) and 'civicAddressText' not in address_components:
             civic_text_fallback = civic_address_element.find('.//civicAddressText')
             if civic_text_fallback is not None and civic_text_fallback.text:
                  address_components['civicAddressText'] = civic_text_fallback.text.strip()
                  logger.debug("Used civicAddressText fallback within ca:civicAddress.")


        logger.debug(f"Parsed civic address components: {address_components}")
        return address_components

    except Exception as e:
        logger.error(f"Unexpected error parsing civic address XML: {e}", exc_info=True)
        return None

def format_address_from_components(addr_components: Dict[str, str]) -> Optional[str]:
    """
    Formats a string address from parsed civic address components.
    Prioritizes structured fields, falls back to civicAddressText if needed.
    """
    if not addr_components or not isinstance(addr_components, dict):
        return None

    parts: List[str] = []

    # Prioritize structured components if available
    hno = addr_components.get('HNO', '')
    prd = addr_components.get('PRD', '')
    rd = addr_components.get('RD', '')
    sts = addr_components.get('STS', '')
    pod = addr_components.get('POD', '') # Post-directional

    street_parts = [hno, prd, rd, sts, pod]
    street_line = " ".join(filter(None, street_parts)).strip()
    if street_line:
        parts.append(street_line)

    # Secondary address info (Unit, Floor, etc.)
    unit = addr_components.get('UNIT', '')
    flr = addr_components.get('FLR', '')
    room = addr_components.get('ROOM', '')
    bld = addr_components.get('BLD', '')
    secondary_parts = []
    if bld: secondary_parts.append(f"Bldg {bld}")
    if unit: secondary_parts.append(f"Unit {unit}")
    if flr: secondary_parts.append(f"Flr {flr}")
    if room: secondary_parts.append(f"Room {room}")
    secondary_line = ", ".join(filter(None, secondary_parts)).strip()
    if secondary_line:
        parts.append(secondary_line)

    # Location details (City, State, ZIP)
    city = addr_components.get('A3', '') # Usually City/Municipality
    state = addr_components.get('A1', '') # Usually State/Province
    pc = addr_components.get('PC', '') # Postal Code

    # Try other A-tags if A3/A1 are empty
    if not city: city = addr_components.get('A6', '') # Try A6 (Neighborhood/Community)
    if not city: city = addr_components.get('A4', '') # Try A4
    # State is usually A1

    city_state_zip_parts = [city, state, pc]
    city_state_zip_line = f"{city}, {state} {pc}".replace(" ,", ",").replace("  ", " ").strip(" ,") # Clean up formatting
    if city_state_zip_line and city_state_zip_line != ',':
         parts.append(city_state_zip_line)

    # Landmark / Additional Info
    lmk = addr_components.get('LMK', '')
    loc = addr_components.get('LOC', '')
    if lmk: parts.append(f"({lmk})")
    if loc: parts.append(f"({loc})")

    # If no structured parts were found, use civicAddressText if available
    if not parts and 'civicAddressText' in addr_components:
        full_address = addr_components['civicAddressText']
        logger.debug(f"Formatted address using civicAddressText fallback: {full_address}")
        return full_address
    elif not parts:
        return None # No usable components

    # Join structured parts
    full_address = ", ".join(filter(None, parts))
    logger.debug(f"Formatted address from components: {full_address}")
    return full_address


def parse_comment_from_emergency_data(xml_string: str) -> Optional[str]:
    """
    Parses EmergencyCallData.Comment XML to extract the comment text.
    """
    if not xml_string:
        return None
    try:
        # Handle potential wrappers if necessary
        try:
             root = ET.fromstring(xml_string)
        except ET.ParseError:
             if xml_string.strip().startswith('<commentWrapper>') and xml_string.strip().endswith('</commentWrapper>'): # Example wrapper
                  inner_xml = xml_string.strip()[len('<commentWrapper>'):-len('</commentWrapper>')].strip()
                  try: root = ET.fromstring(inner_xml)
                  except ET.ParseError as inner_e: logger.warning(f"Failed to parse inner comment XML: {inner_e}"); return None
             else: logger.warning(f"Failed to parse comment XML: {xml_string[:200]}..."); return None

        # Find the Comment element using namespaces
        comment_element = root.find('.//com:Comment', NAMESPACES)

        if comment_element is not None and comment_element.text:
            comment = comment_element.text.strip()
            logger.debug(f"Parsed comment: {comment}")
            return comment
        else:
            logger.debug("No Comment element found or comment is empty.")
            return None

    except Exception as e:
        logger.error(f"Unexpected error parsing Comment XML: {e}", exc_info=True)
        return None