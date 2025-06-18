import logging
from typing import Optional, Tuple, Dict, List, Any

from agent.llm_interface import extract_geolocatable_clues, geocode_address_with_llm
from services import geocoding as nominatim_geocoder
from services import campus_geocoder
from services import local_geocoder

logger = logging.getLogger(__name__)

# Define confidence levels
CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_NONE = "None"

class AdvancedGeocodingService:
    def __init__(self):
        logger.info("Advanced Geocoding Service initialized.")

    async def geocode_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """
        Orchestrates the advanced geocoding process for a given narrative.
        This version has a more robust, multi-step fallback logic.
        """
        if not narrative_text or not narrative_text.strip():
            return {"coordinates": None, "confidence": CONFIDENCE_NONE, "method": "No input text", "extracted_clues": {}, "reasoning_log": ["Input narrative was empty."]}

        reasoning_log = [f"Starting advanced geocoding for: '{narrative_text[:100]}...'"]
        
        # 1. Direct Local Store Lookup (Fastest check)
        # Check if the exact narrative has been geocoded and stored before.
        coords_direct = local_geocoder.get_coordinates_from_local_store(narrative_text, use_llm_fallback=False)
        if coords_direct:
            reasoning_log.append(f"Found exact match in local store: '{narrative_text}' -> {coords_direct}")
            return {"coordinates": coords_direct, "confidence": CONFIDENCE_HIGH, "method": "Local Store (Exact Match)", "extracted_clues": {}, "reasoning_log": reasoning_log}
        
        # 2. Extract Geolocatable Clues using LLM
        clues = extract_geolocatable_clues(narrative_text)
        if not clues:
            reasoning_log.append("LLM Clue Extraction failed or returned no clues.")
            # As a final fallback, try to geocode the raw text directly if clue extraction fails
            coords_fallback = await self._final_fallback_geocoding(narrative_text, reasoning_log)
            if coords_fallback:
                return {"coordinates": coords_fallback, "confidence": CONFIDENCE_LOW, "method": "LLM (Direct Fallback)", "extracted_clues": {}, "reasoning_log": reasoning_log}
            return {"coordinates": None, "confidence": CONFIDENCE_NONE, "method": "Clue extraction failed", "extracted_clues": {}, "reasoning_log": reasoning_log}

        reasoning_log.append(f"LLM Extracted Clues: {self._summarize_clues(clues)}")

        # 3. Prioritize Explicit Addresses
        if clues.get("explicit_addresses"):
            for address in clues["explicit_addresses"]:
                reasoning_log.append(f"Attempting to geocode explicit address: '{address}'")
                # Try local store first for the specific address
                coords_local = local_geocoder.get_coordinates_from_local_store(address, use_llm_fallback=False)
                if coords_local:
                    reasoning_log.append(f"Local store success for address: {coords_local}.")
                    return {"coordinates": coords_local, "confidence": CONFIDENCE_HIGH, "method": f"Local Store (explicit address)", "extracted_clues": clues, "reasoning_log": reasoning_log}
                # Then Nominatim
                coords = nominatim_geocoder.get_coordinates(address)
                if coords:
                    reasoning_log.append(f"Nominatim success: {coords}.")
                    return {"coordinates": coords, "confidence": CONFIDENCE_HIGH, "method": f"Nominatim (direct address)", "extracted_clues": clues, "reasoning_log": reasoning_log}
                # Fallback to LLM for the address
                reasoning_log.append(f"Nominatim failed. Trying LLM geocoding for address.")
                coords_llm_addr = await geocode_address_with_llm(address)
                if coords_llm_addr:
                    reasoning_log.append(f"LLM for address success: {coords_llm_addr}.")
                    return {"coordinates": coords_llm_addr, "confidence": CONFIDENCE_MEDIUM, "method": f"LLM (direct address)", "extracted_clues": clues, "reasoning_log": reasoning_log}
        
        # 4. Process Named Entities (The core of finding unknown locations)
        named_entities = clues.get("named_entities", [])
        if named_entities:
            best_candidate = await self._find_best_candidate_from_entities(named_entities, reasoning_log)
            if best_candidate:
                 confidence = CONFIDENCE_HIGH if best_candidate['confidence_raw'] > 0.85 else CONFIDENCE_MEDIUM
                 return {"coordinates": best_candidate['coords'], "confidence": confidence, "method": f"{best_candidate['source']} for entity '{best_candidate['name']}'", "extracted_clues": clues, "reasoning_log": reasoning_log}

        # 5. Fallback to geocoding the whole narrative if no specific clues worked
        coords_fallback = await self._final_fallback_geocoding(narrative_text, reasoning_log)
        if coords_fallback:
             return {"coordinates": coords_fallback, "confidence": CONFIDENCE_LOW, "method": "LLM (Full Narrative Fallback)", "extracted_clues": clues, "reasoning_log": reasoning_log}

        # If we reach here, no method worked
        reasoning_log.append("All geocoding methods failed to find coordinates.")
        return {"coordinates": None, "confidence": CONFIDENCE_NONE, "method": "All methods failed", "extracted_clues": clues, "reasoning_log": reasoning_log}

    async def _find_best_candidate_from_entities(self, named_entities: List[Dict], reasoning_log: List[str]) -> Optional[Dict]:
        candidate_locations = []
        for entity in named_entities:
            entity_name = entity.get("name", "").strip()
            if not entity_name: continue
            
            reasoning_log.append(f"Processing named entity: '{entity_name}'")
            
            # Check known sources in order of reliability
            sources_to_check = [
                ("Local Store", local_geocoder.get_coordinates_from_local_store, 0.95, False),
                ("UCSD Campus Geocoder", campus_geocoder.get_ucsd_coordinates, 0.90, False),
                ("Nominatim POI", lambda name: nominatim_geocoder.get_coordinates(f"{name}, San Diego"), 0.75, False),
                ("LLM Geocoder", geocode_address_with_llm, 0.65, True) # LLM is a fallback per-entity
            ]

            for source_name, geocode_func, confidence_score, is_async in sources_to_check:
                try:
                    coords = await geocode_func(entity_name) if is_async else geocode_func(entity_name)
                    if coords:
                        reasoning_log.append(f"Found '{entity_name}' via {source_name}: {coords}")
                        candidate_locations.append({"name": entity_name, "coords": coords, "source": source_name, "confidence_raw": confidence_score})
                        break # Found a match for this entity, move to the next entity
                except Exception as e:
                    logger.warning(f"Error calling {source_name} for '{entity_name}': {e}", exc_info=False)

        if candidate_locations:
            best_candidate = sorted(candidate_locations, key=lambda x: x["confidence_raw"], reverse=True)[0]
            reasoning_log.append(f"Selected best candidate: '{best_candidate['name']}' from {best_candidate['source']} with score {best_candidate['confidence_raw']:.2f}")
            return best_candidate
        return None

    async def _final_fallback_geocoding(self, narrative_text: str, reasoning_log: List[str]) -> Optional[Tuple[float, float]]:
        if len(narrative_text.split()) > 3: # Avoid geocoding very short, non-address text
            reasoning_log.append("Attempting to geocode full narrative with LLM as a fallback.")
            coords_narrative_llm = await geocode_address_with_llm(narrative_text)
            if coords_narrative_llm:
                reasoning_log.append(f"LLM geocoding of narrative result: {coords_narrative_llm}.")
                return coords_narrative_llm
        return None

    def _summarize_clues(self, clues: Dict[str, Any]) -> str:
        summary_parts = []
        if clues.get("explicit_addresses"): summary_parts.append(f"Addresses: {', '.join(clues['explicit_addresses'])}")
        if clues.get("named_entities"): summary_parts.append(f"Entities: {len(clues['named_entities'])}")
        if clues.get("spatial_relationships"): summary_parts.append(f"Relations: {len(clues['spatial_relationships'])}")
        return "; ".join(summary_parts) if summary_parts else "No distinct clues."

# Singleton instance
advanced_geocoder_service_instance = AdvancedGeocodingService()

async def get_advanced_geocoding_service() -> AdvancedGeocodingService:
    return advanced_geocoder_service_instance