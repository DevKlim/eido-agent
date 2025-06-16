import logging
from typing import List, Tuple, Optional
# Use the NEW core data schema for reports
# Location schema not needed here directly
from data_models.schemas import ReportCoreData, Incident
from config.settings import settings  # For thresholds
from datetime import timedelta, datetime, timezone  # Import datetime/timezone
from geopy.distance import geodesic
import re  # Added for content similarity

logger = logging.getLogger(__name__)


# Helper function for content similarity
def _get_meaningful_words(text: Optional[str]) -> set[str]:
    """
    Extracts meaningful words from a text string, converting to lowercase,
    removing punctuation, common stop words, and very short words.
    """
    if not text:
        return set()

    # Exclude placeholder descriptions
    lower_text = text.lower()
    if "no description" in lower_text or "not yet generated" in lower_text or "error generating" in lower_text:
        return set()

    # Tokenize words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', lower_text)

    # Define a set of common English stop words (can be expanded or loaded from a library)
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not',
        'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was',
        'will', 'with', 'he', 'she', 'it', 'we', 'you', 'i', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
        'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
        'should', 'now', 'etc', 'also', 'about', 'after', 'before', 'between', 'during', 'through', 'upon', 'within',
        'without', 'around', 'above', 'below', 'beside', 'behind', 'across', 'along', 'among', 'amongst', 'around',
        'away', 'back', 'down', 'even', 'ever', 'every', 'first', 'good', 'great', 'last', 'little', 'long', 'many',
        'much', 'new', 'next', 'old', 'one', 'other', 'over', 'part', 'per', 'place', 'point', 'right', 'said', 'say',
        'see', 'seem', 'should', 'since', 'small', 'some', 'still', 'such', 'take', 'tell', 'than', 'that', 'their',
        'them', 'then', 'there', 'these', 'they', 'thing', 'think', 'third', 'those', 'though', 'through', 'time',
        'too', 'under', 'until', 'up', 'upon', 'us', 'use', 'very', 'want', 'way', 'well', 'went', 'were', 'what',
        'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within',
        'without', 'work', 'would', 'year', 'yet', 'you', 'your', 'yourself', 'zero'
    }
    # Filter out stop words and single-character words (unless they are digits, e.g., '1', '2')
    meaningful_words = {word for word in words if word not in stop_words and (
        len(word) > 1 or word.isdigit())}
    return meaningful_words


def calculate_similarity(core_data: ReportCoreData, incident: Incident) -> Tuple[float, str]:
    """
    Calculates a similarity score between new report core data and an existing incident.
    Returns (score, reason_for_match).
    Score is 0-1 (1 being perfect match).
    """
    # --- Configuration ---
    time_window = timedelta(minutes=settings.time_window_minutes)
    distance_threshold_km = settings.distance_threshold_km
    # similarity_threshold = settings.similarity_threshold # Used by the caller function
    content_similarity_min_common_words = getattr(
        # Default to 2 if not in settings
        settings, 'content_similarity_min_common_words', 2)

    # --- Ensure Timestamps are Comparable (UTC Aware) ---
    try:
        # Ensure core_data timestamp is UTC aware
        report_ts = core_data.timestamp
        if report_ts.tzinfo is None or report_ts.tzinfo.utcoffset(report_ts) != timezone.utc.utcoffset(None):
            logger.warning(
                f"Report {core_data.report_id[:8]} timestamp was naive or non-UTC. Converting to UTC for comparison.")
            report_ts = report_ts.astimezone(timezone.utc)

        # Ensure incident timestamp is UTC aware
        incident_ts = incident.last_updated_at
        # Handle case where incident might not have a timestamp yet (shouldn't happen if created properly)
        if not incident_ts:
            logger.warning(
                f"Incident {incident.incident_id[:8]} has no last_updated_at timestamp for comparison.")
            time_match = False  # Cannot match time if incident timestamp is missing
        elif incident_ts.tzinfo is None or incident_ts.tzinfo.utcoffset(incident_ts) != timezone.utc.utcoffset(None):
            logger.warning(
                f"Incident {incident.incident_id[:8]} last_updated_at was naive or non-UTC. Converting to UTC for comparison.")
            incident_ts = incident_ts.astimezone(timezone.utc)
            # Calculate time_diff and time_match immediately after conversion
            time_diff = abs(report_ts - incident_ts)
            time_match = time_diff <= time_window
            logger.debug(
                f"Time check (converted): Report={report_ts}, IncidentLastUpdate={incident_ts}, Diff={time_diff}, Match={time_match} (Window={time_window})")
        else:
            # Both timestamps are valid and aware, calculate difference
            time_diff = abs(report_ts - incident_ts)
            time_match = time_diff <= time_window
            logger.debug(
                f"Time check: Report={report_ts}, IncidentLastUpdate={incident_ts}, Diff={time_diff}, Match={time_match} (Window={time_window})")

    except Exception as ts_error:
        logger.error(
            f"Error comparing timestamps between Report {core_data.report_id[:8]} and Incident {incident.incident_id[:8]}: {ts_error}", exc_info=True)
        time_match = False  # Fail time match on error

    # --- External ID Match (Strongest Hint) ---
    # Check if the new report's external_incident_id matches the incident's *primary* external ID
    # (which we can infer from the first report added to the incident).
    external_id_match = False
    if core_data.external_incident_id and incident.reports_core_data:
        # Check against the external ID of the first report associated with the incident
        # Ensure the first report actually has an external ID
        first_report = incident.reports_core_data[0]
        if first_report.external_incident_id:
            if core_data.external_incident_id == first_report.external_incident_id:
                external_id_match = True
                logger.debug(
                    f"External ID match: Report ExtID '{core_data.external_incident_id}' == Incident's First Report ExtID.")
            # else:
            #      logger.debug(f"External ID mismatch: Report ExtID '{core_data.external_incident_id}' != Incident's First Report ExtID '{first_report.external_incident_id}'.")
        else:
            logger.debug(
                f"Incident's first report ({first_report.report_id[:8]}) lacks an external_incident_id for comparison.")
    elif not core_data.external_incident_id:
        logger.debug(
            "Report Core Data lacks an external_incident_id for comparison.")
    elif not incident.reports_core_data:
        logger.debug(
            "Incident has no reports yet, cannot compare external_incident_id.")

    # --- Location Similarity ---
    # Use coordinates extracted into core_data
    location_match = False
    min_distance = float('inf')
    if core_data.coordinates:
        # Ensure coordinates are valid tuple of floats
        if isinstance(core_data.coordinates, tuple) and len(core_data.coordinates) == 2 and all(isinstance(c, (float, int)) for c in core_data.coordinates):
            report_coords = (float(core_data.coordinates[0]), float(
                core_data.coordinates[1]))
            if incident.locations:  # Use the incident's consolidated list of unique locations
                for inc_lat, inc_lon in incident.locations:
                    try:
                        # Ensure incident location coords are also valid floats before calculating
                        if isinstance(inc_lat, (float, int)) and isinstance(inc_lon, (float, int)):
                            dist = geodesic(
                                report_coords, (float(inc_lat), float(inc_lon))).km
                            min_distance = min(min_distance, dist)
                            if dist <= distance_threshold_km:
                                location_match = True
                                break  # Found a location within threshold
                        else:
                            logger.warning(
                                f"Skipping invalid incident location coordinates: ({inc_lat}, {inc_lon})")
                    except ValueError as e:
                        logger.warning(
                            f"Could not calculate distance: {e} (Coords: {report_coords} vs {(inc_lat, inc_lon)})")
                logger.debug(
                    f"Location check: Report={report_coords}, Incident Locs={len(incident.locations)}, MinDist={min_distance:.2f}km, Match={location_match} (Threshold={distance_threshold_km}km)")
            else:
                logger.debug(
                    "Location check: Incident has no geocoded locations yet.")
        else:
            logger.warning(
                f"Location check: Report Core Data has invalid coordinates format: {core_data.coordinates}")
    else:
        logger.debug(
            "Location check: Report Core Data has no geocoded location.")

    # --- Content Similarity (Refactored with helper and stop words) ---
    content_match = False
    report_meaningful_words = _get_meaningful_words(core_data.description)

    if report_meaningful_words:
        # Compare with incident summary first
        incident_summary_words = _get_meaningful_words(incident.summary)
        if incident_summary_words:
            common_words_summary = report_meaningful_words.intersection(
                incident_summary_words)
            if len(common_words_summary) >= content_similarity_min_common_words:
                content_match = True
                logger.debug(
                    f"Content check (vs Summary): Match=True (Common: {common_words_summary})")

        # If no match with summary, compare with descriptions of previous reports' core data
        if not content_match and incident.reports_core_data:
            for prev_core_data in incident.reports_core_data:
                prev_meaningful_words = _get_meaningful_words(
                    prev_core_data.description)
                if prev_meaningful_words:
                    common_words_history = report_meaningful_words.intersection(
                        prev_meaningful_words)
                    if len(common_words_history) >= content_similarity_min_common_words:
                        content_match = True
                        logger.debug(
                            f"Content check (vs History): Match=True (Common: {common_words_history} with report {prev_core_data.report_id[:8]})")
                        break  # Found a match, no need to check further reports

    if not content_match:
        logger.debug(f"Content check (basic keywords): Match=False")

    # --- Scoring Logic (Prioritize External ID, then Time+Location) ---
    score = 0.0
    reasons = []

    # Highest confidence: Matching External ID + Time + Location
    if external_id_match and time_match and location_match:
        score = 0.98
        reasons.append("ExternalID+Time+Location")
    elif external_id_match and time_match:
        score = 0.95
        reasons.append("ExternalID+Time")
    elif external_id_match and location_match:  # Less likely without time match, but possible
        score = 0.90
        reasons.append("ExternalID+Location")
    elif external_id_match:  # ID alone is a strong indicator
        score = 0.88
        reasons.append("ExternalID Only")
    # Next Strongest: Time + Location + Content
    elif time_match and location_match and content_match:
        score = 0.85
        reasons.append("Time+Location+Content")
    # Next: Time + Location
    elif time_match and location_match:
        score = 0.75  # Default threshold is 0.70, so this would match
        reasons.append("Time+Location")
    # Weaker matches (below typical threshold but calculated for logging)
    elif time_match and content_match:
        score = 0.65
        reasons.append("Time+Content")
    elif location_match and content_match:
        score = 0.60
        reasons.append("Location+Content")
    elif time_match:
        score = 0.40
        reasons.append("Time Only")
    elif location_match:
        score = 0.30
        reasons.append("Location Only")
    elif content_match:
        score = 0.20
        reasons.append("Content Only")

    final_reason = ", ".join(reasons) if reasons else "No Matching Factors"
    logger.debug(
        f"Similarity Score for Inc {incident.incident_id[:8]} vs Report {core_data.report_id[:8]}: {score:.2f} ({final_reason})")
    return score, final_reason


def find_match_for_report(core_data: ReportCoreData, incidents: List[Incident]) -> Tuple[Optional[str], float, str]:
    """
    Finds the best matching incident for new report core data from the provided list.
    Returns (best_match_incident_id, highest_score, best_reason) or (None, 0, "No Match Found").
    Filters the provided list for active incidents internally.
    """
    best_match_id = None
    highest_score = 0.0
    best_reason = "No Match Found"

    # Define statuses considered active (case-insensitive check)
    # These should align with the statuses used in incident_store.get_active_incidents if filtering happens there too
    active_statuses = [
        "active", "updated", "received", "rcvd",
        "dispatched", "dsp", "acknowledged", "ack",
        "enroute", "enr", "onscene", "onscn"
        # Add any other status considered 'matchable'
    ]

    # Filter the provided incidents list to only consider active ones
    # This is where the original error occurred if 'incidents' was not List[Incident]
    try:
        active_incidents_to_check = [
            inc for inc in incidents
            if isinstance(inc, Incident) and inc.status and isinstance(inc.status, str) and inc.status.lower() in active_statuses
        ]
    except AttributeError as e:
        # This catch block is a safeguard but shouldn't be hit if the caller passes List[Incident]
        logger.error(
            f"AttributeError during active incident filtering in matching: {e}. Input type might be incorrect.", exc_info=True)
        return None, 0.0, f"Matching Error: {e}"
    except Exception as e:
        logger.error(
            f"Unexpected error during active incident filtering in matching: {e}", exc_info=True)
        return None, 0.0, f"Matching Error: {type(e).__name__}"

    if not active_incidents_to_check:
        logger.debug(
            "No active incidents found in the provided list to match against.")
        # Determine reason: was the input list empty or just no active ones?
        reason = "No Active Incidents Provided" if incidents else "No Incidents Provided"
        return None, 0.0, reason

    logger.debug(
        f"Attempting to match Report {core_data.report_id[:8]} against {len(active_incidents_to_check)} active incidents.")

    for incident in active_incidents_to_check:
        # Ensure we are definitely working with an Incident object
        if not isinstance(incident, Incident):
            logger.warning(
                f"Skipping item in matching loop - not an Incident object: {type(incident)}")
            continue

        try:
            score, reason = calculate_similarity(core_data, incident)
            if score > highest_score:
                highest_score = score
                best_match_id = incident.incident_id
                best_reason = reason
        except Exception as calc_err:
            logger.error(
                f"Error calculating similarity for Incident {incident.incident_id[:8]}: {calc_err}", exc_info=True)
            # Continue checking other incidents

    # Apply the final threshold check
    if best_match_id and highest_score >= settings.similarity_threshold:
        logger.info(
            f"Match decision: Incident {best_match_id[:8]} selected for Report {core_data.report_id[:8]} with score {highest_score:.2f} ({best_reason})")
        return best_match_id, highest_score, best_reason
    else:
        final_reason = f"Score Below Threshold ({highest_score:.2f} < {settings.similarity_threshold})" if highest_score > 0 else best_reason
        logger.info(
            f"Match decision: No incident found for Report {core_data.report_id[:8]} above threshold ({settings.similarity_threshold}). Highest score: {highest_score:.2f} ({best_reason})")
        return None, highest_score, final_reason
