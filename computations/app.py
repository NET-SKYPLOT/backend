from flask import Flask, request, jsonify
import requests
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import List, Dict
import math
from collections import defaultdict
import uuid
import pytz

from services.core import ComputationPipeline, DOPCalculator

app = Flask(__name__)

# Configuration
DATA_INTEGRATOR_URL = "http://data_integrator:5001"


def parse_iso_datetime(dt_str: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except ValueError:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)


def update_almanac_constellation(almanac, available_constellations):
    """
    Updates the almanac data by adding a 'constellation' key to each satellite record
    based on matching NORAD catalog IDs with the available_constellations data.
    Satellites without a matching constellation are removed from the almanac.

    Parameters:
        almanac (dict): Dictionary containing almanac data under the key "data".
        available_constellations (dict): Dictionary of constellations (e.g., "BEI", "GLO", etc.)
            where each constellation dict has a "sats" key that is a list of satellite dicts
            with a "norad_id" field.

    Returns:
        dict: The updated almanac with only satellites that have matching constellation info,
              each including a new "constellation" key with the full constellation name.
    """
    # Mapping from constellation key to full constellation name.
    constellation_names = {
        "BEI": "BEIDOU",
        "GLO": "GLONASS",
        "GPS": "GPS",
        "GAL": "GALILEO"
    }

    # Build a lookup dictionary mapping each satellite's norad_id to its constellation key.
    norad_to_constellation = {}
    for constellation_key, constellation_data in available_constellations.items():
        for sat in constellation_data.get("sats", []):
            norad_id = sat.get("norad_id")
            if norad_id is not None:
                norad_to_constellation[norad_id] = constellation_key

    # Create a new list to hold only satellites with a matching constellation.
    updated_data = []
    for sat in almanac.get("data", []):
        norad_id = sat.get("NORAD_CAT_ID")
        if norad_id in norad_to_constellation:
            constellation_key = norad_to_constellation[norad_id]
            # Set the full constellation name based on the mapping.
            sat["constellation"] = constellation_names.get(constellation_key, constellation_key)
            updated_data.append(sat)

    # Replace the original almanac data with the filtered list.
    almanac["data"] = updated_data
    return almanac


def lla_to_ecef(lla: Dict) -> np.ndarray:
    """Convert WGS84 (lat, lon, [height]) to ECEF (in km)."""
    lat = math.radians(lla['latitude'])
    lon = math.radians(lla['longitude'])
    alt = lla.get('height', 0)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    sinlat = math.sin(lat)
    coslat = math.cos(lat)
    N = a / math.sqrt(1 - e2 * sinlat ** 2)
    x = (N + alt) * coslat * math.cos(lon)
    y = (N + alt) * coslat * math.sin(lon)
    z = (N * (1 - e2) + alt) * sinlat
    return np.array([x, y, z]) / 1000.0  # convert to km

def ecef_to_latlon(x, y, z):
    a = 6378.137  # Earth's semi-major axis in km
    e2 = 0.00669437999014  # Earth's eccentricity squared

    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, p * (1 - e2)))  # Approximate formula

    return lat, lon


def process_satellite_data(raw_data):
    """
    Process raw satellite data into organized format grouped by satellite.
    
    Args:
        raw_data (dict): Input data with timestamps as keys and satellite lists as values
        
    Returns:
        list: Processed satellite data in format [
            {
                "constellation": "GPS",
                "satellite_id": "GPS BIIR-2 (PRN 13)",
                "position": [
                    {"time": "2025-03-20T12:00:00+00:00", "x": -12909.23, ...},
                    ...
                ]
            },
            ...
        ]
    """
    # Create a dictionary to organize data by satellite
    satellites = defaultdict(lambda: {
        "constellation": None,
        "satellite_id": None,
        "position": []
    })

    # Process each timestamp in chronological order
    for timestamp in sorted(raw_data.keys()):
        for sat_entry in raw_data[timestamp]:
            # Extract satellite information
            constellation = sat_entry[0]
            coords = sat_entry[1]
            sat_id = sat_entry[2]

            # Initialize satellite entry if not exists
            if not satellites[sat_id]["satellite_id"]:
                satellites[sat_id].update({
                    "constellation": constellation,
                    "satellite_id": sat_id
                })

            lat, lon = ecef_to_latlon(coords[0], coords[1], coords[2])

            # Add position data
            satellites[sat_id]["position"].append({
                "time": timestamp,
                "latitude": lat,
                "longitude": lon,
            })

    return list(satellites.values())

# Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/metrics', methods=['POST'])
def compute_metrics():
    try:
        # Parse JSON payload
        data = request.get_json()

        # Extract required parameters
        user_timezone = data.get('timezone', 'UTC')  # Default to UTC if not specified
        raw_start_time = data['start_datetime']
        
        # Parse datetime without timezone first
        try:
            naive_start_time = datetime.fromisoformat(raw_start_time.replace('Z', '+00:00'))
            if naive_start_time.tzinfo is not None:
                naive_start_time = naive_start_time.replace(tzinfo=None)
        except ValueError:
            return jsonify({"error": "Invalid start_datetime format"}), 400

        # Convert user's local time to UTC for computation
        try:
            if user_timezone != 'UTC':
                # Handle both offset strings (like "+02:00") and timezone names (like "Europe/Berlin")
                if user_timezone.startswith(('+', '-')):
                    # Offset format - convert to timezone object
                    offset_hours = int(user_timezone[:3])
                    tz = pytz.FixedOffset(offset_hours * 60)
                else:
                    # Timezone name format
                    tz = pytz.timezone(user_timezone)
                
                # Localize the naive datetime and convert to UTC
                localized = tz.localize(naive_start_time)
                start_time = localized.astimezone(pytz.UTC)
            else:
                start_time = pytz.UTC.localize(naive_start_time)
        except Exception as e:
            app.logger.warning(f"Invalid timezone {user_timezone}, defaulting to UTC: {str(e)}")
            start_time = pytz.UTC.localize(naive_start_time)
            user_timezone = 'UTC'
        
        duration = timedelta(hours=data['duration_hours'])
        cutoff_angle = int(data.get("cutoff_angle", 0.01))
        dem_selection = data['dem']
        application = data["application"]
        constellations = data['constellations']
        receivers = data['receivers']


        
        # Updated almanac request with POST and start_datetime
        almanac = requests.post(
            f"{DATA_INTEGRATOR_URL}/alm",
            json={"start_datetime": data['start_datetime']}
        ).json()


        available_constellations = requests.get(f"{DATA_INTEGRATOR_URL}/constellations").json()
        almanac = update_almanac_constellation(almanac, available_constellations)
        # print(almanac)

        processed_payloads = []  # Each result is a payload with a "receivers" key (a list of one receiver)
        base_receiver_payload = None

        # Process each receiver
        for receiver in receivers:
            rec_lat = receiver["coordinates"]["latitude"]
            rec_lon = receiver["coordinates"]["longitude"]
            dem = requests.post(
                url=f"{DATA_INTEGRATOR_URL}/dem",
                json={
                    "coordinates": [{"lat": rec_lat, "lon": rec_lon}],
                    "selected_source": dem_selection["source"],
                    "dem_type": dem_selection["type"]
                }
            ).json()

            obstacles = receiver["obstacles"]

            # Initialize computation pipeline for this receiver
            pipeline = ComputationPipeline(
                almanac_data=almanac,
                dem_data=dem,
                constellations=constellations,
                obstacles=obstacles,
                cutoff=cutoff_angle
            )

            # Process receiver; each payload includes a "raw_visible" field for common computations.
            result = pipeline.process_receiver(
                receiver=receiver,
                start_time=start_time,
                duration=duration
            )
            processed_payloads.append(result)
            # Assume each result's "receivers" is a list with one element.
            if receiver.get("role", "").lower() == "base":
                base_receiver_payload = result["receivers"][0]

                # processing all the satellites for ll positions
                all_sats_pos = process_satellite_data(result["satellites_positions"])

                # getting time intervals
                intervals = result["planning_details"]["interval_minutes"]


        # Flatten receiver payloads into a single list.
        all_receivers = []
        for proc in processed_payloads:
            all_receivers.extend(proc["receivers"])

        # Compute common_visibility and common_dop if a base receiver exists.
        if base_receiver_payload is not None:
            base_raw = base_receiver_payload.get("raw_visible",
                                                 {})  # time -> list of tuples (constellation, sat_ecef, sat_id)
            base_ecef = lla_to_ecef(base_receiver_payload["coordinates"])
            for rec_payload in all_receivers:
                if rec_payload.get("role", "").lower() == "rover":
                    rover_raw = rec_payload.get("raw_visible", {})
                    common_visibility = {}  # { constellation: { time: count, ... } }
                    common_dop_times = []
                    common_dop_gdop = []
                    common_dop_pdop = []
                    common_dop_hdop = []
                    common_dop_vdop = []

                    # Iterate over each time step present in both base and rover raw data.
                    for t in base_raw.keys():
                        if t in rover_raw:
                            # Build dictionaries keyed by satellite ID.
                            base_dict = {sat[2]: sat for sat in base_raw[t]}
                            rover_dict = {sat[2]: sat for sat in rover_raw[t]}
                            common_sat_list = []
                            for prn, sat in rover_dict.items():
                                if prn in base_dict:
                                    # Append only (constellation, sat_ecef) for DOP calculation.
                                    common_sat_list.append((sat[0], sat[1]))
                                    cons = sat[0]
                                    if cons not in common_visibility:
                                        common_visibility[cons] = {}
                                    common_visibility[cons][t] = common_visibility[cons].get(t, 0) + 1
                            # Compute DOP for this time step if common satellites exist.
                            if common_sat_list:
                                dop_common = DOPCalculator.calculate_dop(common_sat_list, base_ecef)
                                common_dop_times.append(t)
                                common_dop_gdop.append(dop_common["gdop"])
                                common_dop_pdop.append(dop_common["pdop"])
                                common_dop_hdop.append(dop_common["hdop"])
                                common_dop_vdop.append(dop_common["vdop"])

                    # Format common_visibility as required.
                    common_visibility_agg = {}
                    for cons, time_counts in common_visibility.items():
                        satellite_count = [{"time": t, "count": count} for t, count in time_counts.items()]
                        common_visibility_agg[cons] = {"satellite_count": satellite_count}

                    rec_payload["common_visibility"] = common_visibility_agg
                    rec_payload["common_dop"] = {
                        "time": common_dop_times,
                        "gdop": common_dop_gdop,
                        "pdop": common_dop_pdop,
                        "hdop": common_dop_hdop,
                        "vdop": common_dop_vdop
                    }
        # Remove the internal "raw_visible" field from all receivers (base and rover)
        for rec_payload in all_receivers:
            if "raw_visible" in rec_payload:
                del rec_payload["raw_visible"]



        def convert_to_naive(iso_str):
            """Convert ISO string to naive datetime string (without timezone)"""
            try:
                dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
                if dt.tzinfo is not None:
                    if user_timezone != 'UTC':
                        # Convert to user's timezone first, then make naive
                        if user_timezone.startswith(('+', '-')):
                            offset_hours = int(user_timezone[:3])
                            tz = pytz.FixedOffset(offset_hours * 60)
                        else:
                            tz = pytz.timezone(user_timezone)
                        dt = dt.astimezone(tz)
                    return dt.replace(tzinfo=None).isoformat()
                return dt.isoformat()
            except:
                return iso_str
        
        def convert_timestamps(obj):
            """Recursively convert all ISO timestamps in object to naive strings"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        obj[key] = convert_to_naive(value)
                    elif isinstance(value, (dict, list)):
                        convert_timestamps(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str):
                        obj[i] = convert_to_naive(item)
                    elif isinstance(item, (dict, list)):
                        convert_timestamps(item)
        
        # Build final response payload
        final_payload = {
            "status": "success",
            "request_id": str(uuid.uuid4()),
            "planning_details": {
                "start_datetime": convert_to_naive(start_time.isoformat()),
                "duration_hours": duration.total_seconds() / 3600,
                "interval_minutes": intervals,
                "application": application,
                "timezone": user_timezone  # Still include timezone in response for reference
            },
            "receivers": all_receivers,
            "world_view": all_sats_pos
        }
        
        # Convert all timestamps in receivers and world_view to naive format
        convert_timestamps(final_payload['receivers'])
        convert_timestamps(final_payload['world_view'])
        
        return jsonify(final_payload), 200

    except Exception as e:
        app.logger.error(f"Computation error: {str(e)}")
        return jsonify({"error": "Internal computation error"}), 500

        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
