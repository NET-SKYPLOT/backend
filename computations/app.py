from flask import Flask, request, jsonify
import requests
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import List, Dict
import math
import uuid

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


# # Working...
# @app.route('/metrics', methods=['POST'])
# def compute_metrics():
#     try:
#         # Parse JSON payload
#         data = request.get_json()

#         # Extract required parameters
#         start_time = parse_iso_datetime(data['start_datetime'])
#         duration = timedelta(hours=data['duration_hours'])
#         dem_selection = data['dem']

#         constellations = data['constellations']
#         receivers = data['receivers']
#         almanac = requests.get(f"{DATA_INTEGRATOR_URL}/alm").json()

#         available_constellations = requests.get(f"{DATA_INTEGRATOR_URL}/constellations").json()
#         almanac = update_almanac_constellation(almanac, available_constellations)

#         # Build a base payload following the documented structure.
#         final_payload = {
#             "status": "success",
#             "request_id": str(uuid.uuid4()),
#             "planning_details": {
#                 "start_datetime": start_time.isoformat(),
#                 "duration_hours": duration.total_seconds() / 3600,
#                 "interval_minutes": 30,
#                 "application": "GNSS Planning"
#             },
#             "receivers": []
#         }

#         # Process each receiver
#         for receiver in receivers:
#             rec_lat = receiver["coordinates"]["latitude"]
#             rec_lon = receiver["coordinates"]["longitude"]
#             dem = requests.post(
#                 url=f"{DATA_INTEGRATOR_URL}/dem",
#                 json={
#                     "coordinates": [{"lat": rec_lat, "lon": rec_lon}],
#                     "selected_source": dem_selection["source"],
#                     "dem_type": dem_selection["type"]
#                 }
#             ).json()

#             obstacles = receiver["obstacles"]

#             # Initialize computation pipeline with the DEM and obstacles for this receiver.
#             pipeline = ComputationPipeline(
#                 almanac_data=almanac,
#                 dem_data=dem,
#                 constellations=constellations,
#                 obstacles=obstacles
#             )

#             # Process receiver and obtain a payload structured as per documentation.
#             receiver_payload = pipeline.process_receiver(
#                 receiver=receiver,
#                 start_time=start_time,
#                 duration=duration
#             )

#             # Merge the computed receiver payload into the final response.
#             final_payload["receivers"].extend(receiver_payload["receivers"])

#         return jsonify(final_payload), 200

#     except Exception as e:
#         app.logger.error(f"Computation error: {str(e)}")
#         return jsonify({"error": "Internal computation error"}), 500

# Working...
# @app.route('/metrics', methods=['POST'])
# def compute_metrics():
#     try:
#         # Parse JSON payload
#         data = request.get_json()

#         # Extract required parameters
#         start_time = parse_iso_datetime(data['start_datetime'])
#         duration = timedelta(hours=data['duration_hours'])
#         dem_selection = data['dem']

#         constellations = data['constellations']
#         receivers = data['receivers']
#         almanac = requests.get(f"{DATA_INTEGRATOR_URL}/alm").json()

#         available_constellations = requests.get(f"{DATA_INTEGRATOR_URL}/constellations").json()
#         almanac = update_almanac_constellation(almanac, available_constellations)

#         processed_receivers = []
#         base_payload = None

#         # Process each receiver
#         for receiver in receivers:
#             rec_lat = receiver["coordinates"]["latitude"]
#             rec_lon = receiver["coordinates"]["longitude"]
#             dem = requests.post(
#                 url=f"{DATA_INTEGRATOR_URL}/dem",
#                 json={
#                     "coordinates": [{"lat": rec_lat, "lon": rec_lon}],
#                     "selected_source": dem_selection["source"],
#                     "dem_type": dem_selection["type"]
#                 }
#             ).json()

#             obstacles = receiver["obstacles"]

#             # Initialize computation pipeline for this receiver
#             pipeline = ComputationPipeline(
#                 almanac_data=almanac,
#                 dem_data=dem,
#                 constellations=constellations,
#                 obstacles=obstacles
#             )

#             # Process receiver; note that the returned payload now includes an extra key "raw_visible"
#             # containing the per–time-step list of visible satellites (each tuple is (constellation, sat_ecef, prn)).
#             result = pipeline.process_receiver(
#                 receiver=receiver,
#                 start_time=start_time,
#                 duration=duration
#             )
#             processed_receivers.append(result)
#             if receiver.get("role", "").lower() == "base":
#                 base_payload = result["receivers"][0]

#         # If a base receiver exists, compute common_visibility and common_dop for rovers
#         if base_payload is not None:
#             base_raw = base_payload.get("raw_visible", {})
#             base_ecef = lla_to_ecef(base_payload["coordinates"])
#             for proc in processed_receivers:
#                 rec = proc["receivers"][0]
#                 if rec.get("role", "").lower() == "rover":
#                     rover_raw = rec.get("raw_visible", {})
#                     common_vis = {}  # Structure: { constellation: { time: count, ... } }
#                     common_dop_times = []
#                     common_dop_gdop = []
#                     common_dop_pdop = []
#                     common_dop_hdop = []
#                     common_dop_vdop = []
#                     for t in rover_raw.keys():
#                         # Create dictionaries keyed by satellite PRN
#                         base_dict = {sat[2]: sat for sat in base_raw.get(t, [])}
#                         rover_dict = {sat[2]: sat for sat in rover_raw.get(t, [])}
#                         common = []
#                         for prn, sat in rover_dict.items():
#                             if prn in base_dict:
#                                 common.append(sat)
#                                 cons = sat[0]
#                                 if cons not in common_vis:
#                                     common_vis[cons] = {}
#                                 common_vis[cons][t] = common_vis[cons].get(t, 0) + 1
#                         dop_common = DOPCalculator.calculate_dop(common, base_ecef)
#                         common_dop_times.append(t)
#                         common_dop_gdop.append(dop_common["gdop"])
#                         common_dop_pdop.append(dop_common["pdop"])
#                         common_dop_hdop.append(dop_common["hdop"])
#                         common_dop_vdop.append(dop_common["vdop"])
#                     # Convert common_vis to required structure
#                     common_visibility = {}
#                     for cons, time_counts in common_vis.items():
#                         satellite_count = [{"time": t, "count": count} for t, count in time_counts.items()]
#                         common_visibility[cons] = {"satellite_count": satellite_count}

#                     rec["common_visibility"] = common_visibility
#                     rec["common_dop"] = {
#                         "time": common_dop_times,
#                         "gdop": common_dop_gdop,
#                         "pdop": common_dop_pdop,
#                         "hdop": common_dop_hdop,
#                         "vdop": common_dop_vdop
#                     }
#                 # Remove the internal "raw_visible" field
#                 if "raw_visible" in rec:
#                     del rec["raw_visible"]
#         else:
#             # If no base receiver exists, leave common fields empty
#             for proc in processed_receivers:
#                 rec = proc["receivers"][0]
#                 rec["common_visibility"] = {}
#                 rec["common_dop"] = {}
#                 if "raw_visible" in rec:
#                     del rec["raw_visible"]

#         response = {
#             "receivers": processed_receivers,
#             "metadata": {
#                 "processing_time": datetime.utcnow().isoformat(),
#                 "computation_id": str(uuid.uuid4())
#             }
#         }
#         return jsonify(response), 200

#     except Exception as e:
#         app.logger.error(f"Computation error: {str(e)}")
#         return jsonify({"error": "Internal computation error"}), 500


# working not correct payload
# @app.route('/metrics', methods=['POST'])
# def compute_metrics():
#     try:
#         # Parse JSON payload
#         data = request.get_json()

#         # Extract required parameters
#         start_time = parse_iso_datetime(data['start_datetime'])
#         duration = timedelta(hours=data['duration_hours'])
#         dem_selection = data['dem']

#         constellations = data['constellations']
#         receivers = data['receivers']
#         almanac = requests.get(f"{DATA_INTEGRATOR_URL}/alm").json()

#         available_constellations = requests.get(f"{DATA_INTEGRATOR_URL}/constellations").json()
#         almanac = update_almanac_constellation(almanac, available_constellations)

#         processed_receivers = []
#         base_receiver_payload = None

#         # Process each receiver
#         for receiver in receivers:
#             rec_lat = receiver["coordinates"]["latitude"]
#             rec_lon = receiver["coordinates"]["longitude"]
#             dem = requests.post(
#                 url=f"{DATA_INTEGRATOR_URL}/dem",
#                 json={
#                     "coordinates": [{"lat": rec_lat, "lon": rec_lon}],
#                     "selected_source": dem_selection["source"],
#                     "dem_type": dem_selection["type"]
#                 }
#             ).json()

#             obstacles = receiver["obstacles"]

#             # Initialize computation pipeline for this receiver
#             pipeline = ComputationPipeline(
#                 almanac_data=almanac,
#                 dem_data=dem,
#                 constellations=constellations,
#                 obstacles=obstacles
#             )

#             # Process receiver; each receiver payload now includes a "raw_visible" field.
#             result = pipeline.process_receiver(
#                 receiver=receiver,
#                 start_time=start_time,
#                 duration=duration
#             )
#             processed_receivers.append(result)
#             if receiver.get("role", "").lower() == "base":
#                 base_receiver_payload = result["receivers"][0]

#         # Compute common_visibility and common_dop if a base receiver exists.
#         if base_receiver_payload is not None:
#             base_raw = base_receiver_payload.get("raw_visible", {})  # dict: time -> list of tuples (constellation, sat_ecef, sat_id)
#             base_ecef = lla_to_ecef(base_receiver_payload["coordinates"])
#             for proc in processed_receivers:
#                 rec_payload = proc["receivers"][0]
#                 if rec_payload.get("role", "").lower() == "rover":
#                     rover_raw = rec_payload.get("raw_visible", {})
#                     common_visibility = {}  # { constellation: { time: count, ... } }
#                     common_dop_times = []
#                     common_dop_gdop = []
#                     common_dop_pdop = []
#                     common_dop_hdop = []
#                     common_dop_vdop = []

#                     # Iterate over each time step that appears in both base and rover raw data.
#                     for t in base_raw.keys():
#                         if t in rover_raw:
#                             # Build dictionaries keyed by satellite ID.
#                             base_dict = {sat[2]: sat for sat in base_raw[t]}
#                             rover_dict = {sat[2]: sat for sat in rover_raw[t]}
#                             common_sat_list = []
#                             for prn, sat in rover_dict.items():
#                                 if prn in base_dict:
#                                     # Append only (constellation, sat_ecef) for DOP calculation.
#                                     common_sat_list.append((sat[0], sat[1]))
#                                     cons = sat[0]
#                                     if cons not in common_visibility:
#                                         common_visibility[cons] = {}
#                                     common_visibility[cons][t] = common_visibility[cons].get(t, 0) + 1
#                             # Compute DOP for this time step if common satellites exist.
#                             dop_common = DOPCalculator.calculate_dop(common_sat_list, base_ecef)
#                             common_dop_times.append(t)
#                             common_dop_gdop.append(dop_common["gdop"])
#                             common_dop_pdop.append(dop_common["pdop"])
#                             common_dop_hdop.append(dop_common["hdop"])
#                             common_dop_vdop.append(dop_common["vdop"])

#                     common_visibility_agg = {}
#                     for cons, time_counts in common_visibility.items():
#                         satellite_count = [{"time": t, "count": count} for t, count in time_counts.items()]
#                         common_visibility_agg[cons] = {"satellite_count": satellite_count}

#                     rec_payload["common_visibility"] = common_visibility_agg
#                     rec_payload["common_dop"] = {
#                         "time": common_dop_times,
#                         "gdop": common_dop_gdop,
#                         "pdop": common_dop_pdop,
#                         "hdop": common_dop_hdop,
#                         "vdop": common_dop_vdop
#                     }
#                     # Optionally remove the raw_visible field.
#                     if "raw_visible" in rec_payload:
#                         del rec_payload["raw_visible"]
#         else:
#             # No base receiver found; leave common fields empty.
#             for proc in processed_receivers:
#                 rec_payload = proc["receivers"][0]
#                 rec_payload["common_visibility"] = {}
#                 rec_payload["common_dop"] = {}
#                 if "raw_visible" in rec_payload:
#                     del rec_payload["raw_visible"]

#         # Structure final response
#         final_payload = {
#             "status": "success",
#             "request_id": str(uuid.uuid4()),
#             "planning_details": {
#                 "start_datetime": start_time.isoformat(),
#                 "duration_hours": duration.total_seconds() / 3600,
#                 "interval_minutes": 30,
#                 "application": "GNSS Planning"
#             },
#             "receivers": processed_receivers
#         }

#         return jsonify(final_payload), 200


#     except Exception as e:
#         app.logger.error(f"Computation error: {str(e)}")
#         return jsonify({"error": "Internal computation error"}), 500

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
        start_time = parse_iso_datetime(data['start_datetime'])
        duration = timedelta(hours=data['duration_hours'])
        dem_selection = data['dem']
        application = data["application"]
        constellations = data['constellations']
        receivers = data['receivers']
        almanac = requests.get(f"{DATA_INTEGRATOR_URL}/alm").json()

        available_constellations = requests.get(f"{DATA_INTEGRATOR_URL}/constellations").json()
        almanac = update_almanac_constellation(almanac, available_constellations)

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
                obstacles=obstacles
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

        # Build final response payload.
        final_payload = {
            "status": "success",
            "request_id": str(uuid.uuid4()),
            "planning_details": {
                "start_datetime": start_time.isoformat(),
                "duration_hours": duration.total_seconds() / 3600,
                "interval_minutes": 30,
                "application": application
            },
            "receivers": all_receivers
        }

        return jsonify(final_payload), 200

    except Exception as e:
        app.logger.error(f"Computation error: {str(e)}")
        return jsonify({"error": "Internal computation error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
