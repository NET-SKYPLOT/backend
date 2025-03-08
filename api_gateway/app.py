from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
import math

app = Flask(__name__)
CORS(app)

# Service URLs
DATA_INTEGRATOR_URL = "http://data_integrator:8761"
COMPUTATIONS_URL = "http://computations:8762"

# Maximum allowed distance between receivers in kilometers
MAX_RECEIVER_DISTANCE_KM = 20


# Helper Functions
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic points in kilometers using the Haversine formula."""
    R = 6371.0  # Earth radius in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def validate_receivers(receivers):
    """Validate receiver coordinates and ensure they are within the maximum allowed distance."""
    if not receivers:
        raise ValueError("No receivers provided")

    # Check if all receivers have valid coordinates
    for receiver in receivers:
        if 'coordinates' not in receiver or not all(
                key in receiver['coordinates'] for key in ['latitude', 'longitude']):
            raise ValueError("Each receiver must have latitude and longitude coordinates")
        lat = receiver['coordinates']['latitude']
        lon = receiver['coordinates']['longitude']
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError(f"Invalid coordinates: latitude {lat}, longitude {lon} out of bounds")

    # For single receiver, no distance check needed
    if len(receivers) == 1:
        return

    # Check pairwise distances for multiple receivers
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            lat1 = receivers[i]['coordinates']['latitude']
            lon1 = receivers[i]['coordinates']['longitude']
            lat2 = receivers[j]['coordinates']['latitude']
            lon2 = receivers[j]['coordinates']['longitude']
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance > MAX_RECEIVER_DISTANCE_KM:
                raise ValueError(
                    f"Receivers are too far apart: {distance:.2f} km (max allowed: {MAX_RECEIVER_DISTANCE_KM} km)")


def validate_plan_data(data):
    """Validate the input data for the /api/v1/plan endpoint."""
    required_fields = ['start_datetime', 'duration_hours', 'dem', 'constellations', 'receivers']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate start_datetime (ISO format, e.g., "2023-10-25T10:00:00Z")
    try:
        datetime.fromisoformat(data['start_datetime'].replace('Z', '+00:00'))
    except ValueError:
        raise ValueError("Invalid start_datetime format; use ISO format (e.g., '2020-10-25T10:00:00Z')")

    # Validate duration_hours
    if not isinstance(data['duration_hours'], (int, float)) or data['duration_hours'] <= 0:
        raise ValueError("duration_hours must be a positive number")

    # Validate dem
    if not isinstance(data['dem'], dict) or not all(key in data['dem'] for key in ['type', 'source']):
        raise ValueError("dem must be an object with 'type' and 'source' fields")

    # Validate constellations
    if not isinstance(data['constellations'], list) or not data['constellations']:
        raise ValueError("constellations must be a non-empty list")

    # Validate receivers
    validate_receivers(data['receivers'])

    # Handle application type (single or differential_gnss)
    application = data.get('application', 'single')
    if application not in ['single', 'differential_gnss']:
        raise ValueError("application must be 'single' or 'differential_gnss'")

    if application == 'differential_gnss':
        if len(data['receivers']) < 2:
            raise ValueError("Differential GNSS requires at least two receivers")
        roles = [receiver.get('role') for receiver in data['receivers']]
        if roles.count('base') != 1:
            raise ValueError("Exactly one receiver must have the role 'base' for differential GNSS")
        if roles.count('rover') < 1:
            raise ValueError("At least one receiver must have the role 'rover' for differential GNSS")
    elif application == 'single':
        if len(data['receivers']) != 1:
            raise ValueError("Single receiver application requires exactly one receiver")


# Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/api/v1/dems', methods=['POST'])
def get_available_dems():
    """Get available DEMs for given receiver coordinates."""
    try:
        data = request.get_json()
        if not data or 'receivers' not in data:
            return jsonify({"error": "Missing 'receivers' field in request body"}), 400

        receivers = data['receivers']
        if not isinstance(receivers, list):
            return jsonify({"error": "'receivers' must be a list"}), 400

        # Validate receivers
        validate_receivers(receivers)

        # Forward to Data Integrator
        response = requests.post(
            f"{DATA_INTEGRATOR_URL}/dem_availability",
            json={"receivers": receivers},
            timeout=10  # Add timeout to prevent hanging
        )

        if response.status_code != 200:
            return jsonify({"error": "DEM service unavailable"}), 503

        return jsonify(response.json()), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except requests.RequestException as re:
        return jsonify({"error": f"Failed to contact DEM service: {str(re)}"}), 503
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/v1/plan', methods=['POST'])
def create_plan():
    """Main planning endpoint with validation and dummy response."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be a JSON object"}), 400

        # Validate plan data
        validate_plan_data(data)

        metrics_response = requests.post(f"{COMPUTATIONS_URL}/metrics", json=data)

        # Dummy response for now
        # return jsonify({
        #     "status": "success",
        #     "message": "Plan created successfully",
        #     "data": data
        # }), 200

        return jsonify(metrics_response.json()), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Run the Application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
