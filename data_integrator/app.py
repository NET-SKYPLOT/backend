from flask import Flask, request, jsonify
import requests
import rasterio
from rasterio.transform import rowcol
from io import BytesIO
import numpy as np
import math
import json
from datetime import datetime
import logging
from config import config, DEM_SOURCES
from services.almanac.almanac import AlmanacService
from services.dem import DEMService
from services.constellations import ConstellationManager

app = Flask(__name__)
almanac_service = AlmanacService()
dem_service = DEMService()
constellation_manager = ConstellationManager()


@app.route('/dem_availability', methods=['POST'])
def get_available_dems():
    """Get available DEM sources for given coordinates"""
    try:
        data = request.json

        receivers = data["receivers"]
        print(receivers)

        if not receivers:
            return jsonify({"error": "No coordinates provided"}), 400

        available_sources = {}
        recommendation = None
        highest_res = 0

        for source_id, source_config in DEM_SOURCES.items():
            # Check coverage
            if source_config.get('global', False):
                # Global source always available
                available = True
            else:
                # Check regional coverage
                available = all(
                    source_config['coverage']['min_lat'] <= receiver["coordinates"]['longitude'] <=
                    source_config['coverage']['max_lat'] and
                    source_config['coverage']['min_lon'] <= receiver["coordinates"]['longitude'] <=
                    source_config['coverage']['max_lon']
                    for receiver in receivers
                )

            if available:
                # Get DEM options for this source
                dems = [
                    {
                        "type": dem_type,
                        "resolution": dem_cfg['resolution'],
                        "description": dem_cfg['description']
                    }
                    for dem_type, dem_cfg in source_config['dems'].items()
                ]

                # Find best resolution in this source
                source_best_res = min(d['resolution'] for d in dems)

                available_sources[source_id] = {
                    "description": "Global DEM service" if source_config.get('global') else "Regional DEM service",
                    "name": source_config["name"],
                    "dems": dems,
                    "best_resolution": source_best_res
                }

                # # Update recommendation
                # if not recommendation or (
                #     # Prefer regional over global
                #     (not DEM_SOURCES[recommendation['source']].get('global') and
                #      source_config.get('global')) or
                #     # Same type, higher resolution
                #     (source_best_res > recommendation['resolution'])
                # ):
                #     recommendation = {
                #         "source": source_id,
                #         "dem_type": next(d['type'] for d in dems if d['resolution'] == source_best_res),
                #         "resolution": source_best_res,
                #         "reason": "Highest available resolution" + (" (regional)" if not source_config.get('global') else "")
                #     }

        if not available_sources:
            return jsonify({"error": "No DEM sources available for this area"}), 404

        return jsonify({
            "available_sources": available_sources,
            "recommended": recommendation
        })

    except Exception as e:
        logging.error(f"DEM availability error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/dem', methods=['POST'])
def get_dem_data():
    """Fetch DEM data"""
    data = request.json
    dem_source = data.get('selected_source', 'ot')
    dem_type = data.get('dem_type', 'SRTMGL1')
    print(dem_source, dem_type)
    dem_data = dem_service.get_dem(
        data['coordinates'],
        dem_source,
        dem_type
    )

    if dem_data:
        return jsonify(dem_data)
    return jsonify({"error": "DEM data unavailable"}), 500


@app.route('/alm', methods=['GET'])
def get_almanac():
    data = almanac_service.get_almanac()
    if data:
        return jsonify({'status': 'success', 'data': data})
    return jsonify({'error': 'Almanac unavailable'}), 500


@app.route('/constellations', methods=['GET'])
def get_constellations():
    return jsonify(constellation_manager.get_constellations())


@app.route('/constellations/update', methods=['POST'])
def update_constellations():
    updated = constellation_manager.update_constellations()
    if updated:
        return jsonify({'status': 'success', 'updated': updated})
    return jsonify({'error': 'Update failed'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
