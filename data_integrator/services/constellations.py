import json
import requests
import logging


class ConstellationManager:
    def __init__(self):
        self.config_file = "config.json"
        self.constellations = self._load_constellations()

    def _load_constellations(self):
        with open(self.config_file) as f:
            return json.load(f)['constellations']

    def _prune_satellite(self, sat_data):
        return {
            'name': sat_data.get('OBJECT_NAME', ''),
            'norad_id': sat_data.get('NORAD_CAT_ID', ''),
            'object_id': sat_data.get('OBJECT_ID', '')
        }

    def update_constellations(self):
        updated = []
        print(self.constellations.items())
        for const_name, const_data in self.constellations.items():
            try:
                response = requests.get(const_data['url'])
                if response.status_code == 200:
                    satellites = [
                        self._prune_satellite(sat)
                        for sat in response.json()
                    ]
                    self.constellations[const_name]['sats'] = satellites
                    updated.append(const_name)
            except Exception as e:
                logging.error(f"Error updating {const_name}: {str(e)}")

        if updated:
            self._save_config()
        return updated

    def _save_config(self):
        with open(self.config_file, 'r+') as f:
            config = json.load(f)
            config['constellations'] = self.constellations
            f.seek(0)
            json.dump(config, f, indent=2)
            f.truncate()

    def get_constellations(self):
        return self.constellations
