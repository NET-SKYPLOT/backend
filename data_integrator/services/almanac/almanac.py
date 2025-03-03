import json
import requests
from datetime import datetime, timedelta
import os


class AlmanacService:
    CACHE_FILE = "services/almanac/alm.json"

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        with open('config.json') as f:
            return json.load(f)['almanac'][0]

    def _is_stale(self, cached_data):
        if not os.path.exists(self.CACHE_FILE):
            return True
        last_fetched = datetime.fromisoformat(cached_data['timestamp'])
        return datetime.now() - last_fetched > timedelta(
            hours=self.config['refresh_hours']
        )

    def _fetch_fresh_data(self):
        try:
            print(self.config)
            response = requests.get(self.config['url'])
            tle_response = requests.get(self.config['tle_url'])

            if response.status_code == 200 and tle_response.status_code == 200:
                # print(response.json())
                # print(self._parse_tle_data(tle_response.text))
                json_data = self._merge_alamancs(response.json(), self._parse_tle_data(tle_response.text))
                return json_data

        except Exception as e:
            print(f"Almanac fetch error: {str(e)}")
        return None

    def _update_cache(self, data):
        with open(self.CACHE_FILE, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f)

    def _parse_tle_data(self, tle_text):
        """Parse raw TLE data into a list of TLE sets."""
        lines = tle_text.strip().split('\n')
        tle_sets = []

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            tle_sets.append({
                'name': name,
                'line1': line1,
                'line2': line2
            })

        return tle_sets

    def _merge_alamancs(self, json_data, tle_data):
        # Create a lookup dictionary from list2 keyed by the value of "name"
        lookup = {item["name"]: item for item in tle_data}

        # Update each dictionary in list1 with the corresponding dictionary from list2 if there's a match.
        for item in json_data:
            object_name = item.get("OBJECT_NAME")
            if object_name in lookup:
                # Use update to add key-value pairs from the matching dictionary.
                # This will overwrite any duplicate keys in list1 with those from list2.
                data = lookup[object_name]
                additional_data = {k: v for k, v in data.items() if k != "name"}
                item.update(additional_data)

        return json_data

    def get_almanac(self):
        # Try to read cached data
        try:
            with open(self.CACHE_FILE) as f:
                cached = json.load(f)
                # print(self._is_stale(cached))
                if not self._is_stale(cached):
                    return cached['data']
        except FileNotFoundError:
            pass

        # Fetch fresh data if cache is stale/missing
        fresh_data = self._fetch_fresh_data()
        # print(fresh_data)
        if fresh_data:
            self._update_cache(fresh_data)
            return fresh_data

        # Fallback to cache even if stale
        try:
            return cached['data']
        except UnboundLocalError:
            return None
