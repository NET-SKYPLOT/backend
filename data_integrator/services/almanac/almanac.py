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
            return json.load(f)['almanac']

    def _is_stale(self, cached_data):
        if not os.path.exists(self.CACHE_FILE):
            return True
        last_fetched = datetime.fromisoformat(cached_data['timestamp'])
        return datetime.now() - last_fetched > timedelta(
            hours=self.config[0]['refresh_hours']
        )

    def _fetch_fresh_data(self):
        try:
            print(self.config[0])
            response = requests.get(self.config[0]['url'])
            tle_response = requests.get(self.config[0]['tle_url'])

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


    def get_historical_almanac(self, target_date: datetime):
        """Retrieve historical almanac from Space-Track.org"""
        print("get Historical almanac...")
        try:
            # Authenticate with Space-Track
            session = self._authenticate_spacetrack()
            # print(session)
            # Get TLEs closest to target date
            tle_data = self._query_spacetrack(session, target_date)
            # print("tle_data", tle_data)
            # Convert to compatible format
            return self._convert_to_almanac_format(tle_data, target_date)
            
        except Exception as e:
            print(f"Historical almanac error: {str(e)}")
            return None
        finally:
            if 'session' in locals():
                session.close()

    def _authenticate_spacetrack(self):
        """Authenticate with Space-Track.org"""
        auth_url = "https://www.space-track.org/ajaxauth/login"
        credentials = {
            'identity': self.config[1]['spacetrack_user'],
            'password': self.config[1]['spacetrack_pass']
        }
        
        session = requests.Session()
        response = session.post(auth_url, data=credentials)
        response.raise_for_status()
        return session

    def _query_spacetrack(self, session, target_date):
        """Query Space-Track for TLEs around target date"""
        # Calculate date range (±3 days)
        start_date = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")

        # GNSS NORAD ID ranges
        gnss_ranges = [
            "25000,27000",  # GPS
            "27000,29000",  # GLONASS
            "36000,38000",  # BeiDou
            "41000,42000"   # Galileo
        ]

        # Build query URL
        base_url = "https://www.space-track.org/basicspacedata/query/class/gp_history/"
        query = [
            f"EPOCH/{start_date}--{end_date}",
            "format/json",
            "orderby/EPOCH desc",  # Get nearest first
            f"NORAD_CAT_ID/{','.join(gnss_ranges)}"
        ]
        
        response = session.get(f"{base_url}{'/'.join(query)}")
        # print(response.json())
        response.raise_for_status()
        
        return response.text

    def _convert_to_almanac_format(self, tle_text, target_date):
        """Convert raw TLEs to your existing almanac format"""
        tle_sets = self._parse_tle_data(tle_text)
        
        formatted_data = []
        for tle in tle_sets:
            # Extract NORAD ID from line1
            norad_id = int(tle['line1'][2:7])
            
            formatted_data.append({
                "OBJECT_NAME": tle['name'],
                "NORAD_CAT_ID": norad_id,
                "EPOCH": target_date.isoformat(),
                "line1": tle['line1'],
                "line2": tle['line2'],
                # Add constellation based on NORAD ID ranges
                "constellation": self._get_constellation(norad_id)
            })
        
        print("form tle data...", formatted_data)
        return {
            "timestamp": datetime.now().isoformat(),
            "data": formatted_data
        }

    def _get_constellation(self, norad_id: int) -> str:
        """Determine constellation from NORAD ID"""
        if 25000 <= norad_id <= 27000:
            return "GPS"
        elif 27000 < norad_id <= 29000:
            return "GLONASS"
        elif 36000 <= norad_id <= 38000:
            return "BEIDOU"
        elif 41000 <= norad_id <= 42000:
            return "GALILEO"
        return "UNKNOWN"