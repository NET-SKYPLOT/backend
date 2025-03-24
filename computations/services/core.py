from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from sgp4.api import Satrec, WGS84, jday
import numpy.typing as npt
import numpy.linalg as LA
from shapely.geometry import Point, Polygon, LineString
import math

from skyfield.api import Loader, EarthSatellite


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


def geodetic2ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """Convert geodetic coordinates (lat [deg], lon [deg], alt [km]) to ECEF coordinates (km)."""
    # WGS84 parameters
    a = 6378.137       # Earth's equatorial radius in km
    e2 = 0.00669437999014  # Earth's eccentricity squared
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = ((1 - e2) * N + alt_km) * np.sin(lat)
    return np.array([x, y, z])


class Satellite:
    def __init__(self, satrec: Satrec, norad_id: int, prn: str, epoch: datetime, constellation):
        self.satrec = satrec
        self.norad_id = norad_id
        self.prn = prn
        self.epoch = epoch
        self.constellation = constellation
        # These will store the original TLE lines for Skyfield cross-checking
        self._tle_line1 = None
        self._tle_line2 = None

    @classmethod
    def from_tle(cls, line1: str, line2: str, prn: str, constellation):
        """Create satellite from TLE lines"""
        satrec = Satrec.twoline2rv(line1, line2, WGS84)
        norad_id = int(line2[2:7])
        epoch = cls._parse_tle_epoch(line1)
        instance = cls(satrec, norad_id, prn, epoch, constellation)
        instance.store_tle(line1, line2)
        return instance

    @classmethod
    def from_params(cls, params: Dict):
        """Create satellite from individual parameters"""
        satrec = Satrec()

        # Convert parameters to TLE-like format
        epoch_jd = (datetime.fromisoformat(params['EPOCH']) - datetime(1949, 12, 31)).days + 0.5
        no_kozai = (params['MEAN_MOTION'] * 2 * np.pi) / 1440.0  # rad/min

        satrec.sgp4init(
            WGS84,
            'i',
            params['NORAD_CAT_ID'],
            epoch_jd,
            params['BSTAR'],
            params['MEAN_MOTION_DOT'],
            params['MEAN_MOTION_DDOT'],
            params['ECCENTRICITY'],
            np.deg2rad(params['ARG_OF_PERICENTER']),
            np.deg2rad(params['INCLINATION']),
            np.deg2rad(params['MEAN_ANOMALY']),
            no_kozai,
            np.deg2rad(params['RA_OF_ASC_NODE'])
        )

        if satrec.error != 0:
            raise ValueError(f"SGP4 init failed: {satrec.error}")

        instance = cls(satrec, params['NORAD_CAT_ID'],
                       params['OBJECT_NAME'].split('PRN ')[-1].strip(')'),
                       datetime.fromisoformat(params['EPOCH']),
                       None)
        # Optionally store TLE lines if available in params
        if 'TLE_LINE1' in params and 'TLE_LINE2' in params:
            instance.store_tle(params['TLE_LINE1'], params['TLE_LINE2'])
        return instance

    @staticmethod
    def _parse_tle_epoch(line1: str) -> datetime:
        """Parse TLE epoch date from line1"""
        epoch_str = line1[18:32]
        year = int(epoch_str[:2]) + (2000 if int(epoch_str[:2]) < 57 else 1900)
        day_of_year = float(epoch_str[2:])
        base_date = datetime(year, 1, 1)
        return base_date + timedelta(days=day_of_year - 1)

    def get_position(self, dt: datetime) -> np.ndarray:
        """Calculate ECEF position for given datetime using SGP4 propagation.
           Returns a NumPy array [x, y, z] in kilometers."""
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        jd, fr = jday(*dt.timetuple()[:6])
        error_code, r, _ = self.satrec.sgp4(jd, fr)
        if error_code != 0:
            raise ValueError(f"SGP4 propagation failed: {self.satrec.error}")
        return np.array(r)

    @staticmethod
    def ecef2lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF coordinates to geodetic coordinates (lat, lon, alt) using WGS84 model.
           Returns latitude (deg), longitude (deg), altitude (km)."""
        a = 6378.137       # Earth's equatorial radius in km
        e2 = 0.00669437999014  # Earth's eccentricity squared
        r = np.sqrt(x*x + y*y)
        # Initial guess for latitude
        lat = np.arctan2(z, r)
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            alt = r / np.cos(lat) - N
            lat = np.arctan2(z, r * (1 - e2 * N/(N+alt)))
        lon = np.arctan2(y, x)
        return np.degrees(lat), np.degrees(lon), alt

    def get_geodetic_position(self, dt: datetime) -> Tuple[float, float, float]:
        """Compute geodetic position (latitude, longitude, altitude) for the given datetime.
           Uses the SGP4-computed ECEF position and converts it to lat, lon, alt."""
        x, y, z = self.get_position(dt)
        return Satellite.ecef2lla(x, y, z)

    def get_position_skyfield(self, dt: datetime) -> Tuple[float, float, float]:
        """Cross-check satellite position using Skyfield for the given datetime.
           Returns geodetic coordinates (latitude, longitude, altitude)."""
        load = Loader('/tmp/skyfield_data')  # Adjust this path as needed
        ts = load.timescale()
        t_sf = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        if not self._tle_line1 or not self._tle_line2:
            raise AttributeError("TLE lines not stored in the Satellite instance for Skyfield cross-check.")
        sat_sf = EarthSatellite(self._tle_line1, self._tle_line2, f"Sat {self.norad_id}", ts)
        subpoint = sat_sf.at(t_sf).subpoint()
        return subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km

    def store_tle(self, line1: str, line2: str):
        """Store the original TLE lines for use with Skyfield cross-checking."""
        self._tle_line1 = line1
        self._tle_line2 = line2




class OrbitalCalculator:
    def __init__(self, almanac_data: Dict):
        self.satellites = self._parse_almanac(almanac_data)

    def _parse_almanac(self, data: Dict) -> Dict[str, Satellite]:
        """Parse almanac data into satellite objects"""
        satellites = {}
        for entry in data.get('data', []):
            try:
                # Prioritize TLE lines if available
                if 'line1' in entry and 'line2' in entry:
                    sat = Satellite.from_tle(
                        line1=entry['line1'],
                        line2=entry['line2'],
                        prn=entry['OBJECT_NAME'],
                        constellation=entry["constellation"]
                    )
                else:
                    sat = Satellite.from_params(entry)
                satellites[sat.norad_id] = sat
            except (KeyError, ValueError) as e:
                print(f"Error parsing satellite {entry.get('NORAD_CAT_ID', 'unknown')}: {str(e)}")
        return satellites


class VisibilityAnalyzer:
    def __init__(self, dem_data: Dict, obstacles: List[Dict], cutoff: int):
        self.elevation_grid = np.array(dem_data['elevation'])
        self.resolution = dem_data['resolution']  # meters per cell
        self.obstacles = self._parse_obstacles(obstacles)
        self.cutoff = cutoff

    def _parse_obstacles(self, obstacles: List[Dict]) -> List[Dict]:
        """Convert obstacles to a consistent internal format.
        Each obstacle is stored with:
          - 'vertices': an array of [latitude, longitude]
          - 'height': extra (relative) height to add to DEM at that location.
        """
        parsed = []
        for obs in obstacles:
            vertices = np.array([[v['latitude'], v['longitude']] for v in obs['vertices'][:-1]])
            parsed.append({
                'vertices': vertices,
                'height': obs['height']
            })
        return parsed

    def _compute_bbox(self, receiver_lla: Dict) -> Dict:
        """Compute geographic bounding box for the DEM (receiver is at center)."""
        lat0 = receiver_lla['latitude']
        lon0 = receiver_lla['longitude']
        rows, cols = self.elevation_grid.shape
        half_height = (rows * self.resolution) / 2.0  # meters
        half_width = (cols * self.resolution) / 2.0  # meters
        dlat = half_height / 111320.0  # ~meters per degree latitude
        dlon = half_width / (111320.0 * math.cos(math.radians(lat0)))
        return {
            'south': lat0 - dlat,
            'north': lat0 + dlat,
            'west': lon0 - dlon,
            'east': lon0 + dlon
        }

    def _get_dem_elevation(self, lat: float, lon: float, receiver_lla: Dict) -> float:
        """Interpolate DEM elevation (in meters) at (lat,lon) using bilinear interpolation."""
        bbox = self._compute_bbox(receiver_lla)
        nrows, ncols = self.elevation_grid.shape
        # Compute fractional row, col indices (DEM grid arranged north-up)
        row_f = (bbox['north'] - lat) / (bbox['north'] - bbox['south']) * (nrows - 1)
        col_f = (lon - bbox['west']) / (bbox['east'] - bbox['west']) * (ncols - 1)

        if row_f < 0 or row_f > nrows - 1 or col_f < 0 or col_f > ncols - 1:
            return 0  # default if out of bounds

        row0 = int(math.floor(row_f))
        row1 = min(row0 + 1, nrows - 1)
        col0 = int(math.floor(col_f))
        col1 = min(col0 + 1, ncols - 1)
        dr = row_f - row0
        dc = col_f - col0
        elev00 = self.elevation_grid[row0, col0]
        elev01 = self.elevation_grid[row0, col1]
        elev10 = self.elevation_grid[row1, col0]
        elev11 = self.elevation_grid[row1, col1]
        elevation = (elev00 * (1 - dc) * (1 - dr) +
                     elev01 * dc * (1 - dr) +
                     elev10 * (1 - dc) * dr +
                     elev11 * dc * dr)
        return elevation

    def _get_obstacle_extra_height(self, lat: float, lon: float) -> float:
        """Return the extra height (in meters) from obstacles at (lat,lon).
        If multiple obstacles cover the point, use the maximum extra height.
        """
        point = Point(lon, lat)
        max_extra = 0
        for obs in self.obstacles:
            poly_points = [(v[1], v[0]) for v in obs['vertices']]  # (lon, lat)
            poly = Polygon(poly_points)
            if poly.contains(point):
                max_extra = max(max_extra, obs['height'])
        return max_extra

    def _get_unified_elevation(self, lat: float, lon: float, receiver_lla: Dict) -> float:
        """Compute unified elevation = DEM elevation + extra obstacle height (if any)."""
        dem_elev = self._get_dem_elevation(lat, lon, receiver_lla)
        extra = self._get_obstacle_extra_height(lat, lon)
        return dem_elev + extra

    def _enu_to_lla(self, receiver_lla: Dict, enu: npt.NDArray) -> Tuple[float, float, float]:
        """Convert local ENU coordinates (meters) back to geographic coordinates.
        (A simple small-distance approximation.)
        """
        # Approximate conversion factors:
        dlat = enu[1] / 111320.0
        dlon = enu[0] / (111320.0 * math.cos(math.radians(receiver_lla['latitude'])))
        # For altitude, we assume the DEM value at receiver_lla is the receiver's altitude.
        alt = self._get_dem_elevation(receiver_lla['latitude'], receiver_lla['longitude'], receiver_lla)
        return (receiver_lla['latitude'] + dlat, receiver_lla['longitude'] + dlon, alt)

    def _create_elevation_profile(self, receiver_lla: Dict, los_vector: npt.NDArray) -> npt.NDArray:
        """Sample the unified elevation along the line-of-sight (LOS) up to 20 km."""
        max_dist = 20 * 1000  # 20 km in meters
        step_size = self.resolution
        steps = int(max_dist / step_size)
        profile = []
        for i in range(steps):
            dist = i * step_size
            point = los_vector * dist  # ENU offset
            lat_pt, lon_pt, _ = self._enu_to_lla(receiver_lla, point)
            elevation = self._get_unified_elevation(lat_pt, lon_pt, receiver_lla)
            profile.append((dist, elevation))
        return np.array(profile)

    def _check_terrain(self, profile: npt.NDArray) -> bool:
        """Check if the LOS ray clears the unified terrain using a fixed 5° mask angle."""
        receiver_height = profile[0, 1]
        for dist, elev in profile:
            los_height = receiver_height + dist * math.tan(math.radians(self.cutoff))
            if elev > los_height:
                return False
        return True

    def _line_intersects_polygon(self, los_vector: npt.NDArray, polygon_enu: npt.NDArray) -> bool:
        """Check if the horizontal LOS (in ENU) intersects a polygon defined in ENU.
        LOS is assumed to extend to 20 km.
        """
        max_dist = 20 * 1000  # 20 km
        line_end = los_vector * max_dist
        los_line = LineString([(0, 0), (line_end[0], line_end[1])])
        poly = Polygon([(pt[0], pt[1]) for pt in polygon_enu])
        return los_line.intersects(poly)

    def _lla_to_enu(self, receiver_lla: Dict, point_lla: Dict) -> npt.NDArray:
        """Convert a point (lat,lon) to ENU coordinates relative to receiver.
        Uses our helper lla_to_ecef and then a simple ENU conversion.
        """
        rec_ecef = lla_to_ecef(receiver_lla)
        point_ecef = lla_to_ecef(point_lla)
        lat = math.radians(receiver_lla['latitude'])
        lon = math.radians(receiver_lla['longitude'])
        R = np.array([
            [-math.sin(lon), math.cos(lon), 0],
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
        ])
        return R.dot((point_ecef - rec_ecef) * 1000)  # result in meters

    def _check_obstacles(self, receiver_lla: Dict, los_vector: npt.NDArray) -> bool:
        """Use ENU geometry to check if the LOS ray intersects any obstacle polygon."""
        for obs in self.obstacles:
            vertices_enu = np.array([
                self._lla_to_enu(receiver_lla, {'latitude': v[0], 'longitude': v[1]})
                for v in obs['vertices']
            ])
            if self._line_intersects_polygon(los_vector, vertices_enu):
                return False
        return True

    def check_visibility(self, receiver_lla: Dict, satellite_ecef: npt.NDArray) -> bool:
        """Determine satellite visibility considering unified terrain (DEM + obstacles).
        The receiver's elevation is derived from the DEM.
        """
        rec_ecef = lla_to_ecef(receiver_lla)
        enu = self._ecef_to_enu(receiver_lla, satellite_ecef)
        los_vector = enu / LA.norm(enu)
        profile = self._create_elevation_profile(receiver_lla, los_vector)
        if not self._check_terrain(profile):
            return False
        if not self._check_obstacles(receiver_lla, los_vector):
            return False
        return True

    def _ecef_to_enu(self, receiver_lla: Dict, point_ecef: npt.NDArray) -> npt.NDArray:
        """Convert ECEF (km) to local ENU (meters) relative to receiver."""
        lat = math.radians(receiver_lla['latitude'])
        lon = math.radians(receiver_lla['longitude'])
        R = np.array([
            [-math.sin(lon), math.cos(lon), 0],
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
        ])
        rec_ecef = lla_to_ecef(receiver_lla)
        return R.dot((point_ecef - rec_ecef) * 1000)


# =============================
# SkyplotGenerator Implementation
# =============================
class SkyplotGenerator:
    @staticmethod
    def calculate_azimuth_elevation(receiver_lla: Dict, satellite_ecef: npt.NDArray) -> Tuple[float, float]:
        """Convert satellite ECEF to azimuth/elevation angles relative to receiver.
        Uses our own lla_to_ecef helper.
        """
        rec_ecef = lla_to_ecef(receiver_lla)
        vec = satellite_ecef - rec_ecef
        lat = math.radians(receiver_lla['latitude'])
        lon = math.radians(receiver_lla['longitude'])
        R = np.array([
            [-math.sin(lon), -math.sin(lat) * math.cos(lon), math.cos(lat) * math.cos(lon)],
            [math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat) * math.sin(lon)],
            [0, math.cos(lat), math.sin(lat)]
        ])
        enu = R.dot(vec)
        e, n, u = enu
        azimuth = (math.degrees(math.atan2(e, n)) + 360) % 360
        elevation = math.degrees(math.atan2(u, math.sqrt(e ** 2 + n ** 2)))
        return azimuth, elevation


# =============================
# DOPCalculator Implementation
# =============================
class DOPCalculator:
    @staticmethod
    def calculate_dop(visible_sats: List[Tuple[str, npt.NDArray]], receiver_ecef: npt.NDArray) -> Dict[str, float]:
        """
        Compute DOP metrics given a list of visible satellites.
        Each element in visible_sats is a tuple (constellation, sat_ecef).
        We form a design matrix H from unit vectors from receiver to each satellite.
        Returns gdop, pdop, hdop, vdop.
        If fewer than 4 satellites, return large values.
        """
        if len(visible_sats) < 4:
            return {'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9}

        H = []
        for cons, sat_ecef in visible_sats:
            vec = sat_ecef - receiver_ecef
            r = LA.norm(vec)
            if r == 0:
                continue
            u = vec / r
            # The row is [u_x, u_y, u_z, 1] (1 for clock bias)
            H.append([u[0], u[1], u[2], 1])
        H = np.array(H)
        try:
            Q = LA.inv(H.T.dot(H))
            gdop = math.sqrt(np.trace(Q))
            pdop = math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
            hdop = math.sqrt(Q[0, 0] + Q[1, 1])
            vdop = math.sqrt(Q[2, 2])
            return {'gdop': gdop, 'pdop': pdop, 'hdop': hdop, 'vdop': vdop}
        except LA.LinAlgError:
            return {'gdop': 99.9, 'pdop': 99.9, 'hdop': 99.9, 'vdop': 99.9}




class ComputationPipeline:
    def __init__(self, almanac_data: Dict, dem_data: Dict, constellations: List[str], obstacles: List[Dict], cutoff: int):
        self.constellations = constellations
        self.orbital_calc = OrbitalCalculator(almanac_data)
        self.visibility_analyzer = VisibilityAnalyzer(dem_data, obstacles, cutoff)
        self.cutoff = cutoff
        self.intervals = 10

    def process_receiver(self, receiver: Dict, start_time: datetime, duration: timedelta) -> Dict:
        """
        Process a receiver scenario and build the final payload.
        Payload structure:
          {
            "status": "success",
            "request_id": "string",
            "planning_details": { ... },
            "receivers": [
              {
                "id": "string",
                "role": "string",
                "coordinates": { "latitude": ..., "longitude": ..., "height": ... },
                "visibility": { <constellation_name>: [ { "time": "string", "count": number }, ... ] },
                "dop": { "time": [...], "gdop": [...], "pdop": [...], "hdop": [...], "vdop": [...] },
                "common_visibility": {},
                "common_dop": {},
                "skyplot_data": { "satellites": [ ... ] },
                "raw_visible": { <time>: [ (constellation, sat_ecef, sat_id), ... ], ... }  // internal field for common computation
              }
            ]
          }
        """
        planning_details = {
            "start_datetime": start_time.isoformat(),
            "duration_hours": duration.total_seconds() / 3600,
            "interval_minutes": self.intervals,
            "application": "GNSS Planning"
        }
        receiver_lla = receiver["coordinates"]
        # Derive receiver height from DEM (interpolated at receiver location)
        receiver_lla["height"] = self.visibility_analyzer._get_dem_elevation(receiver_lla["latitude"],
                                                                             receiver_lla["longitude"],
                                                                             receiver_lla)
        receiver_ecef = lla_to_ecef(receiver_lla)
        time_steps = self._generate_time_steps(start_time, duration, step_minutes=self.intervals)
        # For each time step, store visible satellites (aggregated) and raw visible satellites (with sat_id)
        all_visible_sats = {t.isoformat(): [] for t in time_steps}
        raw_visible = {t.isoformat(): [] for t in time_steps}
        sat_pos_for_wv = {t.isoformat(): [] for t in time_steps}
        skyplot_data = {}  # Keyed by satellite PRN
        for sat in self.orbital_calc.satellites.values():
            if sat.constellation not in self.constellations:
                continue
            sat_traj = []
            for dt in time_steps:
                try:
                    # sat_ecef = sat.get_position(dt)  # Expected as np.ndarray in km

                    lat, lon, alt = sat.get_position_skyfield(dt)
                    # Convert geodetic to ECEF (in km) to match the previous expected format
                    sat_ecef = geodetic2ecef(lat, lon, alt)

                    sat_pos_for_wv[dt.isoformat()].append((sat.constellation, sat_ecef, sat.prn))
                    visible = self.visibility_analyzer.check_visibility(receiver_lla, sat_ecef)
                    az, el = SkyplotGenerator.calculate_azimuth_elevation(receiver_lla, sat_ecef)
                    # sat_traj.append({
                    #     "time": dt.isoformat(),
                    #     "azimuth": round(az, 2),
                    #     "elevation": round(el, 2),
                    #     "visible": visible
                    # })
                    if visible and el >= self.cutoff:
                        sat_traj.append({
                            "time": dt.isoformat(),
                            "azimuth": round(az, 2),
                            "elevation": round(el, 2),
                            "visible": visible
                        })
                        all_visible_sats[dt.isoformat()].append((sat.constellation, sat_ecef))
                        raw_visible[dt.isoformat()].append((sat.constellation, sat_ecef, sat.prn))
                except Exception as e:
                    print(f"Error processing {sat.prn} at {dt}: {str(e)}")
                    continue
            skyplot_data[sat.prn] = {
                "constellation": sat.constellation,
                "satellite_id": sat.prn,
                "trajectory": sat_traj
            }
        # Build DOP values per time step
        dop_list = []
        for dt_iso, visible_list in all_visible_sats.items():
            dop = DOPCalculator.calculate_dop(visible_list, receiver_ecef)
            dop_list.append({
                "time": dt_iso,
                "gdop": dop["gdop"],
                "pdop": dop["pdop"],
                "hdop": dop["hdop"],
                "vdop": dop["vdop"]
            })
        # Build visibility counts per constellation
        vis_by_const = {}
        for dt_iso, vis_list in all_visible_sats.items():
            counts = {}
            for cons, _ in vis_list:
                counts[cons] = counts.get(cons, 0) + 1
            for cons, cnt in counts.items():
                if cons not in vis_by_const:
                    vis_by_const[cons] = []
                vis_by_const[cons].append({"time": dt_iso, "count": cnt})
        # Assemble receiver payload
        receiver_payload = {
            "id": receiver.get("id", "unknown"),
            "role": receiver.get("role", ""),
            "coordinates": receiver_lla,
            "visibility": vis_by_const,
            "dop": {
                "time": [entry["time"] for entry in dop_list],
                "gdop": [entry["gdop"] for entry in dop_list],
                "pdop": [entry["pdop"] for entry in dop_list],
                "hdop": [entry["hdop"] for entry in dop_list],
                "vdop": [entry["vdop"] for entry in dop_list],
            },
            "common_visibility": {},
            "common_dop": {},
            "skyplot_data": {
                "satellites": list(skyplot_data.values())
            },
            "raw_visible": raw_visible  # Internal field for computing common visibility/dop
        }
        payload = {
            "status": "success",
            "request_id": "req_123",  # Placeholder; generate as needed.
            "planning_details": planning_details,
            "receivers": [receiver_payload],
            "satellites_positions": sat_pos_for_wv
        }
        return payload

    def _generate_time_steps(self, start: datetime, duration: timedelta, step_minutes: int) -> List[datetime]:
        """Generate time steps at custom minute intervals."""
        step = timedelta(minutes=step_minutes)
        total_seconds = duration.total_seconds()
        step_seconds = step_minutes * 60
        
        # Calculate number of steps including partial interval
        num_steps = int(total_seconds // step_seconds) + 1  # +1 ensures we include start time
        
        return [start + n * step for n in range(num_steps)]

    @staticmethod
    def _lla_to_ecef(lla: Dict) -> np.ndarray:
        """Static version of lla_to_ecef for internal use."""
        return lla_to_ecef(lla)
