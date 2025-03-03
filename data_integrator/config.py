import json
import os

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)

# API Keys
OT_DEM_API_KEY = "517b932fccc829b669db6b210158e497"
PGP_API_KEY = ""

DEM_SOURCES = {
    "ot": {
        "url": "https://portal.opentopography.org/API/globaldem",
        "api_key": OT_DEM_API_KEY,
        "global": True,
        "name": "Open Topography",
        "dems": {
            "SRTMGL3": {
                "demtype": "SRTMGL3",
                "resolution": 90,
                "description": "SRTM GL3 90m"
            },
            "SRTMGL1": {
                "demtype": "SRTMGL1",
                "resolution": 30,
                "description": "SRTM GL1 30m"
            },
            "SRTMGL1_E": {
                "demtype": "SRTMGL1_E",
                "resolution": 30,
                "description": "SRTM GL1 Ellipsoidal 30m"
            },
            "AW3D30": {
                "demtype": "AW3D30",
                "resolution": 30,
                "description": "ALOS World 3D 30m"
            },
            "AW3D30_E": {
                "demtype": "AW3D30_E",
                "resolution": 30,
                "description": "ALOS World 3D Ellipsoidal 30m"
            },
            "SRTM15Plus": {
                "demtype": "SRTM15Plus",
                "resolution": 500,
                "description": "Global Bathymetry SRTM15+ V2.1"
            },
            "NASADEM": {
                "demtype": "NASADEM",
                "resolution": 30,
                "description": "NASADEM Global DEM"
            },
            "COP30": {
                "demtype": "COP30",
                "resolution": 30,
                "description": "Copernicus Global DSM 30m"
            },
            "COP90": {
                "demtype": "COP90",
                "resolution": 90,
                "description": "Copernicus Global DSM 90m"
            },
            "EU_DTM": {
                "demtype": "EU_DTM",
                "resolution": 30,
                "description": "EU DTM 30m"
            },
            "GEDI_L3": {
                "demtype": "GEDI_L3",
                "resolution": 1000,
                "description": "GEDI L3 DTM 1000m"
            },
            "GEBCOIceTopo": {
                "demtype": "GEBCO2021",
                "resolution": 500,
                "description": "GEBCO Ice Topography 500m"
            },
            "GEBCOSubIceTopo": {
                "demtype": "GEBCO2021_sub_ice_topo",
                "resolution": 500,
                "description": "GEBCO Sub-Ice Topography 500m"
            }
        }
    },

    "pgp": {
        "url": "https://www.geoportale.piemonte.it/geonetwork/srv/ita/catalog.search#/metadata/r_piemon:3ffe6b7b-9abe-4459-8305-e444e8eb197c",
        "name": "Piemote Geoportale",
        "data_path": "services/dem/dems/PGP/DTM10",
        "api_key": PGP_API_KEY,
        "global": False,
        "coverage": {
            "min_lat": 44.0,
            "max_lat": 46.5,
            "min_lon": 6.5,
            "max_lon": 9.0
        },
        "dems": {
            "PGP10": {
                "resolution": 10,
                "description": "Piemonte 10m DTM",
                "files": {
                    "nord": {
                        "utm_bounds": [
                            403256.181155,  # min_x (xllcorner)
                            5056949.74709,  # min_y (yllcorner)
                            403256.181155 + (9000 * 10),  # max_x = xllcorner + (ncols * cellsize)
                            5056949.74709 + (8884 * 10)  # max_y = yllcorner + (nrows * cellsize)
                        ],
                        "calculated_bounds": [403256.181155, 5056949.74709, 493256.181155, 5145789.74709]
                    },
                    "cen_ovest": {
                        "utm_bounds": [
                            313256.181155,  # min_x
                            4968109.74709,  # min_y
                            313256.181155 + (10000 * 10),  # max_x
                            4968109.74709 + (8884 * 10)  # max_y
                        ],
                        "calculated_bounds": [313256.181155, 4968109.74709, 413256.181155, 5056949.74709]
                    },
                    "cen_est": {
                        "utm_bounds": [
                            413256.181155,  # min_x
                            4968109.74709,  # min_y
                            413256.181155 + (10000 * 10),  # max_x
                            4968109.74709 + (8884 * 10)  # max_y
                        ],
                        "calculated_bounds": [413256.181155, 4968109.74709, 513256.181155, 5056949.74709]
                    },
                    "sud_ovest": {
                        "utm_bounds": [
                            323256.181155,  # min_x
                            4879269.74709,  # min_y
                            323256.181155 + (9000 * 10),  # max_x
                            4879269.74709 + (8884 * 10)  # max_y
                        ],
                        "calculated_bounds": [323256.181155, 4879269.74709, 413256.181155, 4968109.74709]
                    },
                    "sud_est": {
                        "utm_bounds": [
                            413256.181155,  # min_x
                            4879269.74709,  # min_y
                            413256.181155 + (10374 * 10),  # max_x
                            4879269.74709 + (8884 * 10)  # max_y
                        ],
                        "calculated_bounds": [413256.181155, 4879269.74709, 516996.181155, 4968109.74709]
                    }
                }
            }
        }
    }
}
