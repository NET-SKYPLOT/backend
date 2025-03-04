import requests
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
import pyproj
from io import BytesIO
import numpy as np
import os
import math
import json
import logging
from config import config, DEM_SOURCES


class DEMService:
    @staticmethod
    def get_dem(coordinates, dem_source, dem_type):
        """Fetch DEM data from different sources"""
        try:
            # Validate input parameters
            if dem_source not in DEM_SOURCES:
                raise ValueError(f"Invalid DEM source: {dem_source}")

            source_config = DEM_SOURCES[dem_source]
            dem_config = source_config['dems'].get(dem_type)

            if not dem_config:
                raise ValueError(f"Invalid DEM type {dem_type} for source {dem_source}")

            # Calculate area of interest
            lats = [c['lat'] for c in coordinates]
            lons = [c['lon'] for c in coordinates]
            avg_lat = sum(lats) / len(lats)
            avg_lon = sum(lons) / len(lons)

            # Calculate bounding box (20km buffer)
            lat_buffer = 0.05  # ~20km at equator
            lon_buffer = 0.05 / math.cos(math.radians(avg_lat))
            bbox = {
                'south': avg_lat - lat_buffer,
                'north': avg_lat + lat_buffer,
                'west': avg_lon - lon_buffer,
                'east': avg_lon + lon_buffer
            }

            # Source-specific handling
            if dem_source == 'ot':
                return DEMService._handle_opentopography(source_config, dem_config, bbox)
            elif dem_source == 'pgp':
                return DEMService._handle_piemonte(source_config, dem_config, bbox)
            else:
                raise ValueError(f"Unsupported DEM source: {dem_source}")

        except Exception as e:
            logging.error(f"DEM Processing Error: {str(e)}")
            return None

    @staticmethod
    def _handle_opentopography(config, dem_config, bbox):
        """Handle OpenTopography API requests"""
        params = {
            "demtype": dem_config['demtype'],
            "south": bbox['south'],
            "north": bbox['north'],
            "west": bbox['west'],
            "east": bbox['east'],
            "outputFormat": "GTiff",
            "API_Key": config['api_key']
        }

        response = requests.get(config['url'], params=params)
        if response.status_code != 200:
            return None

        return DEMService._process_raster_data(
            BytesIO(response.content),
            dem_config,
            'ot',
            dem_config['demtype']
        )

    ### working...
    # def _handle_piemonte(config, dem_config, bbox):
    #     """Handle Piemonte DEM ASC files"""
    #     try:
    #         print('ok')
    #         # Convert WGS84 coordinates to UTM Zone 32N
    #         transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    #         utm_west, utm_south = transformer.transform(bbox['west'], bbox['south'])
    #         utm_east, utm_north = transformer.transform(bbox['east'], bbox['north'])

    #         # Find which ASC files we need to read
    #         matching_files = []
    #         for file_name, file_info in dem_config['files'].items():
    #             file_min_x, file_min_y, file_max_x, file_max_y = file_info['utm_bounds']
    #             if not (utm_east < file_min_x or utm_west > file_max_x or
    #                     utm_north < file_min_y or utm_south > file_max_y):
    #                 matching_files.append(file_name)

    #         print(matching_files)

    #         if not matching_files:
    #             return None

    #         # Read data from all matching files
    #         merged_data = []
    #         for file_name in matching_files:
    #             asc_path = os.path.join(config['data_path'], f"DTM10_{file_name}.asc")
    #             file_data = DEMService._read_asc_file(asc_path, utm_west, utm_east, utm_south, utm_north)
    #             print(file_data[0])
    #             print(type(file_data))
    #             if file_data.any():
    #                 print("appending...")
    #                 merged_data.append(file_data)
    #             else:
    #                 print("something happend...")

    #         # print(merged_data)
    #         if not merged_data:
    #             return None

    #         # Combine data from multiple files
    #         # (This needs proper spatial merging logic depending on file arrangement)
    #         final_elevation = np.vstack(merged_data)

    #         return {
    #             "elevation": final_elevation.tolist(),
    #             "resolution": dem_config['resolution'],
    #             "metadata": {
    #                 "source": "pgp",
    #                 "type": "PGP10",
    #                 "description": dem_config['description'],
    #                 # "files_used": matching_files
    #             }
    #         }

    #     except Exception as e:
    #         logging.error(f"PGP DEM Error: {str(e)}")
    #         return None

    # @staticmethod
    # def _read_asc_file(file_path, utm_west, utm_east, utm_south, utm_north):
    #     """Read portion of ASC file that overlaps with requested bbox"""
    #     try:
    #         with open(file_path, 'r') as f:
    #             # Read header
    #             ncols = int(f.readline().split()[1])
    #             nrows = int(f.readline().split()[1])
    #             xllcorner = float(f.readline().split()[1].replace(',', '.'))
    #             yllcorner = float(f.readline().split()[1].replace(',', '.'))
    #             cellsize = float(f.readline().split()[1])
    #             nodata = float(f.readline().split()[1])

    #             print(ncols, nrows)

    #             # Calculate bounds
    #             xurcorner = xllcorner + ncols * cellsize
    #             yurcorner = yllcorner + nrows * cellsize

    #             # Calculate overlap
    #             start_col = max(0, int((utm_west - xllcorner) / cellsize))
    #             end_col = min(ncols, int((utm_east - xllcorner) / cellsize))
    #             start_row = max(0, int((yurcorner - utm_north) / cellsize))
    #             end_row = min(nrows, int((yurcorner - utm_south) / cellsize))

    #             print(start_row, end_row)

    #             # Read relevant portion
    #             data = []
    #             for i, line in enumerate(f):
    #                 if start_row <= i < end_row:
    #                     row = list(map(float, line.strip().split()))
    #                     data.append(row[start_col:end_col])

    #             print(len(data))

    #             return np.array(data)

    #     except Exception as e:
    #         logging.error(f"Error reading {file_path}: {str(e)}")
    #         return None

    ## working too...
    def _handle_piemonte(config, dem_config, bbox):
        """Handle Piemonte DEM ASC files"""
        try:
            print('ok')
            # Convert WGS84 coordinates to UTM Zone 32N
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
            utm_west, utm_south = transformer.transform(bbox['west'], bbox['south'])
            utm_east, utm_north = transformer.transform(bbox['east'], bbox['north'])

            # Find which ASC files we need to read
            matching_files = []
            for file_name, file_info in dem_config['files'].items():
                file_min_x, file_min_y, file_max_x, file_max_y = file_info['utm_bounds']
                if not (utm_east < file_min_x or utm_west > file_max_x or
                        utm_north < file_min_y or utm_south > file_max_y):
                    matching_files.append(file_name)

            print(matching_files)

            if not matching_files:
                return None

            # Sort files geographically (west-to-east, north-to-south)
            sorted_files = DEMService._sort_pgp_files(matching_files, dem_config)

            # Initialize merged data structure
            merged_data = None

            # First pass: Merge east-west adjacent files
            for file_group in DEMService._group_horizontal_neighbors(sorted_files, dem_config):
                row_data = [
                    DEMService._read_asc_file(os.path.join(config['data_path'], f"DTM10_{fn}.asc"), utm_west, utm_east,
                                              utm_south, utm_north) for fn in file_group]
                row_data = [rd for rd in row_data if rd is not None]

                if row_data:
                    # Merge horizontally (axis=1)
                    merged_row = np.hstack(row_data)
                    if merged_data is None:
                        merged_data = merged_row
                    else:
                        # Merge vertically (axis=0)
                        merged_data = np.vstack((merged_data, merged_row))

            return {
                "elevation": merged_data.tolist(),
                "resolution": dem_config['resolution'],
                "metadata": {
                    "source": "pgp",
                    "type": "PGP10",
                    "description": dem_config['description'],
                    # "files_used": matching_files
                }
            }

        except Exception as e:
            logging.error(f"PGP DEM Error: {str(e)}")
            return None

    @staticmethod
    def _sort_pgp_files(filenames, dem_config):
        """Sort files west-to-east, north-to-south"""
        file_bounds = [(fn, dem_config['files'][fn]['utm_bounds']) for fn in filenames]
        # Sort by Y max (north first), then X min (west first)
        return sorted(file_bounds,
                      key=lambda x: (-x[1][3], x[1][0]))

    @staticmethod
    def _group_horizontal_neighbors(sorted_files, dem_config):
        """Group files that are in the same latitude band"""
        groups = []
        current_group = []
        prev_max_y = None

        for fn, bounds in sorted_files:
            _, min_y, _, max_y = bounds
            if prev_max_y is None or abs(max_y - prev_max_y) < 100:  # 100m tolerance
                current_group.append(fn)
            else:
                groups.append(current_group)
                current_group = [fn]
            prev_max_y = max_y

        if current_group:
            groups.append(current_group)

        return groups

    @staticmethod
    def _read_asc_file(file_path, req_west, req_east, req_south, req_north):
        """Read only the overlapping portion of an ASC file"""
        with open(file_path, 'r') as f:
            # Read header
            ncols = int(f.readline().split()[1])
            nrows = int(f.readline().split()[1])
            xllcorner = float(f.readline().split()[1].replace(',', '.'))
            yllcorner = float(f.readline().split()[1].replace(',', '.'))
            cellsize = float(f.readline().split()[1])
            nodata = float(f.readline().split()[1])

            # Calculate file bounds
            file_east = xllcorner + ncols * cellsize
            file_north = yllcorner + nrows * cellsize

            # Calculate overlap window
            col_start = max(0, int((req_west - xllcorner) / cellsize))
            col_end = min(ncols, int((req_east - xllcorner) / cellsize))
            row_start = max(0, int((file_north - req_north) / cellsize))
            row_end = min(nrows, int((file_north - req_south) / cellsize))

            # Read only needed rows
            data = []
            for i, line in enumerate(f):
                if row_start <= i < row_end:
                    if len(data) >= (row_end - row_start):
                        break
                    row = list(map(float, line.strip().split()))
                    data.append(row[col_start:col_end])

            return np.array(data, dtype=np.float32)

    @staticmethod
    def _process_raster_data(file_bytes, dem_config, source, dem_type):
        """Common raster processing for both sources"""
        with rasterio.open(file_bytes) as src:
            return {
                "elevation": src.read(1).tolist(),
                # "transform": {
                #     "a": src.transform.a,
                #     "b": src.transform.b,
                #     "c": src.transform.c,
                #     "d": src.transform.d,
                #     "e": src.transform.e,
                #     "f": src.transform.f
                # },
                # "crs": str(src.crs),
                "resolution": dem_config['resolution'],
                "metadata": {
                    "source": source,
                    "type": dem_type,
                    "description": dem_config['description']
                }
            }
