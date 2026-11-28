"""Which neighbourhood is a point in, and how far apart are two points.

Hand-written ray casting over the SPD MCPP polygons, plus haversine for distances.
Used by Task 4 (hotel -> neighbourhood, hotels within 2 km) and Task 8.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SPD Micro Community Policing Plans from Seattle City GIS, downloaded in EPSG:2926
# and reprojected once to WGS84 so coords are [lon, lat]. 58 features, all Polygon, no holes
GEOJSON_PATH = ROOT / "data" / "geo" / "mcpp.geojson"
NAME_FIELD = "neighborhood"
EARTH_RADIUS_KM = 6371.0088     # mean earth radius, what most haversine examples use

_features = None                # loaded the first time someone asks, then reused


# read the geojson once and keep (name, bbox, polygons) for each area
def _load():
    global _features
    if _features is None:
        data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
        _features = [_prepare(f["properties"][NAME_FIELD], f["geometry"]) for f in data["features"]]
        print(f"[geo] loaded {len(_features)} neighbourhoods")
    return _features


# turn a geometry into a list of polygons (each one a list of rings) plus its bounding box
def _prepare(name, geometry):
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"[geo] {name} is a {geometry['type']}, expected a polygon")
    lons = [p[0] for poly in polygons for p in poly[0]]
    lats = [p[1] for poly in polygons for p in poly[0]]
    return name, (min(lons), min(lats), max(lons), max(lats)), polygons


# shoot a ray to the right and count how many edges it crosses, odd means inside
def _point_in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[j][0], ring[j][1]
        # only edges that straddle our latitude can be crossed
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


# inside the outer ring and not inside any of the holes
def _point_in_polygon(lon, lat, rings):
    if not _point_in_ring(lon, lat, rings[0]):
        return False
    return not any(_point_in_ring(lon, lat, hole) for hole in rings[1:])


# a multipolygon counts if the point is in any of its parts
def _point_in_parts(lon, lat, polygons):
    return any(_point_in_polygon(lon, lat, rings) for rings in polygons)


# which neighbourhood a lat/lon falls in, or None if it's outside seattle
def find_neighbourhood(lat, lon):
    for name, (min_lon, min_lat, max_lon, max_lat), polygons in _load():
        # cheap box check first, skips most of the 58 areas straight away
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        if _point_in_parts(lon, lat, polygons):
            return name
    return None


# straight-line distance over the earth's surface in km
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


if __name__ == "__main__":
    for label, lat, lon in [("Space Needle", 47.6205, -122.3493), ("UW", 47.6553, -122.3035),
                            ("Pike Place", 47.6097, -122.3422), ("Bellevue", 47.6101, -122.2015)]:
        print(f"{label}: {find_neighbourhood(lat, lon)}")
    print(f"Space Needle -> Pike Place: {haversine_km(47.6205, -122.3493, 47.6097, -122.3422):.2f} km")
