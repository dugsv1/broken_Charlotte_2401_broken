import geopandas as pd
from pathlib import Path

def getPolyCoords(row, geom, coord_type):
    if coord_type == 'x':
        return list(row[geom].exterior.coords.xy[0])
    elif coord_type == 'y':
        return list(row[geom].exterior.coords.xy[1])

def check_polys(gdf):
    
    has_polygons = any(gdf.geometry.type == 'Polygon')
    print("Contains Polygons:", has_polygons)

    # Check if the GeoDataFrame contains MultiPolygons
    has_multipolygons = any(gdf.geometry.type == 'MultiPolygon')
    print("Contains MultiPolygons:", has_multipolygons)

def multipolygon_to_polygons(geodataframe):
    """
    Convert MultiPolygon geometries to individual Polygon geometries.
    """
    polygons = geodataframe.explode(index_parts=False)
    polygons = polygons.reset_index(drop=True)
    return polygons