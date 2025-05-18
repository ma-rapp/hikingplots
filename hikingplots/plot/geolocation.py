import re

import diskcache
import overpy
from geopy.geocoders import Nominatim

from hikingplots.plot.map import MapSection


class GeoLocator(object):
    _cache = diskcache.Cache("cache/geolocator")

    @staticmethod
    @_cache.memoize()
    def lookup(latitude, longitude):
        user_agent = "hikingplots.geolocator"
        geolocator = Nominatim(user_agent=user_agent)
        return geolocator.reverse(
            (latitude, longitude), exactly_one=True, language="de-DE"
        )

    @staticmethod
    def get_named_mountain_peaks(map_section):
        return GeoLocator.get_named_elements(
            map_section, type_="node", attribute="natural=peak"
        )

    @staticmethod
    def get_named_mountain_saddles(map_section):
        return GeoLocator.get_named_elements(
            map_section, type_="node", attribute="natural=saddle"
        )

    @staticmethod
    def get_alpine_huts(map_section):
        alpine_huts = GeoLocator.get_named_elements(
            map_section, type_="way", attribute="tourism=alpine_hut"
        ) + GeoLocator.get_named_elements(
            map_section, type_="node", attribute="tourism=alpine_hut"
        )
        restaurants = GeoLocator.get_named_elements(
            map_section, type_="way", attribute="amenity=restaurant"
        )
        alpine_restaurants = [
            r
            for r in restaurants
            if re.search(
                r"(hütte)|(alpe)|(rifugio)|(alm([^a-z]|$))",
                r["tags"].get("name", ""),
                re.IGNORECASE,
            )
        ]
        return alpine_huts + alpine_restaurants

    @staticmethod
    def get_named_water_bodies(map_section):
        water_bodies = (
            GeoLocator.get_named_elements(
                map_section, type_="relation", attribute="type=waterway"
            )
            + GeoLocator.get_named_elements(
                map_section, type_="relation", attribute="natural=water"
            )
            + GeoLocator.get_named_elements(
                map_section, type_="way", attribute="natural=water"
            )
            + GeoLocator.get_named_elements(
                map_section, type_="node", attribute="natural=cape"
            )
        )
        return [
            water_body
            for water_body in water_bodies
            if "zoo" not in water_body["tags"]
            and water_body["tags"].get("amenity") != "kneipp_water_cure"
            and water_body["tags"].get("location") != "underground"
            and water_body["tags"].get("man_made") != "pipeline"
            and water_body["tags"].get("substance") != "sewer"
            and water_body["tags"].get("tunnel") != "culvert"
            and water_body["tags"].get("water") != "wastewater"
        ]

    @staticmethod
    def get_named_cave_entrances(map_section):
        return GeoLocator.get_named_elements(
            map_section, type_="node", attribute="natural=cave_entrance"
        ) + GeoLocator.get_named_elements(
            map_section, type_="way", attribute="natural=cave_entrance"
        )

    @staticmethod
    @_cache.memoize()
    def get_named_elements(map_section: MapSection, type_, attribute):
        extended_map_section = map_section.enlarge_absolute(
            latitude=0.001,  # around 100 meters
            longitude=0.001,  # around 100 meters at the equator
        )  # to capture points on the boundary
        query = f"""\
[out:json][timeout:25][maxsize:100000000];
{type_}
  [{attribute}]({extended_map_section.south_latitude},{extended_map_section.west_longitude},{extended_map_section.north_latitude},{extended_map_section.east_longitude});
  (._;>;);
out;
"""
        api = overpy.Overpass()
        result = api.query(query)
        elements = []
        for node in result.nodes:
            if "name" in node.tags:
                elements.append(
                    {
                        "name": node.tags["name"],
                        "nodes": [{"latitude": node.lat, "longitude": node.lon}],
                        "tags": node.tags,
                    }
                )
        nodes_by_id = {node.id: node for node in result.nodes}
        for way in result.ways:
            if "name" in way.tags:
                elements.append(
                    {
                        "name": way.tags["name"],
                        "nodes": [
                            {
                                "latitude": nodes_by_id[node.id].lat,
                                "longitude": nodes_by_id[node.id].lon,
                            }
                            for node in way.nodes
                            if extended_map_section.contains_point(
                                nodes_by_id[node.id].lat, nodes_by_id[node.id].lon
                            )
                        ],
                        "tags": way.tags,
                    }
                )
        for relation in result.relations:
            if "name" in relation.tags:
                elements.append(
                    {
                        "name": relation.tags["name"],
                        "nodes": [
                            {
                                "latitude": nodes_by_id[node.id].lat,
                                "longitude": nodes_by_id[node.id].lon,
                            }
                            for member in relation.members
                            for node in member._result.nodes
                            if extended_map_section.contains_point(
                                nodes_by_id[node.id].lat, nodes_by_id[node.id].lon
                            )
                        ],
                        "tags": relation.tags,
                    }
                )
        return elements
