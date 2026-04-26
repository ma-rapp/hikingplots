import time

import diskcache
import overpass
from geopy.geocoders import Nominatim

from hikingplots.plot.map import MapSection


class GeoLocator(object):
    _cache = diskcache.Cache("cache/geolocator")

    @staticmethod
    @_cache.memoize()
    def lookup(latitude: float, longitude: float):
        user_agent = "hikingplots.geolocator"
        geolocator = Nominatim(user_agent=user_agent)
        return geolocator.reverse(
            (latitude, longitude), exactly_one=True, language="de-DE"
        )

    @staticmethod
    def get_points_of_interest(map_section: MapSection) -> list[dict]:
        hut_regex = r"(hütte)|(alpe)|(rifugio)|(haus)|(alm([^a-z]|$))"
        water_body_exclude = (
            '["amenity"!="kneipp_water_cure"]'
            '["construction:waterway"!="weir"]'
            '["location"!="underground"]'
            '["man_made"!="pipeline"]'
            '["substance"!="sewer"]'
            '["tunnel"!="culvert"]'
            '["water"!="wastewater"]'
            '["waterway"!="weir"]'
        )
        return GeoLocator.get_named_elements(
            map_section,
            filters=[
                # mountain peaks and saddles
                {"type": "node", "filter": "[natural=peak]"},
                {"type": "node", "filter": "[natural=saddle]"},
                # alpine huts and alpine restaurants
                {
                    "type": "node",
                    "filter": f"[tourism=alpine_hut][name~'{hut_regex}',i]",
                },
                {
                    "type": "way",
                    "filter": f"[tourism=alpine_hut][name~'{hut_regex}',i]",
                },
                {
                    "type": "way",
                    "filter": f"[amenity=restaurant][name~'{hut_regex}',i]",
                },
                # water bodies
                {"type": "node", "filter": f"[natural=cape]{water_body_exclude}"},
                {"type": "way", "filter": f"[natural=water]{water_body_exclude}"},
                {"type": "relation", "filter": f"[type=waterway]{water_body_exclude}"},
                {"type": "relation", "filter": f"[natural=water]{water_body_exclude}"},
                # cave entrances
                {"type": "node", "filter": "[natural=cave_entrance]"},
            ],
        )

    @staticmethod
    def get_name(tags):
        if "name:de" in tags:
            return tags["name:de"]
        elif "name" in tags:
            return tags["name"]
        else:
            return None

    @staticmethod
    @_cache.memoize()
    def get_named_elements(
        map_section: MapSection, filters: list[dict[str, str]]
    ) -> list[dict]:
        """
        filters is a list of dicts with keys "type" (node, way, relation) and "filter" (e.g. "[natural=peak]")
        """
        extended_map_section = map_section.enlarge_absolute(
            latitude=0.001,  # around 100 meters
            longitude=0.001,  # around 100 meters at the equator
        )  # to capture points on the boundary

        overpass_statements = [
            f"{filter_['type']}{filter_['filter']}[name];" for filter_ in filters
        ]

        query = f"""\
[out:json][timeout:25][maxsize:100000000][bbox:{extended_map_section.south_latitude:.6f},{extended_map_section.west_longitude:.6f},{extended_map_section.north_latitude:.6f},{extended_map_section.east_longitude:.6f}];
(
{"\n".join(overpass_statements)}
);
(._;>;);
out;
"""
        api = overpass.API(headers={"User-Agent": "ma-rapp.hikingplots.geolocator"})
        max_retries = 5
        for retry in range(max_retries):
            try:
                response = api.get(query, build=False)
                break
            except overpass.errors.MultipleRequestsError:
                if retry == max_retries - 1:
                    raise
                else:
                    time.sleep(2 * 2**retry)  # exponential backoff
        else:
            assert False, "should never reach this point, but here we are..."

        elements = []
        nodes_by_id = {
            element["id"]: element
            for element in response["elements"]
            if element["type"] == "node"
        }
        ways_by_id = {
            element["id"]: element
            for element in response["elements"]
            if element["type"] == "way"
        }
        for element in response["elements"]:
            if "name" not in element.get("tags", {}):
                continue
            if element["type"] == "node":
                elements.append(
                    {
                        "name": GeoLocator.get_name(element["tags"]),
                        "nodes": [
                            {"latitude": element["lat"], "longitude": element["lon"]}
                        ],
                        "tags": element["tags"],
                    }
                )
            elif element["type"] == "way":
                elements.append(
                    {
                        "name": GeoLocator.get_name(element["tags"]),
                        "nodes": [
                            {
                                "latitude": nodes_by_id[node_id]["lat"],
                                "longitude": nodes_by_id[node_id]["lon"],
                            }
                            for node_id in element["nodes"]
                        ],
                        "tags": element["tags"],
                    }
                )
            elif element["type"] == "relation":
                nodes = []
                for member in element["members"]:
                    if member["type"] == "node":
                        nodes.append(
                            {
                                "latitude": nodes_by_id[member["ref"]]["lat"],
                                "longitude": nodes_by_id[member["ref"]]["lon"],
                            }
                        )
                    elif member["type"] == "way":
                        way = ways_by_id[member["ref"]]
                        nodes.extend(
                            [
                                {
                                    "latitude": nodes_by_id[node_id]["lat"],
                                    "longitude": nodes_by_id[node_id]["lon"],
                                }
                                for node_id in way["nodes"]
                            ]
                        )
                    else:
                        raise NotImplementedError(
                            f"Member type {member['type']} of relation not supported yet"
                        )
                elements.append(
                    {
                        "name": GeoLocator.get_name(element["tags"]),
                        "nodes": nodes,
                        "tags": element["tags"],
                    }
                )
            else:
                raise NotImplementedError(
                    f"Element type {element['type']} not supported yet"
                )

        # filter nodes that are outside the map section
        for element in elements:
            element["nodes"] = [
                node
                for node in element["nodes"]
                if extended_map_section.contains_point(
                    node["latitude"], node["longitude"]
                )
            ]

        # filter elements that have no nodes in the map section
        elements = [element for element in elements if len(element["nodes"]) > 0]

        return elements
