from hikingplots.plot.geolocation import GeoLocator
from hikingplots.plot.map import MapSection

SULZFLUH = (47.0126709, 9.8396931)
LUENERKRINNE = (47.0573517, 9.7688625)
GEPATSCH_ALM = (46.8962833, 10.7350495)
LINDAUER_HUETTE = (47.034420, 9.835660)
LUENERSEE_ALPE = (47.046457, 9.753194)
LANDS_END = (50.0662633, -5.7148222)
TILISUNASEE = (47.026615, 9.879949)
ILL = (47.067218, 9.926235)
VILSALPSEE = (47.464168, 10.504131)
GAUABLICKHOEHLE = (47.0221488, 9.8465177)


def test_lookup():
    GeoLocator._cache.clear()
    address = GeoLocator.lookup(*GAUABLICKHOEHLE).raw["address"]
    assert address["country"] == "Österreich"
    assert address["state"] == "Vorarlberg"


def _get_map_section(poi):
    return MapSection(
        north_latitude=poi[0],
        south_latitude=poi[0],
        east_longitude=poi[1],
        west_longitude=poi[1],
    ).enlarge_absolute(latitude=0.002, longitude=0.002)


def test_get_node_mountain_peak():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(_get_map_section(SULZFLUH))
    assert any(
        point["name"] == "Sulzfluh" for point in points_of_interest
    ), points_of_interest


def test_get_node_saddle():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(LUENERKRINNE)
    )
    assert any(
        point["name"] == "Lünerkrinne" for point in points_of_interest
    ), points_of_interest


def test_get_node_hut():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(GEPATSCH_ALM)
    )
    assert any(
        point["name"] == "Gepatsch Alm" for point in points_of_interest
    ), points_of_interest


def test_get_way_hut():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(LINDAUER_HUETTE)
    )
    assert any(
        point["name"] == "Lindauer Hütte" for point in points_of_interest
    ), points_of_interest


def test_get_way_restaurant():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(LUENERSEE_ALPE)
    )
    assert any(
        point["name"] == "Lünersee Alpe" for point in points_of_interest
    ), points_of_interest


def test_get_node_natural_cape():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(_get_map_section(LANDS_END))
    assert any(
        point["name"] == "Land’s End" for point in points_of_interest
    ), points_of_interest


def test_get_way_natural_water():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(TILISUNASEE)
    )
    assert any(
        point["name"] == "Tilisunasee" for point in points_of_interest
    ), points_of_interest


def test_get_relation_waterway():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(_get_map_section(ILL))
    assert any(
        point["name"] == "Ill" for point in points_of_interest
    ), points_of_interest


def test_get_relation_natural_water():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(_get_map_section(VILSALPSEE))
    assert any(
        point["name"] == "Vilsalpsee" for point in points_of_interest
    ), points_of_interest


def test_get_node_natural_cave_entrance():
    GeoLocator._cache.clear()
    points_of_interest = GeoLocator.get_points_of_interest(
        _get_map_section(GAUABLICKHOEHLE)
    )
    assert any(
        point["name"] == "Gauablickhöhle" for point in points_of_interest
    ), points_of_interest
