import datetime
import tempfile

import pytest
import yaml
from gpxpy.gpx import GPX, GPXTrack, GPXTrackPoint, GPXTrackSegment

from hikingplots.plot.track import Track


def _gpx(points):
    segment = GPXTrackSegment(
        points=[
            GPXTrackPoint(
                latitude=point[0],
                longitude=point[1],
                time=datetime.datetime.fromisoformat(point[2])
                if len(point) > 2
                else None,
            )
            for point in points
        ]
    )

    track = GPXTrack()
    track.segments.append(segment)

    gpx = GPX()
    gpx.tracks.append(track)

    return gpx


def test_load_from_folder():
    gpx = _gpx(
        [
            (47.465850, 10.508765, "2025-01-01T10:00:00Z"),
        ]
    )
    metadata = {"type": "hiking", "who": ["Alice", "Bob"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/test.gpx", "w") as gpx_file:
            gpx_file.write(gpx.to_xml())
        with open(f"{tmpdir}/metadata.yaml", "w") as metadata_file:
            yaml.dump(metadata, metadata_file)
        track = Track.from_folder(tmpdir)

    assert track.track_type == "hiking"
    assert track.who == ["Alice", "Bob"]
    assert len(track.waypoints) == 1


@pytest.mark.parametrize(
    "limit_tag, limit_who",
    [
        (None, None),
        ("test", None),
        (None, "Alice"),
        ("test", "Alice"),
        ("other", None),
        (None, "Charlie"),
        ("other", "Charlie"),
    ],
)
def test_load_tracks_limit_tag_who(limit_tag: str | None, limit_who: str | None):
    gpx = _gpx(
        [
            (47.465850, 10.508765, "2025-01-01T10:00:00Z"),
        ]
    )
    metadata = {"type": "hiking", "tags": ["test"], "who": ["Alice", "Bob"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/test.gpx", "w") as gpx_file:
            gpx_file.write(gpx.to_xml())
        with open(f"{tmpdir}/metadata.yaml", "w") as metadata_file:
            yaml.dump(metadata, metadata_file)
        track = Track.from_folder(tmpdir, limit_tag=limit_tag, limit_who=limit_who)

    should_load = limit_tag in [None, "test"] and limit_who in [None, "Alice", "Bob"]

    assert (track is not None) == should_load


def test_track_properties():
    gpx = _gpx(
        [
            (47.465850, 10.508765, "2025-01-02T10:00:00Z"),
        ]
    )
    track = Track(gpx)
    assert track.year == 2025
    assert track.month == 1
    assert track.day == 2
    assert track.country == "Österreich"
    assert track.state == "Tirol"
    assert track.city == "Tannheim"
