import datetime
import tempfile

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
