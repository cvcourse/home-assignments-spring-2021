#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]

    points = cv2.goodFeaturesToTrack(image_0,
                                     maxCorners=300, qualityLevel=0.001, minDistance=10, blockSize=10).squeeze(1)
    ids = np.array(range(len(points)));
    sizes = np.array([10] * len(points))
    corners = FrameCorners(ids, points, sizes)

    builder.set_corners_at_frame(0, corners)

    corners_count = len(points)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        i0 = cv2.convertScaleAbs(image_0, alpha=255)
        i1 = cv2.convertScaleAbs(image_1, alpha=255)
        tracked_points, status, err = cv2.calcOpticalFlowPyrLK(i0, i1, points, None)
        status = status.squeeze(1)
        tracked_points = tracked_points[status == 1]
        tracked_ids = ids[status == 1]

        new_points = cv2.goodFeaturesToTrack(image_1, maxCorners=300, qualityLevel=0.001, minDistance=10, blockSize=10).squeeze(1)
        dist = np.linalg.norm(tracked_points[None, :] - new_points[:, None], axis=2)
        new_points = new_points[np.min(dist, axis=1) >= 10, :]
        new_ids = np.array(range(corners_count, corners_count + len(new_points)), dtype=np.int32);
        corners_count += len(new_points)
        tracked_points = np.concatenate((tracked_points, new_points))

        points = tracked_points[:min(300, len(tracked_points))]
        ids = np.append(tracked_ids, new_ids, axis=0)[:len(points)]
        sizes = np.array([10] * len(points))
        corners = FrameCorners(ids, points, sizes)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
