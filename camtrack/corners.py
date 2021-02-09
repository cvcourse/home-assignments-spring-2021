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


def _merge_corners(corner_points, new_corner_points, maxCorners):
    new_corner_points = np.array([i for i in new_corner_points if i not in corner_points]) # todo

    corner_points = np.concatenate((corner_points, new_corner_points), axis=0)
    
    np.random.shuffle(corner_points)
    return corner_points[:maxCorners]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    
    image_0 = frame_sequence[0]

    maxCorners = image_0.shape[0] * image_0.shape[1] // 1000
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 7

    corner_points = cv2.goodFeaturesToTrack(image_0, maxCorners, qualityLevel, minDistance, blockSize = blockSize)
    ids = np.arange(0, corner_points.shape[0], 1).reshape((-1, 1))
    sizes = np.full((corner_points.shape[0], 2), blockSize)
    corners = FrameCorners(
        ids,
        corner_points,
        sizes
    )
    builder.set_corners_at_frame(0, corners)

    maxLevel = 3
    maxCount = 10
    epsilon = 0.03
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, maxCount, epsilon)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_0_8u = (image_0 * 255).astype(np.uint8) 
        image_1_8u = (image_1 * 255).astype(np.uint8) 
    
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(image_0_8u, image_1_8u, corner_points, None,
         winSize = (blockSize, blockSize), maxLevel = maxLevel, criteria = criteria)

        corner_points = nextPts[status == 1]
        
        new_corner_points = cv2.goodFeaturesToTrack(image_1, maxCorners, qualityLevel, minDistance, blockSize = blockSize)
        corner_points = _merge_corners(nextPts, new_corner_points, maxCorners)
        ids = np.arange(0, corner_points.shape[0], 1).reshape((-1, 1)) # todo
        sizes = np.full((corner_points.shape[0], 2), blockSize)
        
        corners = FrameCorners(
            ids,
            corner_points,
            sizes
        )

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
