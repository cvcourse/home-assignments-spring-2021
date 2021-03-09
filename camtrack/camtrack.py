#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    view_1_id, view_1 = known_view_1[0], pose_to_view_mat3x4(known_view_1[1])
    view_2_id, view_2 = known_view_2[0], pose_to_view_mat3x4(known_view_2[1])

    view_mats[view_1_id] = view_1
    view_mats[view_2_id] = view_2

    correspondences = build_correspondences(corner_storage[view_1_id], corner_storage[view_2_id], None)
    triangulation_parameters = TriangulationParameters(1, 5, 0)
    points, ids, _ = triangulate_correspondences(correspondences, view_1, view_2,
                                                 intrinsic_mat, TriangulationParameters(0.1, 0.1, 0))

    #corners_0 = corner_storage[0]
    point_cloud_builder = PointCloudBuilder(ids, points)

    unknown_frames = frame_count - 2
    known_frames = {view_1_id, view_2_id}

    while unknown_frames > 0:
        for i in range(frame_count):
            if view_mats[i] is None:
                corners = corner_storage[i]
                interesting_ids, in_corners, in_cloud = np.intersect1d(corners.ids.flatten(), point_cloud_builder.ids.flatten(), return_indices=True)
                points_2d = corners.points[in_corners]
                points_3d = point_cloud_builder.points[in_cloud]

                if len(ids) < 3:
                    continue

                method = cv2.SOLVEPNP_EPNP
                if len(ids) == 3:
                    method = cv2.SOLVEPNP_P3P
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, None,
                                                                 flags=method)
                if not retval:
                    continue
                retval, rvec, tvec = cv2.solvePnP(points_3d[inliers], points_2d[inliers], intrinsic_mat, None,
                                                  rvec=rvec, tvec=tvec, useExtrinsicGuess=True)
                if not retval:
                    continue

                view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                unknown_frames -= 1
                known_frames.add(i)

                outliers = np.delete(interesting_ids, inliers)
                for frame in known_frames:
                    correspondences = build_correspondences(corners, corner_storage[frame], outliers)
                    points_3d, corner_ids, _ = triangulate_correspondences(
                        correspondences, view_mats[i], view_mats[frame], intrinsic_mat, triangulation_parameters)
                    point_cloud_builder.add_points(corner_ids, points_3d)




    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
