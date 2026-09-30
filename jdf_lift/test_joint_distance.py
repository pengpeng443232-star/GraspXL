import numpy as np

from joint_distance import RewardWeights, grasp_lift_reward, joint_distance_feature


def test_joint_distance_feature_is_translation_invariant():
    joints = np.zeros((1, 21, 3), np.float32)
    surface = np.asarray([[[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.float32)
    vec1, dist1 = joint_distance_feature(joints, surface)
    offset = np.asarray([0.3, -0.2, 0.1], np.float32)
    vec2, dist2 = joint_distance_feature(joints + offset, surface + offset)
    np.testing.assert_allclose(vec1, vec2, atol=1e-6)
    np.testing.assert_allclose(dist1, dist2, atol=1e-6)


def test_target_lift_with_three_contacts_is_success():
    distances = np.full((1, 21), 0.01, np.float32)
    contacts = np.zeros((1, 16), np.float32)
    contacts[0, [1, 4, 7]] = 1
    reward, terms = grasp_lift_reward(
        distances, contacts, np.asarray([0.65]), np.asarray([0.55]), np.zeros((1, 6)),
        np.zeros((1, 51)), np.zeros((1, 51)), 0.10, np.ones(1), 0.01, 0.12, RewardWeights(),
    )
    assert terms["success"][0] == 1
    assert reward[0] > 10

