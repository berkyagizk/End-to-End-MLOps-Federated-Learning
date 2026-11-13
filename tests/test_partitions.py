import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.utils import MIN_TRAIN_PER_CLASS, build_partition_plan, load_client_data


def make_frame():
    positions = np.arange(2320, dtype=float)
    return pd.DataFrame({
        "Time": positions,
        "V1": positions / 10,
        "Amount": positions + 1,
        "Class": np.r_[np.zeros(2200, dtype=int), np.ones(120, dtype=int)],
    })


def counts(labels):
    return np.bincount(labels, minlength=2).tolist()


class PartitionTests(unittest.TestCase):
    def check_plan(self, frame, plan, num_clients):
        labels = frame["Class"].to_numpy()
        self.assertEqual(len(plan.clients), num_clients)
        rows = np.concatenate([
            *(client.train for client in plan.clients),
            *(client.validation for client in plan.clients),
            plan.final_test,
        ])
        self.assertEqual(len(rows), len(np.unique(rows)), "Overlap across clients or boundaries")
        np.testing.assert_array_equal(
            np.sort(np.r_[rows, plan.excluded_duplicates]), np.arange(len(frame)),
        )
        self.assertFalse(frame.iloc[rows].drop(columns="Class").duplicated().any())
        self.assertTrue(all(n >= 1 for n in counts(labels[plan.final_test])))
        for client in plan.clients:
            self.assertTrue(all(n >= MIN_TRAIN_PER_CLASS for n in counts(labels[client.train])))
            self.assertTrue(all(n >= 1 for n in counts(labels[client.validation])))
        class_totals = np.array([
            counts(labels[np.r_[client.train, client.validation]]) for client in plan.clients
        ])
        self.assertTrue((np.ptp(class_totals, axis=0) <= 1).all())
        for split in ("train", "validation"):
            class_counts = [counts(labels[getattr(client, split)]) for client in plan.clients]
            self.assertTrue((np.ptp(class_counts, axis=0) <= 1).all())

    def check_same_global_membership(self, first, other):
        for split in ("train", "validation"):
            first_rows = np.concatenate([getattr(client, split) for client in first.clients])
            other_rows = np.concatenate([getattr(client, split) for client in other.clients])
            np.testing.assert_array_equal(np.sort(first_rows), np.sort(other_rows), err_msg=split)
        np.testing.assert_array_equal(np.sort(first.final_test), np.sort(other.final_test))

    def test_all_boundaries_and_duplicate_policy_for_supported_counts(self):
        base = make_frame()
        frame = pd.concat([base, base.iloc[[17, 2199, 2220, 2220]]])
        for num_clients in (2, 5, 10):
            with self.subTest(num_clients=num_clients):
                plan = build_partition_plan(frame, num_clients, seed=42)
                self.check_plan(frame, plan, num_clients)
                np.testing.assert_array_equal(plan.excluded_duplicates, np.arange(len(base), len(frame)))

    def test_global_split_membership_independent_of_client_count(self):
        base = make_frame()
        frame = pd.concat([base, base.iloc[[17, 2220]]])
        reference = build_partition_plan(frame, 2, seed=72)
        for num_clients in (2, 5, 10):
            with self.subTest(num_clients=num_clients):
                plan = build_partition_plan(frame, num_clients, seed=72)
                self.check_plan(frame, plan, num_clients)
                self.check_same_global_membership(reference, plan)

    def test_seed_reproducibility(self):
        frame = make_frame()
        first = build_partition_plan(frame, 5, seed=42)
        again = build_partition_plan(frame, 5, seed=42)
        np.testing.assert_array_equal(first.final_test, again.final_test)
        for left, right in zip(first.clients, again.clients):
            np.testing.assert_array_equal(left.train, right.train)
            np.testing.assert_array_equal(left.validation, right.validation)
        other_seed = build_partition_plan(frame, 5, seed=43)
        self.assertNotEqual(set(first.final_test), set(other_seed.final_test))

    def test_loader_returns_assigned_rows_and_scaler_excludes_evaluation_rows(self):
        frame = make_frame()
        plan = build_partition_plan(frame, 5, seed=72)
        client_id = 3
        client = plan.clients[client_id]
        with patch("src.utils.pd.read_csv", return_value=frame):
            train, train_labels, validation, validation_labels = load_client_data("unused", 5, client_id, seed=72)

        np.testing.assert_array_equal(train[:, 1], frame.iloc[client.train]["V1"])
        np.testing.assert_array_equal(validation[:, 1], frame.iloc[client.validation]["V1"])
        np.testing.assert_array_equal(train_labels, frame.iloc[client.train]["Class"])
        np.testing.assert_array_equal(validation_labels, frame.iloc[client.validation]["Class"])
        pooled_train = frame.iloc[np.concatenate([part.train for part in plan.clients])]
        for index, column in ((0, "Time"), (2, "Amount")):
            mean, scale = pooled_train[column].mean(), pooled_train[column].std(ddof=0)
            np.testing.assert_allclose(train[:, index], (frame.iloc[client.train][column] - mean) / scale)
            np.testing.assert_allclose(validation[:, index], (frame.iloc[client.validation][column] - mean) / scale)

        changed = frame.copy()
        held_out = np.concatenate([plan.final_test, *(part.validation for part in plan.clients)])
        changed.loc[held_out, ["Time", "Amount"]] += 1_000_000
        with patch("src.utils.pd.read_csv", return_value=changed):
            changed_train, *_ = load_client_data("unused", 5, client_id, seed=72)
        np.testing.assert_array_equal(changed_train, train)

    def test_invalid_client_counts_and_ids(self):
        for num_clients in (-1, 0, 1, 2.5, True):
            with self.subTest(num_clients=num_clients), self.assertRaisesRegex(ValueError, "num_clients"):
                load_client_data("unused", num_clients, 0)
        for client_id in (-1, 5, 0.5, False):
            with self.subTest(client_id=client_id), self.assertRaisesRegex(ValueError, "client_id.*0, 4"):
                load_client_data("unused", 5, client_id)

    def test_invalid_seed_and_class_labels(self):
        frame = make_frame()
        with self.assertRaisesRegex(ValueError, "seed"):
            build_partition_plan(frame, 2, seed=-1)
        with self.assertRaisesRegex(ValueError, "both binary labels"):
            build_partition_plan(frame.assign(Class=0), 2)

    def test_minority_capacity_uses_unique_rows(self):
        base = make_frame()
        small = base.iloc[:2220]
        frame = pd.concat([small, small.iloc[2200:]] * 10)
        with self.assertRaisesRegex(ValueError, r"Class 1 has 13 unique rows.*10 clients need at least 60"):
            build_partition_plan(frame, 10)

    def test_conflicting_labels_for_identical_features_are_rejected(self):
        base = make_frame()
        frame = pd.concat([base, base.iloc[[0]].assign(Class=1)])
        with self.assertRaisesRegex(ValueError, "conflicting Class labels"):
            build_partition_plan(frame, 2)

    @unittest.skipUnless(Path("data/creditcard.csv").is_file(), "Optional reference CSV is unavailable")
    def test_reference_csv(self):
        frame = pd.read_csv("data/creditcard.csv")
        labels = frame["Class"].to_numpy()
        reference = build_partition_plan(frame, 2)
        for num_clients in (2, 5, 10):
            with self.subTest(num_clients=num_clients):
                plan = reference if num_clients == 2 else build_partition_plan(frame, num_clients)
                self.check_plan(frame, plan, num_clients)
                self.check_same_global_membership(reference, plan)
                train = np.concatenate([client.train for client in plan.clients])
                validation = np.concatenate([client.validation for client in plan.clients])
                print(
                    f"\n{num_clients} clients, class counts [0, 1]"
                    f" | train {counts(labels[train])} | validation {counts(labels[validation])}"
                    f" | final test {counts(labels[plan.final_test])}"
                    f" | excluded duplicates {counts(labels[plan.excluded_duplicates])}"
                    " | row and duplicate-content overlaps: 0"
                    " | all global row-ID sets match the 2-client plan"
                )
                for client_id, client in enumerate(plan.clients):
                    print(f"  Client {client_id}: train {counts(labels[client.train])}, "
                          f"validation {counts(labels[client.validation])}")
