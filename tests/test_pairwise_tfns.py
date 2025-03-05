from unittest import TestCase
import time
from bnetdiff.utils import fisher_r_to_z
from bnetdiff.pairwise_tfns import (_permutation_task_ind,
                                    _permutation_task_paired,
                                    compute_null_dist,
                                    compute_t_stat_diff,
                                    compute_permute_t_stat_ind,
                                    compute_permute_t_stat_diff,
                                    compute_p_val,
                                    compute_t_stat,
                                    compute_t_stat_tfnos,
                                    compute_t_stat_tfnos_diffs)

from bnetdiff.datasets import generate_fc_matrices
from bnetdiff.datasets import (create_simple_random,
                               create_nd_random_arr)
import numpy as np


class TestBasicStats(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        N = 30  # Number of ROIs
        effect_size = 0.2  # Magnitude of group differences

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size, n_samples_group1=30,
                                                            n_samples_group2=20)
        cls.fc_sim = {"group1": fisher_r_to_z(group1.copy()),
                      "group2": fisher_r_to_z(group2.copy()),
                      "true_diff": cov2 - cov1,
                      'cov2': cov2,
                      'cov1': cov1}

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size, n_samples_group1=40,
                                                            n_samples_group2=40)
        cls.fc_sim_paired = {"group1": fisher_r_to_z(group1.copy()),
                             "group2": fisher_r_to_z(group2.copy()),
                             "true_diff": cov2 - cov1,
                             'cov2': cov2,
                             'cov1': cov1}

    def run_and_measure(self, func, arr1, arr2, n_permutations, random_state, use_mp):
        """Helper function to measure execution time of a function."""
        start_time = time.time()
        compute_null_dist(arr1, arr2, func, n_permutations=n_permutations, random_state=random_state, use_mp=use_mp)
        return time.time() - start_time

    def test_compute_t_stat(self):
        group_dict = self.fc_sim

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], paired=False)

        self.assertLess(2, emp_t_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertEqual(0, emp_t_dict["g1>g2"][np.triu_indices(10, k=1)].mean())

    def test_compute_t_stat_diff(self):
        group_dict = self.fc_sim_paired
        t_stat_dict = compute_t_stat_diff(group_dict['group2'] - group_dict['group1'])
        self.assertLess(2, t_stat_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertEqual(0, t_stat_dict["g1>g2"][np.triu_indices(10, k=1)].mean())

    def test_compute_permut_t_stat_ind(self):
        group_dict = self.fc_sim

        perm_t_pos, perm_t_neg = compute_permute_t_stat_ind(group_dict['group1'], group_dict['group2'])

        self.assertGreater(perm_t_pos, 1)
        self.assertGreater(perm_t_neg, 1)
        self.assertLess(perm_t_pos, 5)

    def test_compute_t_stat_tfnos(self):
        group_dict = self.fc_sim

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], paired=False)
        emp_tfnos_dict = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'], paired=False)

        self.assertLess(10, emp_tfnos_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertLess(1, emp_t_dict["g2>g1"][np.triu_indices(10, k=1)].mean())

    def test_compute_t_stat_tfnos_paired(self):
        group_dict = self.fc_sim_paired

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], paired=True)
        emp_tfnos_dict = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'], paired=True)
        emp_tfnos_sp_dict = compute_t_stat_tfnos_diffs(group_dict['group2'] - group_dict['group1'])

        self.assertIsNone(np.testing.assert_almost_equal(emp_tfnos_dict["g2>g1"],
                                                         emp_tfnos_sp_dict["g2>g1"]))
        self.assertGreater(emp_tfnos_dict["g2>g1"].sum(), emp_t_dict["g2>g1"].sum())

    def test_compute_t_stat_tfnos_list_pars(self):
        group_dict = self.fc_sim
        t_stat_mod = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'], e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(t_stat_mod["g2>g1"].shape[-1], 2)
        self.assertEqual(t_stat_mod["g1>g2"].shape[-1], 2)

    def test__permutation_task_ind_t(self):
        group_dict = self.fc_sim
        t_stat = compute_t_stat(group_dict['group1'], group_dict['group2'], paired=False)
        full_group = np.concatenate((group_dict['group1'], group_dict['group2']), axis=0)
        t_maxes = _permutation_task_ind(full_group, compute_t_stat,
                                        30, 42)
        self.assertIsInstance(t_maxes, dict)
        self.assertEqual(len(t_maxes.values()), 2)
        self.assertGreater(np.max(t_stat["g2>g1"]), t_maxes["g2>g1"])

    def test__permutation_task_ind(self):
        group_dict = self.fc_sim
        t_stat_mod = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'], e=[0.4, 0.6], h=[1, 2])
        full_group = np.concatenate((group_dict['group1'], group_dict['group2']), axis=0)
        t_maxes = _permutation_task_ind(full_group, compute_t_stat_tfnos,
                                        30, 42, e=[0.4, 0.6], h=[1, 2])
        self.assertIsInstance(t_maxes, dict)
        self.assertIsInstance(t_stat_mod, dict)
        self.assertGreater(t_stat_mod['g2>g1'].mean(), t_maxes['g2>g1'].mean())
        self.assertGreater(t_stat_mod['g1>g2'].max(), t_maxes['g1>g2'].mean())

    def test__permutation_task_paired(self):
        group_dict = self.fc_sim_paired
        emp_t = compute_t_stat_diff(group_dict['group2'] - group_dict['group1'])
        emp_tfnos = compute_t_stat_tfnos_diffs(group_dict['group2'] - group_dict['group1'], e=[0.4, 0.6], h=[1, 2])
        t_max_t = _permutation_task_paired(group_dict['group2'] - group_dict['group1'], compute_t_stat_diff, 30)
        t_maxes = _permutation_task_paired(group_dict['group2'] - group_dict['group1'], compute_t_stat_tfnos_diffs,
                                           e=[0.4, 0.6], h=[1, 2])
        self.assertIsInstance(t_maxes, dict)
        self.assertIsInstance(t_max_t, dict)
        self.assertGreater(emp_tfnos['g2>g1'].mean(), t_maxes['g2>g1'].mean())
        self.assertGreater(emp_t['g1>g2'].max(), t_max_t['g1>g2'])
        self.assertGreater(emp_tfnos['g1>g2'].max(), t_maxes['g1>g2'].mean())

    def test_compute_null_t_stat_ind(self):
        group_dict = self.fc_sim

        n_permutations = 100

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat, paired=False,
                                   n_permutations=n_permutations, random_state=42, use_mp=False)
        null_t_mc = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat, paired=False,
                                      n_permutations=n_permutations, random_state=42, use_mp=True)

        self.assertIsInstance(null_t, dict)
        self.assertIsInstance(null_t_mc, dict)
        self.assertEqual((null_t["g2>g1"].mean() - null_t_mc["g1>g2"].mean()).round(), 0)

    def test_compute_null_t_stat_tfnos_ind(self):
        group_dict = self.fc_sim

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'], paired=False)

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnos, paired=False,
                                   n_permutations=n_permutations, random_state=42, use_mp=False)
        null_t_mc = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnos, paired=False,
                                      n_permutations=n_permutations, random_state=42, use_mp=True)

        self.assertEqual((null_t["g2>g1"].mean() - null_t_mc["g1>g2"].mean()).round(), 0)
        self.assertGreater(emp_tfnos["g2>g1"].mean(), null_t_mc["g1>g2"].mean())

    def test_compute_null_t_stat_tfnos_ind_multi(self):
        group_dict = self.fc_sim

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'],
                                         e=[0.4, 0.6], h=[1, 2], paired=False)

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnos, paired=False,
                                   n_permutations=n_permutations, use_mp=False, e=[0.4, 0.6], h=[1, 2])
        null_t_mp = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnos, paired=False,
                                      n_permutations=n_permutations, use_mp=True, e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(null_t["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])
        self.assertEqual(null_t_mp["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])

    def test_compute_null_t_stat_tfnos_paired_multi(self):
        group_dict = self.fc_sim_paired

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnos(group_dict['group1'], group_dict['group2'],
                                         e=[0.4, 0.6], h=[1, 2], paired=True)

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnos_diffs, paired=True,
                                   n_permutations=n_permutations, use_mp=False, e=[0.4, 0.6], h=[1, 2])
        null_t_mp = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnos_diffs, paired=True,
                                      n_permutations=n_permutations, use_mp=True, e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(null_t["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])
        self.assertEqual(null_t_mp["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])

    def test_compute_null_t_stat_ind_eff(self):
        group_dict = self.fc_sim

        n_permutations = 1000
        random_state = 42

        time_mp = self.run_and_measure(compute_t_stat_tfnos,
                                       group_dict['group1'], group_dict['group2'],
                                       n_permutations, random_state, True)

        time_cycle = self.run_and_measure(compute_t_stat_tfnos,
                                          group_dict['group1'], group_dict['group2'],
                                          n_permutations, random_state, False)

        self.assertLess(time_mp, time_cycle)

    def test_compute_p_val_indep(self):
        group_dict = self.fc_sim
        n_permutations = 1000
        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, paired=False, tf=False, use_mp=True)

        self.assertLess(p_vals["g2>g1"][np.triu_indices(10, k=1)].mean(), 0.3)
        self.assertGreater(p_vals["g1>g2"][np.triu_indices(10, k=1)].mean(), 0.3)

    def test_compute_p_val_indep_tf(self):
        group_dict = self.fc_sim
        n_permutations = 1000

        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, paired=False, tf=True, use_mp=True)

        self.assertLess(p_vals["g2>g1"][np.triu_indices(10, k=1)].mean(), 0.3)
        self.assertGreater(p_vals["g1>g2"][np.triu_indices(10, k=1)].mean(), 0.3)

    def test_compute_p_val_indep_tf_multi(self):
        group_dict = self.fc_sim
        n_permutations = 1000

        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, paired=False, tf=True, use_mp=True, e=[0.4, 0.6],
                               h=[1, 2])

        self.assertLess(p_vals["g2>g1"][..., 0][np.triu_indices(10, k=1)].mean(), 0.05)
        self.assertLess(p_vals["g2>g1"][..., 1][np.triu_indices(10, k=1)].mean(), 0.05)

    def test_compute_p_val_indep_tf_orig(self):
        group_dict = self.fc_sim
        n_permutations = 1000
        p_vals_orig = compute_p_val(group_dict['group1'], group_dict['group2'],
                                    n_permutations=n_permutations, paired=False, tf=False, use_mp=True)
        p_vals_tf = compute_p_val(group_dict['group1'], group_dict['group2'],
                                  n_permutations=n_permutations, paired=False, tf=True, use_mp=True)

        self.assertLess(p_vals_tf["g2>g1"][np.triu_indices(10, k=1)].mean(),
                        p_vals_orig["g2>g1"][np.triu_indices(10, k=1)].mean())

    def test_compute_p_val_paired_tf_orig(self):
        group_dict = self.fc_sim_paired
        n_permutations = 1000
        p_vals_orig = compute_p_val(group_dict['group1'], group_dict['group2'],
                                    n_permutations=n_permutations, paired=True, tf=False, use_mp=True)
        p_vals_tf = compute_p_val(group_dict['group1'], group_dict['group2'],
                                  n_permutations=n_permutations, paired=True, tf=True, use_mp=True)

        self.assertLess(p_vals_tf["g2>g1"][np.triu_indices(10, k=1)].mean(),
                        p_vals_orig["g2>g1"][np.triu_indices(10, k=1)].mean())
