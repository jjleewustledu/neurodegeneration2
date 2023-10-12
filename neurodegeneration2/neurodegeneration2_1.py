# Copyright 2023 John J. Lee. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""High-level classes for manuscript in progress by J.J. Lee, et al. 'Patterns of Neurodegeneration'.
Prepare brain map txt files using https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dmaskdump.html.
E.g.:  $ 3dmaskdump -mask mask.nii.gz -o map.txt -noijk map.nii.gz
See also https://brainsmash.readthedocs.io/en/latest/example.html#whole-brain-volume."""

import os
import pickle
from brainsmash.workbench.geo import volume
from brainsmash.mapgen.eval import sampled_fit
from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from brainsmash.mapgen.stats import nonparp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Neurodegeneration2(object):
    """Generic base class for neurodegeneration2 project."""
    description = None
    home = None

    @property
    def memorymaps_dir(self):
        d = os.path.join(self.home, 'memorymaps')
        if not os.path.isdir(d):
            os.mkdir(d)
        return d

    @property
    def eigenbrains_dir(self):
        d = os.path.join(os.getenv('ADNI_HOME'), 'Jones_2022', 'Eigenbrains')
        # os.path.join(self.home, '..', 'neurosynth.org')
        assert (os.path.exists(d))
        return d

    @property
    def neurosynth_dir(self):
        d = os.path.join(os.getenv('ADNI_HOME'), 'neurosynth.org')
        # os.path.join(self.home, '..', 'neurosynth.org')
        assert (os.path.exists(d))
        return d

    @property
    def neurosynth_topic50_dir(self):
        d = os.path.join(os.getenv('ADNI_HOME'), 'neurosynth.org', 'v4-topics-50')
        # os.path.join(self.home, '..', 'neurosynth.org', 'v4-topics-50')
        assert (os.path.exists(d))
        return d

    @property
    def niiImg(self):
        d = os.path.join(self.home, 'baseline_cn', 'NumBases22', 'OPNMF', 'niiImg')
        assert (os.path.exists(d))
        return d

    @property
    def volBin(self):
        d = os.path.join(os.getenv('ADNI_HOME'), 'VolBin')
        assert (os.path.exists(d))
        return d

    @staticmethod
    def pickle_dump(filename='neurodegeneration2.pkl', obj=None):
        if not '.pkl' in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def pickle_load(filename='neurodegeneration2.pkl'):
        if not '.pkl' in filename:
            filename = filename + '.pkl'
        try:
            if os.path.isfile(filename):
                with open(filename, 'rb') as inp:
                    return pickle.load(inp)
        except (TypeError, IOError) as e:
            pass

    @staticmethod
    def variogram_params():
        # These are three of the key parameters affecting the variogram fit,
        # tunable as noted in https://brainsmash.readthedocs.io/en/latest/example.html#whole-brain-volume
        kwargs = {'ns': 2000,
                  'pv': 70,
                  'nh': 25,
                  'knn': 4000,
                  'verbose': True,
                  'n_jobs': 12}  # use knn ~ 2*ns
        return kwargs

    def basis_map(self, basis=None):
        return np.loadtxt(os.path.join(self.niiImg, 'Basis_' + str(basis) + '.txt'))

    def build_distance_matrix(self):
        """This text to memory-mapped array conversion only ever needs to be run once for a given distance matrix."""
        coord_file = os.path.join(self.volBin, 'mask.txt')
        return volume(coord_file, self.memorymaps_dir)

    def build_stats(self, basis_map=None, basis_num=None, surrogates=None, new_map=None, new_label='new',
                    check_fit=False):
        """https://brainsmash.readthedocs.io/en/latest/example.html"""

        # compute the Pearson correlation between each surrogate map and the new map
        surrogate_brainmap_corrs = pearsonr(new_map, surrogates).flatten()
        # surrogate_pairwise_corrs = pairwise_r(surrogates, flatten=True)

        # this is the empirical statistic we're creating a null distribution for
        test_stat = stats.pearsonr(basis_map, new_map)[0]
        p = nonparp(test_stat, surrogate_brainmap_corrs)

        if p <= 0.05:
            # repeat using randomly shuffled basis map
            naive_surrogates = np.array([np.random.permutation(basis_map) for _ in range(1000)])
            naive_brainmap_corrs = pearsonr(new_map, naive_surrogates).flatten()
            # naive_pairwise_corrs = pairwise_r(naive_surrogates, flatten=True)

            # plot results
            sac = '#377eb8'  # autocorr-preserving
            rc = '#e41a1c'  # randomly shuffled
            bins = np.linspace(-1, 1, 201)  # correlation b

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_axes([0.2, 0.25, 0.6, 0.6])  # autocorr preserving
            ax2 = ax.twinx()  # randomly shuffled

            # plot the data
            ax.axvline(test_stat, 0, 0.8, color='k', linestyle='dashed', lw=1)
            ax.hist(surrogate_brainmap_corrs, bins=bins, color=sac, alpha=1,
                    density=True, clip_on=False, zorder=1)
            ax2.hist(naive_brainmap_corrs, bins=bins, color=rc, alpha=0.7,
                     density=True, clip_on=False, zorder=2)

            # make the plot nice...
            # ax.set_xticks(np.arange(-1, 1.1, 0.5))
            ax.spines['left'].set_color(sac)
            ax.tick_params(axis='y', colors=sac)
            ax2.spines['right'].set_color(rc)
            ax2.tick_params(axis='y', colors=rc)
            # ax.set_ylim(0, 2)
            # ax2.set_ylim(0, 6)
            ax.set_xlim(-0.25, 0.25)
            [s.set_visible(False) for s in [
                ax.spines['top'], ax.spines['right'], ax2.spines['top'], ax2.spines['left']]]
            ax.text(0.97, 1.1, 'SA-independent', ha='right', va='bottom',
                    color=rc, transform=ax.transAxes)
            ax.text(0.97, 1.03, 'SA-preserving', ha='right', va='bottom',
                    color=sac, transform=ax.transAxes)
            ax.text(test_stat, 1.65, new_label + "\nmap", ha='center', va='bottom')
            ax.text(0.5, -0.2, "Pearson r = " + str(test_stat) + "\n(p = " + str(p) + ") for ADNI P" + str(basis_num),
                    ha='center', va='top', transform=ax.transAxes)
            ax.text(-0.3, 0.5, "Density", rotation=90, ha='left', va='center', transform=ax.transAxes)
            # plt.show()
            fig.savefig(new_label + '_' + str(basis_num) + '.pdf', dpi=1200)
            # plt.figure().close()

            if check_fit:
                filenames = self.distmap_files()
                kwargs = self.variogram_params()
                sampled_fit(basis_map, filenames['D'], filenames['index'], nsurr=100, **kwargs)
                fig.savefig(new_label + '_' + str(basis_num) + '_check_fit.pdf', dpi=1200)
                # plt.figure().close()

        # compute non-parametric p-values using our two different null distributions
        # print("Spatially naive p-value:", nonparp(test_stat, naive_brainmap_corrs))
        print("Pearson r: ", str(test_stat), "(p = ", str(p), ") for ", new_label, " basis ", basis_num)
        return p

    def build_surrogate_maps(self, basis=None, n=1000):
        """ is best run after running inspect_variogram_fit()
        Args:
            basis: index from NMF
        Returns:
            surrogate brain maps with spatial autocorrelation matched to target, self.basis_map()
        See also:  https://brainsmash.readthedocs.io/en/latest/example.html#whole-brain-volume and
        https://brainsmash.readthedocs.io/en/latest/gettingstarted.html#keyword-arguments-to-brainsmash-mapgen-sampled-sampled
        """
        filenames = self.distmap_files()
        brain_map = self.basis_map(basis)
        kwargs = self.variogram_params()
        gen = Sampled(x=brain_map, D=filenames['D'], index=filenames['index'], **kwargs)
        surrogates = gen(n=n)

        filename = 'surrogates_1k_patt' + str(basis) + '.pkl'
        self.pickle_dump(filename, surrogates)  # ~1.8 GB for 1k; ~18 GB for 10k samples
        return surrogates

    def distmap_files(self):
        """memory-mapped binary files, per numpy"""
        filenames = {'D': os.path.join(self.memorymaps_dir, 'distmat.npy'),
                     'index': os.path.join(self.memorymaps_dir, 'index.npy')}
        return filenames

    def eigenbrain_map(self, fileprefix=None):
        return np.loadtxt(os.path.join(self.eigenbrains_dir, fileprefix + '.txt'))

    def inspect_variogram_fit(self, basis=None):
        filenames = self.distmap_files()
        brain_map = self.basis_map(basis)
        kwargs = self.variogram_params()

        # Running this command will generate a matplotlib figure
        sampled_fit(brain_map, filenames['D'], filenames['index'], nsurr=20, **kwargs)

    def neurosynth_map(self, fileprefix=None):
        return np.loadtxt(os.path.join(self.neurosynth_dir, fileprefix + '_association-test_z_FDR_0.01.txt'))

    def neurosynth_topic50_map(self, fileprefix=None):
        return np.loadtxt(os.path.join(self.neurosynth_topic50_dir, fileprefix + '.txt'))

    def __init__(self, description=None, home=os.getcwd()):
        """Instantiate this class with a textual description.

        Args:
            description:  An optional string succinctly describing the intent of the class instance.
            home:  f.q. path to Singularity/ADNI/NMF_FDG
        """
        self.description = description
        self.home = home
