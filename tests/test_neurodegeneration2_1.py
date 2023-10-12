import unittest
import os
from neurodegeneration2.neurodegeneration2_1 import Neurodegeneration2
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor


class TestNeurodegeneration2(unittest.TestCase):
    _test_obj = []

    @property
    def home(self):
        return os.path.join(os.getenv('ADNI_HOME'), 'NMF_FDG')

    def setUp(self):
        self._test_obj = Neurodegeneration2("unit-testing", home=self.home)
        os.chdir(self.home)

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_ctor(self):
        pprint(self._test_obj)

    def test_build_distance_matrix(self):
        dict = self._test_obj.build_distance_matrix()
        pprint(dict)  # 4 hours for mask.txt

    def test_ppe(self):
        with ProcessPoolExecutor(max_workers=2) as executor:
            for idx, product in zip(range(1, 25), executor.map(str,range(1, 25))):
                print('index %d had value %s' % (idx, product))

    def test_build_surrogate_maps(self):
        filenames = self._test_obj.distmap_files()
        self.assertTrue(os.path.isfile(filenames['D']))
        self.assertTrue(os.path.isfile(filenames['index']))

        with ProcessPoolExecutor(max_workers=3) as executor:
            for idx, product in zip(range(1, 25), executor.map(self._test_obj.build_surrogate_maps,
                                                               range(1, 25))):
                print('surrogate %d had size %d' % (idx, product.size))

        # for b in range(1, 25):  # 1 .. 24, inclusive
        #     # self._test_obj.inspect_variogram_fit(b) # an hour or so
        #     surrogates = self._test_obj.build_surrogate_maps(b, filename='surrogates_1k_patt' + str(b) + '.pkl',
        #                                                      n=1000)  # 5.9 hours for 1k, 25 hours for 10k
        #     pprint(surrogates.size)

    def test_build_stats104(self):
        labels = ['action', 'adaptation', 'addiction', 'anticipation', 'anxiety', 'arousal', 'association',
                  'autobiographical_memory', 'awareness', 'balance', 'belief', 'categorization', 'choice',
                  'cognitive_control', 'communication', 'competition', 'concept', 'consciousness', 'consolidation',
                  'context', 'coordination', 'decision_making', 'discrimination', 'distraction', 'eating', 'efficiency',
                  'effort', 'emotion_regulation', 'empathy', 'encoding', 'episodic_memory', 'executive_control',
                  'executive_function', 'expectancy', 'expertise', 'face_recognition', 'facial_expression',
                  'familiarity', 'fear', 'gaze', 'goal', 'hyperactivity', 'impulsivity', 'induction', 'inference',
                  'integration', 'intelligence', 'intention', 'interference', 'knowledge', 'language_comprehension',
                  'learning', 'listening', 'loss', 'maintenance', 'manipulation', 'memory_retrieval', 'mental_imagery',
                  'monitoring', 'mood', 'morphology', 'motor_control', 'movement', 'multisensory', 'naming',
                  'navigation', 'object_recognition', 'pain', 'planning', 'priming', 'psychosis', 'reading',
                  'reasoning', 'recall', 'rehearsal', 'remembering', 'response_inhibition', 'response_selection',
                  'retention', 'reward', 'rhythm', 'risk', 'rule', 'salience', 'selective_attention', 'semantic_memory',
                  'sentence_comprehension', 'sleep', 'social_cognition', 'spatial_attention', 'speech_perception',
                  'speech_production', 'strategy', 'stress', 'sustained_attention', 'thought', 'uncertainty',
                  'updating', 'valence', 'verbal_fluency', 'visual_attention', 'visual_perception', 'word_recognition',
                  'working_memory']
        for label in labels:
            for basis_idx in range(1, 25):  # 1 .. 24, inclusive
                basis_map = self._test_obj.basis_map(basis_idx)
                surrogates = self._test_obj.pickle_load('surrogates_1k_patt' + str(basis_idx) + '.pkl')
                new_map = self._test_obj.neurosynth_map(label)
                self._test_obj.build_stats(basis_map=basis_map, basis_num=basis_idx, surrogates=surrogates,
                                           new_map=new_map, new_label=label)
                # 40 sec if check_fit=false

    def test_build_stats27(self):
        labels = ['Langue_comprehension', 'Social', 'Memory', 'Language_semantics', 'Negative_emotion',
                  'Visual_attention', 'Language_perception', 'Numerical', 'Working_Memory', 'Emotional_cues',
                  'Reward', 'Response_preparation', 'Hearing', 'Facial_recognition', 'Addiction',
                  'Objects', 'Sustenance_state', 'Error_learning', 'Response_inhibition', 'Praxis',
                  'Stimulus_response', 'Motion_perception', 'Perception', 'Pain', 'Directed_gaze',
                  'Somatosensory', 'Motor']
        for label in labels:
            for basis_idx in range(1, 25):  # 1 .. 24, inclusive
                basis_map = self._test_obj.basis_map(basis_idx)
                surrogates = self._test_obj.pickle_load('surrogates_1k_patt' + str(basis_idx) + '.pkl')
                new_map = self._test_obj.neurosynth_topic50_map(label)
                self._test_obj.build_stats(basis_map=basis_map, basis_num=basis_idx, surrogates=surrogates,
                                           new_map=new_map, new_label=label)
                # 40 sec if check_fit=false

    def test_build_eigenbrains(self):
        labels = ['EB1', 'EB2', 'EB3', 'EB4', 'EB5', 'EB6', 'EB7', 'EB8', 'EB9', 'EB10']
        for label in labels:
            for basis_idx in range(1, 25):  # 1 .. 24, inclusive
                basis_map = self._test_obj.basis_map(basis_idx)
                surrogates = self._test_obj.pickle_load('surrogates_1k_patt' + str(basis_idx) + '.pkl')
                new_map = self._test_obj.eigenbrain_map(label)
                self._test_obj.build_stats(basis_map=basis_map, basis_num=basis_idx, surrogates=surrogates,
                                           new_map=new_map, new_label=label)
                # 40 sec if check_fit=false


if __name__ == '__main__':
    unittest.main()
