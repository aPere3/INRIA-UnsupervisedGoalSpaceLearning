#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This scipt execute the debug campaign.

Author: Alexandre Péré
"""

import sys
sys.path.append("../../src")
import experiment
import environments
import rendering
import embeddings
import utils
import exploration
import measures
import numpy as np
import scipy.stats
import json
import os
import itertools
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy


class Experiment(experiment.BaseExperiment):
    """
    This class contain the logic for an instance of an experiment.
    """

    def _run(self):
        """
        The Experiment logic.
        """

        # We set the armball configuration
        self._log_lgr.info("Setting the Environment Configuration")
        environment_config = self._params['env_config']

        # We instantiate the environment
        self._log_lgr.info("Instantiating the Environment")
        environment = self._params['env'](environment_config, n_dmp_basis=3, goal_size=1.)

        # We instantiate the renderer
        self._log_lgr.info("Instantiating the Renderer")
        renderer = self._params['renderer'](noise=self._params['noise'], distractor=self._params['distractor'],
                                            deformation=self._params['deformation'], outliers=self._params['outliers'])

        # We train the embedding
        if self._params['embedding'] not in ['none', 'state']:
            # We sample the environment state space
            self._log_lgr.info("Sampling the state space")
            X = renderer.sample(nb_samples=self._params['nb_samples'],type=self._params['sampling'])
            # We instantiate the embedding
            self._log_lgr.info("Instantiating the Embedding Model")
            embedder = self._params['embedding'](emb_size=self._params['emb_size'],
                                                 logs_path=self._log_path, name=self._name)
            # We train the model
            self._log_lgr.info("Training Embedding")
            embedder.fit(X)
            loss, log_likelihood = embedder.get_training_data()
            self._res_dict['loss'] = loss[-1] if loss is not None else None
            self._res_dict['log_likelihood'] = log_likelihood[-1] if log_likelihood is not None else None
        else:
            self._params['emb_size'] = renderer.nb_params()

        # We measure quality
        if self._params['embedding'] is not 'none':
            self._log_lgr.info("Generating Random states")
            samples_states = np.random.uniform(size=[NB_POINTS_MANIFOLD_MEASURES, renderer.nb_params()])
            samples_geodesics = utils.manifold_2_space_coordinates(samples_states, renderer.get_factor_dims())
            if self._params['embedding'] is 'state':
                samples_latents = samples_states
                training_latents = np.random.uniform(size=[NB_POINTS_MANIFOLD_MEASURES, renderer.nb_params()])
            else:
                samples_images = np.array([renderer.draw(state) for state in samples_states])
                samples_latents = embedder.transform(samples_images)
                training_latents = embedder.transform(X)
            if self._params['belief'] == 'hull':
                expected_latents = np.random.uniform(high=training_latents.max(axis=0),
                                                    low=training_latents.min(axis=0),
                                                    size=[NB_POINTS_MANIFOLD_MEASURES, self._params['emb_size']])
            elif self._params['belief'] == 'uniform':
                expected_latents = np.random.uniform(size=[NB_POINTS_MANIFOLD_MEASURES, self._params['emb_size']])
            elif self._params['belief'] == 'normal':
                expected_latents = np.random.randn(NB_POINTS_MANIFOLD_MEASURES, self._params['emb_size'])
            elif self._params['belief'] == 'kde':
                kde = scipy.stats.gaussian_kde(training_latents)
                expected_latents = kde.resample(size=NB_POINTS_MANIFOLD_MEASURES)
            expected_latents = expected_latents.astype(np.float32)
            self._log_lgr.debug(expected_latents.dtype)
            self._log_lgr.info("Measuring divergence between belief and actual distribution")
            self._res_dict.update(measures.distribution_divergence(samples_latents, expected_latents))
            self._log_lgr.info("Measuring local quality")
            self._res_dict.update(measures.embedding_local_quality(samples_geodesics, samples_latents))
            self._log_lgr.info("Measuring global quality")
            self._res_dict.update(measures.embedding_global_quality(samples_geodesics, samples_latents))

        # We define the transformation method
        def transform_method(s):
            if self._params['embedding'] not in ['state', 'none']:
                s = (s+1.)/2.
                image = renderer.draw(s)
                latent = embedder.transform(image.reshape([1,-1])).squeeze()
            else:
                latent = (s+1.)/2.

            return latent

        # We define the sampling method
        def sampling_method():
            if self._params['belief'] == 'hull':
                sample = np.random.uniform(high=training_latents.max(axis=0),
                                                 low=training_latents.min(axis=0),
                                                 size=[1, self._params['emb_size']])
            elif self._params['belief'] == 'uniform':
                sample = np.random.uniform(size=[1, self._params['emb_size']])
            elif self._params['belief'] == 'normal':
                sample = np.random.randn(1, self._params['emb_size'])
            elif self._params['belief'] == 'kde':
                sample = kde.resample(size=1).squeeze()
            return sample.ravel()

        # We instantiate the exploration algorithm
        self._log_lgr.info("Instantiating the explorator")
        if not self._params['embedding'] is 'none':
            explorator = exploration.GoalBabblingExploration(environment=environment,
                                                             explo_ratio=self._params['explo_ratio'],
                                                             emb_size=self._params['emb_size'],
                                                             transform_method=transform_method,
                                                             sampling_method=sampling_method,
                                                             callback_period=EXPLORATION_CALLBACK_PERIOD)
        else:
            explorator = exploration.RandomMotorBabblingExploration(environment=environment,
                                                                    explo_ratio = self._params['explo_ratio'],
                                                                    emb_size=self._params['emb_size'],
                                                                    transform_method=transform_method,
                                                                    sampling_method=sampling_method,
                                                                    callback_period=EXPLORATION_CALLBACK_PERIOD)

        # We set the measures dict values
        self._res_dict['exploration'] = []
        self._res_dict['mse'] = []

        # We define the callback method
        def callback_method():
            self._log_lgr.info("Executing Callback")
            # We append measures
            self._res_dict['mse'].append(measures.model_mse(explorator, nb_samples=NB_SAMPLES_MSE))
            self._res_dict['exploration'].append(measures.exploration(explorator, bins=NB_BINS_EXPLORATION))

        # We set the callback method
        explorator.set_callback(callback_method=callback_method)

        # We perform the exploration
        explorator.explore(nb_iterations=NB_ITERATIONS_EXPLORATION)

        # We retrieve the reach ratio
        reached_states = np.array(explorator.get_reached_states())
        initial_state = environment.initial_pose
        self._res_dict['reach_ratio'] = 1. - float(np.logical_and.reduce(reached_states==initial_state, axis=1).sum())/NB_ITERATIONS_EXPLORATION
        self._log_lgr.info("Reached Ratio: %s"%self._res_dict['reach_ratio'])

        # We announce the end and the results
        self._log_lgr.info("End of experiment reached")


class Campaign(experiment.BaseExperiment):
    """
    This class contains the code for the whole campaign of experiments. It will run multiple instances of the other experiments.
    """

    def _run(self):
        """
        This campaign logic.
        """

        # We set the variables
        self._log_lgr.info("Generate experimental factors list")
        embedding_list = ["state", "none"]
        environment_list = ["armball", "armtwoballs", "armarrow", "armtoolball"]

        # We define the original experiment
        self._log_lgr.info("Generate original experiment parameters")
        original_params = {'embedding_size':10,
                           'sampling':'full',
                           'sampling_size':10000,
                           'distraction':False,
                           'deformation':0.,
                           'outliers':0.,
                           'noise':0.}

        # We instantiate the dataframe that will contain the experiments data
        if os.path.isfile(os.path.join(self._log_path, "campaign_data.csv")):
            campaign_data = pd.read_csv(os.path.join(self._log_path, "campaign_data.csv"))
            if 'Unnamed: 0' in campaign_data.columns.values:
                campaign_data.drop("Unnamed: 0", axis=1, inplace=True)
        else:
            campaign_data = pd.DataFrame(columns=['name'])

        # Group 0: Tests
        if self._params['test']:
            self._log_lgr.info("Launching Group 0: Test")
            group_id = 0
            # We generate tuples of parameters
            params_itr = embedding_list
            # We loop through the parameters
            for exp_id, emb in enumerate(params_itr):
                if raw_input("To test model %s write [t]"%emb) !='t':
                    continue
                params = copy.deepcopy(original_params)
                params.update({'embedding':emb, 'sampling_size':1000})
                params['name']="Group%i-Exp%i-Run0" % (group_id, exp_id)
                with self._init_expe(**params) as expe:
                    self._log_lgr.info("Launching Experiment %s"%expe.get_uuid())
                    # We perform the experiment
                    expe.run()

        else:
            # Group 1: Different embeddings on different environments
            self._log_lgr.info("Launching Group 1")
            group_id = 1
            # We generate tuples of parameters
            params_itr = itertools.product(embedding_list, environment_list)
            # We loop through the parameters
            for exp_id, (emb, env) in enumerate(params_itr):
                params = copy.deepcopy(original_params)
                params.update({'embedding':emb, 'environment':env})
                # We perform the experiment for multiple times
                for run_id in range(NB_RUNS_PER_EXPE):
                    params['name']="Group%i-Exp%i-Run%i" % (group_id, exp_id, run_id)
                    run_dump = copy.deepcopy(params)
                    # We check if the experiment was already made and skip if necessary
                    if params['name'] in campaign_data.loc[:,'name'].values:
                        self._log_lgr.info("Experiment %s already performed, Skipping...." % params['name'])
                        continue
                    with self._init_expe(**params) as expe:
                        self._log_lgr.info("Launching Experiment %s"%expe.get_uuid())
                        # We perform the experiment
                        run_dump.update(expe.run())
                        # We save the experimental data
                        run_dump['uuid'] = expe.get_uuid()
                        campaign_data = campaign_data.append(run_dump, ignore_index=True)
                        campaign_data.to_csv(os.path.join(self._log_path, 'campaign_data.csv'), index=False)

       # End of the experiment
        self._log_lgr.info("End of the Campaign. Thanks!")

    def _init_expe(self, name, embedding, embedding_size, environment, sampling, sampling_size, noise, distraction,
                   deformation, outliers, **kwargs):
        """
        This method generates the parameters of the experiment according to the given inputs, and return the object.
        """

        # We instantiate the parameters dict
        params = dict()
        # Depending on the environment, we set some parameters
        if environment=="armball":
            params['env_config'] = dict(m_mins=[-1.] * 7, m_maxs=[1.] * 7, s_mins=[-1.] * embedding_size, s_maxs=[1.] * embedding_size,
                                    arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05], arm_angle_shift=0.5, arm_rest_state=[0.] * 7,
                                    ball_size=0.1, ball_initial_position=[0.6, 0.6])
            params['env'] = environments.ArmBallDynamic
            params['renderer'] = rendering.ArmBallRenderer
        elif environment=="armtwoballs":
            params['env_config'] = dict(m_mins=[-1.] * 7, m_maxs=[1.] * 7, s_mins=[-1.] * embedding_size, s_maxs=[1.] * embedding_size,
                                    arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05], arm_angle_shift=0.5, arm_rest_state=[0.] * 7,
                                    ball_size=0.1, balls_initial_position=[0.6, 0.6, -0.3, -0.3])
            params['env'] = environments.ArmTwoBallsDynamic
            params['renderer'] = rendering.ArmTwoBallsRenderer
        elif environment=='armarrow':
            params['env_config'] = dict(m_mins=[-1.] * 7, m_maxs=[1.] * 7, s_mins=[-1.] * embedding_size, s_maxs=[1.] * embedding_size,
                                    arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05], arm_angle_shift=0.5, arm_rest_state=[0.] * 7,
                                    arrow_size=0.1, arrow_initial_pose=[0.6, 0.6, 0.6])
            params['env'] = environments.ArmArrowDynamic
            params['renderer'] = rendering.ArmArrowRenderer
        elif environment=="armtoolball":
            params['env_config'] = dict(m_mins=[-1.] * 7, m_maxs=[1.] * 7, s_mins=[-1.] * embedding_size, s_maxs=[1.] * embedding_size,
                                    arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05], arm_angle_shift=0.5, arm_rest_state=[0.] * 7,
                                    ball_size=0.1, ball_initial_position=[0.6, 0.6],tool_length=4./7.,
                                    tool_initial_pose=[-0.7, -0.7, -0.7])
            params['env'] = environments.ArmToolBallDynamic
            params['renderer'] = rendering.ArmToolBallRenderer
        # Depending on the model, we set other parameters
        if embedding=='vae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.VariationalAutoEncoderEmbedding
            params['belief'] = 'normal'
        elif embedding=='bvae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.BetaVariationalAutoEncoderEmbedding
            params['belief'] = 'normal'
        elif embedding=='ae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.AutoEncoderEmbedding
            params['belief'] = 'kde'
        elif embedding=='sdae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.DenoisingAutoEncoderEmbedding
            params['belief'] = 'kde'
        elif embedding=='pfvae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.PlanarVariationalAutoEncoderEmbedding
            params['belief'] = 'normal'
        elif embedding=='rfvae':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.RadialVariationalAutoEncoderEmbedding
            params['belief'] = 'normal'
        elif embedding=='isomap':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.IsomapEmbedding
            params['belief'] = 'kde'
        elif embedding=='pca':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.PcaEmbedding
            params['belief'] = 'kde'
        elif embedding=='mds':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.MdsEmbedding
            params['belief'] = 'kde'
        elif embedding=='tsne':
            params['emb_size'] = embedding_size
            params['embedding'] = embeddings.TsneEmbedding
            params['belief'] = 'kde'
        elif embedding=='state':
            params['emb_size'] = embedding_size
            params['embedding'] = 'state'
            params['belief'] = 'uniform'
        elif embedding=='none':
            params['emb_size'] = embedding_size
            params['embedding'] = 'none'
            params['belief'] = 'uniform'
        # We set the rest of the parameters
        params["noise"] = noise
        params["deformation"] = deformation
        params["outliers"] = outliers
        params["distractor"] = distraction
        params['sampling'] = sampling
        params["nb_samples"] = sampling_size
        params['explo_ratio'] = 0.05

        return Experiment(name=name, params_dict=params, max_attempts=MAX_ATTEMPTS,
                          log_path=self._log_path, res_path=self._res_path)

if __name__ == '__main__':

    # If the user wants, we remove the past results and logs
    remove_past_exp = raw_input("If you want to remove the last logs write [remove]:")
    if remove_past_exp == 'remove':
        print("Removing logs and results")
        os.system("rm -rf logs/*")
        os.system("rm -rf results/*")
        time.sleep(10)

    # If the user wants it can run Tests
    perform_test = raw_input("If you want to perform test write [test]:")
    perform_test = perform_test == "test"
    if perform_test:
        NB_POINTS_MANIFOLD_MEASURES = int(1e2)
        NB_SAMPLES_DIVERGENCE_ESTIMATION = int(1e2)
        NB_SAMPLES_MSE = int(1e2)
        NB_BINS_EXPLORATION = 10
        NB_ITERATIONS_EXPLORATION = int(1e2)
        EXPLORATION_CALLBACK_PERIOD = int(1e2-1)
        NB_RUNS_PER_EXPE = 1
        MAX_ATTEMPTS = 1
    else:
        NB_POINTS_MANIFOLD_MEASURES=int(1e3)
        NB_SAMPLES_DIVERGENCE_ESTIMATION=int(1e3)
        NB_SAMPLES_MSE=int(1e2)
        NB_BINS_EXPLORATION=10
        NB_ITERATIONS_EXPLORATION=int(2e4)
        EXPLORATION_CALLBACK_PERIOD=int(1e2)
        NB_RUNS_PER_EXPE=1
        MAX_ATTEMPTS = 3

    # We start tensorboard
    subprocess.Popen("tensorboard --logdir logs", shell=True)

    # We set the joblib temp to the main folder
    os.environ["JOBLIB_TEMP_FOLDER"] = "."

    # We start the campaign
    with Campaign(name="Campaign", params_dict={'test':perform_test}, log_path='logs', res_path='results') as campaign:
        campaign.run()
