#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import *
import datetime

PATH_TO_RESULTS = "results"
PATH_TO_INTERPRETER = "/home/apere/anaconda3/envs/py-2.7/bin/python"

reps = ["ae", "vae", "rfvae", "isomap", "pca"]
envs = ['armball', 'armarrow']

with open('run_campaign.sh', 'w') as f:
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n"%PATH_TO_INTERPRETER)
    for i in range(5):
        for rep in reps:
            env = 'armarrow'
            name = ("RGE-REP %s %s %s" % (rep, env, str(datetime.datetime.now()))).title()
            f.write("echo '=================> %s';\n" % name)
            f.write("echo '=================> %s' >> log.txt;\n" % name)
            f.write("$EXP_INTERP rge_rep.py %s %s --path=%s --name='%s' --nlatents=3 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); \n" % (rep, env, PATH_TO_RESULTS, name))
