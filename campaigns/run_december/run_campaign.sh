rm log.txt; 
export EXP_INTERP='/home/apere/Applications/anaconda3/envs/py2.7/bin/python' ;
echo '=================> Performing RPE';
echo '=================> Performing RPE' >> log.txt;
echo '=================> Rpe Armball 2017-12-08 11:36:10.511145';
echo '=================> Rpe Armball 2017-12-08 11:36:10.511145' >> log.txt;
$EXP_INTERP rpe.py armball --path=results --name='Rpe Armball 2017-12-08 11:36:10.511145' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rpe Armarrow 2017-12-08 11:36:10.511184';
echo '=================> Rpe Armarrow 2017-12-08 11:36:10.511184' >> log.txt;
$EXP_INTERP rpe.py armarrow --path=results --name='Rpe Armarrow 2017-12-08 11:36:10.511184' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Performing RGE-FI';
echo '=================> Performing RGE-FI'>> log.txt;
echo '=================> Rge-Fi Armball 2017-12-08 11:36:10.511198';
echo '=================> Rge-Fi Armball 2017-12-08 11:36:10.511198' >> log.txt;
$EXP_INTERP rge_fi.py armball --path=results --name='Rge-Fi Armball 2017-12-08 11:36:10.511198' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Fi Armarrow 2017-12-08 11:36:10.511206';
echo '=================> Rge-Fi Armarrow 2017-12-08 11:36:10.511206' >> log.txt;
$EXP_INTERP rge_fi.py armarrow --path=results --name='Rge-Fi Armarrow 2017-12-08 11:36:10.511206' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Performing RGE-REP large dims';
echo '=================> Performing RGE-REP large dims'>> log.txt;
echo '=================> Rge-Rep Ae Armball 2017-12-08 11:36:10.511216';
echo '=================> Rge-Rep Ae Armball 2017-12-08 11:36:10.511216' >> log.txt;
$EXP_INTERP rge_rep.py ae armball --path=results --name='Rge-Rep Ae Armball 2017-12-08 11:36:10.511216' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511225';
echo '=================> Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511225' >> log.txt;
$EXP_INTERP rge_rep.py ae armarrow --path=results --name='Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511225' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Vae Armball 2017-12-08 11:36:10.511232';
echo '=================> Rge-Rep Vae Armball 2017-12-08 11:36:10.511232' >> log.txt;
$EXP_INTERP rge_rep.py vae armball --path=results --name='Rge-Rep Vae Armball 2017-12-08 11:36:10.511232' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511239';
echo '=================> Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511239' >> log.txt;
$EXP_INTERP rge_rep.py vae armarrow --path=results --name='Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511239' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511247';
echo '=================> Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511247' >> log.txt;
$EXP_INTERP rge_rep.py rfvae armball --path=results --name='Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511247' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511254';
echo '=================> Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511254' >> log.txt;
$EXP_INTERP rge_rep.py rfvae armarrow --path=results --name='Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511254' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Isomap Armball 2017-12-08 11:36:10.511261';
echo '=================> Rge-Rep Isomap Armball 2017-12-08 11:36:10.511261' >> log.txt;
$EXP_INTERP rge_rep.py isomap armball --path=results --name='Rge-Rep Isomap Armball 2017-12-08 11:36:10.511261' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511269';
echo '=================> Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511269' >> log.txt;
$EXP_INTERP rge_rep.py isomap armarrow --path=results --name='Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511269' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Pca Armball 2017-12-08 11:36:10.511393';
echo '=================> Rge-Rep Pca Armball 2017-12-08 11:36:10.511393' >> log.txt;
$EXP_INTERP rge_rep.py pca armball --path=results --name='Rge-Rep Pca Armball 2017-12-08 11:36:10.511393' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511408';
echo '=================> Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511408' >> log.txt;
$EXP_INTERP rge_rep.py pca armarrow --path=results --name='Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511408' --nlatents=10 || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Performing RGE-REP small dims';
echo '=================> Performing RGE-REP small dims'>> log.txt;
echo '=================> Rge-Rep Ae Armball 2017-12-08 11:36:10.511417';
echo '=================> Rge-Rep Ae Armball 2017-12-08 11:36:10.511417' >> log.txt;
$EXP_INTERP rge_rep.py ae armball --path=results --name='Rge-Rep Ae Armball 2017-12-08 11:36:10.511417 --nlatents=2' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511425';
echo '=================> Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511425' >> log.txt;
$EXP_INTERP rge_rep.py ae armarrow --path=results --name='Rge-Rep Ae Armarrow 2017-12-08 11:36:10.511425 --nlatents=4' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Vae Armball 2017-12-08 11:36:10.511432';
echo '=================> Rge-Rep Vae Armball 2017-12-08 11:36:10.511432' >> log.txt;
$EXP_INTERP rge_rep.py vae armball --path=results --name='Rge-Rep Vae Armball 2017-12-08 11:36:10.511432 --nlatents=2' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511439';
echo '=================> Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511439' >> log.txt;
$EXP_INTERP rge_rep.py vae armarrow --path=results --name='Rge-Rep Vae Armarrow 2017-12-08 11:36:10.511439 --nlatents=4' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511447';
echo '=================> Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511447' >> log.txt;
$EXP_INTERP rge_rep.py rfvae armball --path=results --name='Rge-Rep Rfvae Armball 2017-12-08 11:36:10.511447 --nlatents=2' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511453';
echo '=================> Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511453' >> log.txt;
$EXP_INTERP rge_rep.py rfvae armarrow --path=results --name='Rge-Rep Rfvae Armarrow 2017-12-08 11:36:10.511453 --nlatents=4' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Isomap Armball 2017-12-08 11:36:10.511461';
echo '=================> Rge-Rep Isomap Armball 2017-12-08 11:36:10.511461' >> log.txt;
$EXP_INTERP rge_rep.py isomap armball --path=results --name='Rge-Rep Isomap Armball 2017-12-08 11:36:10.511461 --nlatents=2' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511470';
echo '=================> Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511470' >> log.txt;
$EXP_INTERP rge_rep.py isomap armarrow --path=results --name='Rge-Rep Isomap Armarrow 2017-12-08 11:36:10.511470 --nlatents=4' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Pca Armball 2017-12-08 11:36:10.511478';
echo '=================> Rge-Rep Pca Armball 2017-12-08 11:36:10.511478' >> log.txt;
$EXP_INTERP rge_rep.py pca armball --path=results --name='Rge-Rep Pca Armball 2017-12-08 11:36:10.511478 --nlatents=2' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
echo '=================> Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511485';
echo '=================> Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511485' >> log.txt;
$EXP_INTERP rge_rep.py pca armarrow --path=results --name='Rge-Rep Pca Armarrow 2017-12-08 11:36:10.511485 --nlatents=4' || (echo 'FAILURE' && echo 'FAILURE' >> log.txt); 
