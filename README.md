# LGH_cluster
Using local graphlet frequency to predict Hawkes process event count

Note: LGH_server is desired to run on a cluster. It has been tested on Big Red II at Indiana University https://kb.iu.edu/d/bcqt. 

## Set up
The following modules need to be loaded on the cluster. 
```
module swap PrgEnv-cray PrgEnv-gnu
module load pcp/2008
module load python/3.6.5
```

Download the codes, and save them in the proper directory. 
```
$ mv LoGraPov_v1.0 ~/
$ mv E-CLoG_code ~/
```
Compile E-CLoG. 
```
$ cd ~/E-CLoG_code
$ make
```
Save the data files in a seperate directory, such as /N/dc2/scratch/usr_name.
```
$ mv soc-dolphins /path/to/data
$ cd /path/to/data
```
Edit config.json and change /path/to/data to the actual path. 
Before running the program, edit config.json setting the parameters. 
To run the program, ./run_lograpov.py task config.json (where task = initialize, rewire, hawkes, eclog, node_centric, or linear_regression).

## initialize
```
$ cd /path/to/data
$ ./run_lograpov.py initialize config.json
```
After this step, three folders (rewired, hawkes, and eclog) and four pbs files (submit_bundle_sample_hawkes.pbs, submit_linear_regression.pbs, submit_edge2node.pbs, and submit_rewiring.pbs) would be generated. 

## rewire
```
$ ./run_lograpov.py rewire config.json
```
This generates rewired graphs with degree distribution preserved. 
After rewiring completes, the following two steps can be performed simultaneously. 

## hawkes
```
$ ./run_lograpov.py hawkes config.json
```
This starts sample hawkes. 

## eclog
```
$ ./run_lograpov.py eclog config.json
```
This runs eclog. 
The following steps should be performed after hawkes and eclog finish. 

## node_centric
```
$ ./run_lograpov.py node_centric config.json
```
This converts edge-centric counts to node-centric counts. 

## linear_regression
```
$ ./run_lograpov.py linear_regression config.json
```
This fits a linear regression model, writing r2-score and mse to linear_regression.log and coefficients of the linear regression to *_linear_coef.txt.

## input file format
For edge list file format, please see soc-dolphins/soc-dolphins.txt. 

For config.json format, please see soc-dolphins/config.json.

## parallelism illustration
```
      initialize
           |
        rewire
       /      \
      /        \
   hawkes     eclog 
      \        /
       \      /
     node_centric
           |
   linear_regression
 ```
