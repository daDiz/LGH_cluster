# LGH_server
Using local graphlet frequency to predict Hawkes process event count

Note: LGH_server is desired to run on a cluster. It has been tested on big red 2, a supercomputer at Indiana University. 

## Set up
```
$ mv LoGraPov_v1.0 ~/
$ mv E-CLoG_code ~/
$ cd ~/E-CLoG_code
$ make
```
```
$ mv soc-dolphins /path/to/data
$ cd /path/to/data
```
Edit config.json and change /path/to/data to the actual path. 

