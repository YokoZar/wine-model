wine-model
==========

A data model for predicting how good Wine will be

### HOWTO:
###  First, read this blog post to learn a bit about the theory behind the model:
###   http://yokozar.org/blog/archives/48
### 
### To run the script, just put this file and cairoplot.py in a folder and run.
### It will display some progress in the terminal, and then generate two files: 
### wine-model-results.svg and wine-model.log. wine-model-results.svg is a 
### pretty chart made using cairoplot of the results, and wine-model.log is a 
### complete log of % apps complete and % happy users per day.  If you want to 
### make your own chart, you can easily parse and feed wine-model.log into a
### spreadsheet program.
### 
### Be aware, the program can take some time to run.  The default settings take
### a few minutes on my 1.5 ghz core2duo.
###
### Known shortcomings:
###  * We don't model applications that partially work despite having bugs, 
###    nor do we model partially happy users.
###  * We don't model new apps or bugs that get created during development 
###    (eg regressions or new versions of Windows)
###  * APIs are given a relative probability of being used, however we don't 
###    cluster them realistically (eg two direct3D APIs may both occur in 10% 
###    of real world programs, but they are 100% found together)  This means some
###    bugs should be "paired", however we can roughly simulate this by just
###    making a particular bug represent the pair and take longer to solve. When
###    the pairing reflects a correlation rather than 100%, however, we can't 
###    model it so specifically.
###  * Similarly, we don't cluster users.  Someone who needs Word to work is 
###    more likely to need Excel to work, but we just assign both a static 
###    probability of being needed by a user.
###  * And, of course, the actual methods for modelling bugs, applications, and
###    users may not reflect reality - however, if you have a better idea, you
###    can work it into this very script.  Please share so we can discuss it :)
###

