wine-model
==========

Wine-model is a script for simulating the growth of a project in terms of features completed, happy
users, and work items.  A neat looking chart of these with respect to time is generated as a result.

This project started out as a thought experiment for the Wine project.  I initially wrote it 
honestly not knowing what the resulting graphs would look like.  You can 
[read the initial blog post here](http://yokozar.org/blog/archives/48).

The theoretical model is fairly straightforward: a project (like Wine) consists of a set of
uncompleted possible **work items** (eg bugs).  Each of these items has an associated difficulty, or
time in man-hours to complete.  **Features** are defined as overlapping sets of these work items; a
feature is considered done when all work items are complete. **Users**, in turn, are defined as sets
of features; a user is considered happy when all their features are complete.

There are two main lessons from the model: first, the *order* in which work items are done can 
matter greatly.  With the right strategy, there may be 10 times as many happy users at a given time
than under a less ideal strategy, even though both will "finish" at the same time.

Secondly, the shape of the happy users curve itself can be quite surprising.  Under the default
settings and strategy, happy users follows a roughly linear growth before dramatically angling
upwards.  Initially, the project mostly gains happy users as individuals one at a time by completing
relatively distinct feature sets.  Eventually, however, there is enough "collateral damage" as work
on one task inadvertently satisfies diverse groups of users who were otherwise almost happy.

Usage:
==========

To run the script, just put the wine-model.py file in a folder and run.  It depends on the pandas
library, which should be available using pip3 if not otherwise available.

The top portion of the script is meant to be played with and modified by the user.  In the future
this could be done with configuration files and command line switches if there is sufficient user
demand.

It will display some progress in the terminal, and then generate two files: wine-model-results.svg
and wine-model-results.csv.  If you want to make your own chart, you can feed the csv into a
spreadsheet program.
 
Be aware, the program can take some time to run.  The default settings should take a few minutes.

Known shortcomings:
==========

 * The landscape does not change during the course of the model: users do not change what they want,
   new work items do not become possible, features do not break, and so on.
 * We do not model partially working features nor partially happy users.  You can somewhat adjust
   for this by simply modelling more features and users, as some users will become supersets of 
   others.
 * We don't (yet) model user or feature archetypes.  A user wanting feature A may have a 10 times
   greater chance of also needing feature B compared with a typical user, but currently specific 
   features and work items aren't deliberately clustered.
 * The strategies used in the model currently operate with perfect information: the "easiest work
   item" or "most popular feature among users" strategies know, precisely, the right answer to
   those questions.  Some real world strategies might be significantly more prone to error than
   others.  To some extent you can reflect reality by alternating between these and more random
   strategies.
 * We don't explicitly model work items that depend on other work items being completed first.
 * And, of course, the actual models used for generating user, feature, and work item sets are open
   for debate -- is Zipf's law or a Pareto distribution a better model for predicting which features
   will be important for users?  If you have an opinion on questions like that, please feel free to
   share and discuss it, and I will do my best to implement it for you.
 * We don't model "dirty fixes", ie solutions to bugs that increase the time to perfection (perhaps 
   by adding new bugs or more difficulty to existing bugs).  These might be thought of as ugly hacks
   to the code that somehow manage to get an app working, which in Wine's case Alexandre opposes.

