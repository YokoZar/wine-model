#!/usr/bin/env python3

###
### Wine-Model v 1.2
###  A python script to model Wine development created by Scott Ritchie
###
### Copyright (c) 2009-2014 Scott Ritchie <scottritchie@ubuntu.com>
### Licensed under the MIT License.  See the LICENSE file for details.
###
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
### Version 1.2 changes:
###  * Use Pandas instead of cairoplot
### Version 1.1 changes:
###  * Relative bug probability by default


import random
import time
from math import sqrt
import pandas
import matplotlib.pyplot as plt

### Basic setup
numberOfBugs, numberOfApps, numberOfUsers = 10000, 2500, 5000
minAppBugs, maxAppBugs = 150, 900 # Applications pick from between these two numbers using a uniform distribution
# Set this to True if you want to ignore the above pre-set number of bugs and instead use the alternative App Maker (see bug probability below)
useAlternativeAppMaker = True
# Number of apps a user uses, not the number of users an app has
minUserApps, maxUserApps = 1, 10 
# Set this to True to prevent making the wine-model.log file (the chart will still be made)
dontLog = False
print("Modeling with", numberOfBugs, "bugs,", numberOfApps, "apps, and", numberOfUsers, "users")
if not useAlternativeAppMaker:
    print("From", minAppBugs, "to", maxAppBugs, "bugs per app and from", minUserApps, "to", maxUserApps, "apps per user")
else:
    print("Using relative probabilities for individual bugs and from", minUserApps, "to", maxUserApps, "apps per user")

chartTitle = "Simulated model of Wine development" # Appears at the top of the chart produced at the end
###

### Relative difficulty of bugs
## bugDifficulty is the number of days it takes to solve a bug.
## When a bug is worked on, it's difficulty is reduced by one until it is 0, so some bugs need to be "solved" (worked on) multiple times.
# Set all to 1 to have all bugs be equally difficult.
#bugDifficulty = dict( (x,1) for x in range(numberOfBugs) ) 
## Here, a positive, almost normally distributed number of days per bug.  Average is just under 5 days per bug, with about 10% taking only 1 day.
bugDifficulty = dict( (x, abs(int(random.normalvariate(4,3))) + 1) for x in range(numberOfBugs) )  
###

### Relative probability of bugs and applications
## bugProbability is a list of the relative probabilities that those bugs will be assigned to an application.  So if bugProbability = [1,2], then the second bug is twice as likely to appear as the first.  Thus really common bugs will have higher numbers.
## if useAlternativeAppMaker = True, then the minAppBugs and maxAppBugs variables are ignored.  Instead, the highest number in bugProbability is interpretted as 100% and will affect all apps; meanwhile lower numbers will be proportionately less likely to affect an app.  So if bugProbability = [1, 2, 4], then the third bug will affect all apps, the second bug will have a 50% chance of affecting any particular app, and the first bug will have a 25% chance.
## appProbability, meanwhile, is the relative probability that that application will be listed on a user.  Thus the most popular application will have a higher number.
# Try pareto-distribution probability.  The 2.2 number was more or less pulled from a hat based on the intuition that a typical bug is about 60 times less likely than the most common bug
#bugProbability = [random.paretovariate(2.2) for x in range(numberOfBugs)]

# use this guy for Vince's idea of "most apps should have one or two bugs that only affect them" -- have useAlternativeAppMaker = True
#bugProbability = [numberOfApps] + [1 for x in range(numberOfApps)] + ... # make the relative probability always between 1 and number of Apps, so on average all the "1" bugs will affect one app
# TODO: implement "80/20" rule here, think about what it means a bit
# 80/20 rule: make 100 max.  Then last 80% can be 20, and first 20% can be 80...
bugProbability = [1.0/sqrt(x+1) for x in range(numberOfBugs)] # zipfs law
# 

appProbability = [random.paretovariate(2.2) for x in range(numberOfApps)]
###

totalTimeToSolve = sum(bugDifficulty.values())

print(totalTimeToSolve, "total days to solve every bug, an average of", totalTimeToSolve/numberOfBugs, "days per bug.")

# TODO: make neater, let it take more configuration data rather than be manually edited
def pick_strategy(day, allowPrevious=True):
    """ Returns a strategy function based on the day.  This is meant to be modified by user.
    If allowPrevious is set to False, then a strategy that doesn't always return a valid bug should never be picked.
    """
    ### This will work on the previously worked on bug until it's done.
    ### Uncomment these two lines if you want to avoid picking a new strategy until each bug is solved.
    #if allowPrevious:
        #return "pickPrevious" 
    ### There are 12 normal strategies to choose from
        #return pick_random_app
        #return pick_first_unsolved_app
        #return pick_nearest_done_app
        #return pick_random_bug
        #return pick_next_bug
        #return pick_most_popular_app
        #return pick_most_common_bug
        #return pick_easiest
        #return pick_random_user
        #return pick_first_unhappy_user
        #return pick_random_least_unhappy_user
        #return pick_first_least_unhappy_user

    ### You can select the strategy based on the day
    if day < 300: # eg do nothing but this strategy for the first 300 days
        return pick_most_common_bug
    ### "Realistic" model: do different plausible strategies for each day of the week
    if day %7 == 6: return pick_most_popular_app
    if day %7 == 5: return pick_most_common_bug
    if day %7 == 4: return pick_nearest_done_app
    if day %7 == 3: return pick_first_unsolved_app
    if day %7 == 2: return pick_first_unhappy_user
    if day %7 == 1: return pick_easiest
    if day %7 == 0: return pick_first_least_unhappy_user


### ----------------------------------------------------------------------------
###  You shouldn't need to modify anything below here to just run a simulation
### ----------------------------------------------------------------------------


# TODO: remove this function
def pick_two_strategies(day):
    """Returns a pair of strategies.  The first is done unless it's impossible, then the second is used as a backup.
    """
    return pick_strategy(day), pick_strategy(day, allowPrevious=False)

### 
### Log file
###

if not dontLog:
    try:
        logfile = open('wine-model.log', 'w')
        try:
            logfile.write("Bugs %i Apps %i Users %i Min App Bugs %i Max App Bugs %i Min User Apps %i Max User Apps %i \n" % 
                (numberOfBugs, numberOfApps, numberOfUsers, minAppBugs, maxAppBugs, minUserApps, maxUserApps) )
        finally:
            logfile.close()
    except IOError: #cannot open file, don't log
        print("Cannot open log file")
        dontLog = True
        pass
###
### Program logic below
###

# TODO: reconsider the naming here
def alternative_make_app(probability, bugs=False):
    """Returns a set of bug numbers that this app depends on.

    Just like make_app, but not using prior information about number of bugs and instead using the relative probability for every bug. probability instead of relative probability.
    Inputs:
    probability, a list of size equal to the number of bugs possible.
        The values of the probability list are the relative probability of that bug being selected. 1 = normal, .5 = half as likely, 2 = twice as likely, and so on.
        Basically, for every bug, we do a probability check to see if we should actually do it.
    bugs, ignored
    """
    possibleBugs = len(probability)

    maxProbability = max(probability)
    appBugs = set([]) # There should be no duplicates, and order doesn't matter, so a set is faster than a list here
    for x in range(possibleBugs):
        if random.uniform(0, maxProbability) <= probability[x]: #roll the dice
            appBugs.add(x)
    return set(appBugs) # We used to make this a frozen set, but now we trim the app in check_apps so subsequent scans of it go faster.    

# TODO: review for speed, possible refactor, possibly make probability+bugs tuples
def make_app(probability, bugs):
    """Returns a set of bug numbers that this app depends on.
    
    Inputs:
    probability, a list of size equal to the number of bugs possible.
        The values of the probability list are the relative probability of that bug being selected. 1 = normal, .5 = half as likely, 2 = twice as likely, and so on.
        Basically, when we pick a bug, we do a probability check to see if we should actually do it.  Otherwise we roll again.
    bugs, an integer for the number of bugs this application will have.
    """
    possibleBugs = len(probability)
    if bugs > possibleBugs: 
        raise(IndexError) # Bugs is greater than possible bugs

    maxProbability = max(probability)
    appBugs = set([]) # There should be no duplicates, and order doesn't matter, so a set is faster than a list here
    for x in range(bugs):
        while(True):
            thisBug = random.randint(0,possibleBugs - 1) #consider a bug.  We subtract 1 here due to the len command.
            if (thisBug not in appBugs) and random.uniform(0, maxProbability) <= probability[thisBug]: #roll the dice, but only if this is a new bug
                appBugs.add(thisBug)
                break # keep trying until we succeed
    return set(appBugs) # We used to make this a frozen set, but now we trim the app in check_apps so subsequent scans of it go faster.

# TODO: decorate as strategy; check for speed
def pick_next_bug(bugsSolved):
    """Picks the smallest bug number not in the bugsSolved list
    Input: bugsSolved
    """
    x = 0 # return 0 if we can, otherwise we have to keep adding 1 until we're not in bugsSolved
    while(True):
        if x not in bugsSolved: return x
        x += 1

# TODO: decorate as strategy; check for speed
def pick_random_bug(bugsSolved, numberOfBugs):
    """Picks a new bug not in the bugsSolved list at random
    Inputs:
        bugsSolved: a list of integers representing bugs already solved
        numberOfBugs: total possible number of bugs
    """
    #while(True):
    #    x = random.randint(0,numberOfBugs - 1)
    #    if not x in bugsSolved:
    #        return x
    #This above is technically equivalent but is really, really slow
    return random.choice([x for x in range(numberOfBugs) if not x in bugsSolved])

# TODO: decorate as strategy; check for speed
def pick_random_app(bugsSolved, apps):
    """Picks a new bug not in the bugsSolved list by randomly selecting an app and selecting a random unsolved bug in it.  We assume that apps has been passed through check_apps.  If all apps are solved, returns pick_next_bug
    """
    unsolvedApps = [a for a in apps if not apps[a] is True] # Exclude apps that are set to True because already solved 
    if unsolvedApps == []: # all apps are solved
        return pick_next_bug(bugsSolved)
    appToSolve = apps[random.choice(unsolvedApps)]
    return random.choice([y for y in appToSolve if not y in bugsSolved]) 

# TODO: decorate as strategy; check for speed
def pick_random_user(bugsSolved, apps, users):
    """Picks a random unhappy user, picks a random one of his unsolved applications, and then picks a random bug in it.  If all users are happy, picks a random App instead.
    """
    unhappyUsers = [a for a in users.keys() if not users[a] is True]
    if unhappyUsers == []: #happens when all users are happy
        return pick_random_app(bugsSolved, apps)
    thisUser = random.choice(unhappyUsers) # Gets a random unhappy user, which is a list of apps. happy users are set to True
    limitedApps = dict( (x,apps[x]) for x in users[thisUser] ) # Make a new "apps" that is only this user's apps
    return pick_random_app(bugsSolved, limitedApps) # Then just use our existing pick_random_app function

# TODO: decorate as strategy; check for speed
def pick_first_unhappy_user(bugsSolved, apps, users):
    """Returns an unsolved bug from an application from the first unhappy user.  If all users are happy, returns a bug from an unused application.
    """
    unhappyUsers = [a for a in users.keys() if not users[a] is True]
    if unhappyUsers == []: #happens when all users are happy
        return pick_random_app(bugsSolved, apps)
    thisUser = unhappyUsers[0] # thisUser is a list of apps
    limitedApps = dict( (x,apps[x]) for x in users[thisUser] ) # Make a new "apps" that is only this user's apps
    return pick_random_app(bugsSolved, limitedApps) # Then just use our existing pick_random_app function

# TODO: decorate as strategy; check for speed
def pick_random_least_unhappy_user(bugsSolved, apps, users):
    """Returns an unsolved bug from an application from a random user among those closest to being happy (has the fewest nonworking applications).  Note that this is based on applications - it doesn't analyze total bugs left or difficulty.
    """
    unhappyUsers = [users[a] for a in users.keys() if not users[a] is True] # here we are converting users from a dictionary to a list of lists of apps
    if unhappyUsers == []: #happens when all users are happy
        return pick_random_app(bugsSolved, apps)
    unhappyUsers = [ [x for x in y if not x is True] for y in unhappyUsers ] # purge all solved apps from the unhappy users
    appsLeft = min([len(x) for x in unhappyUsers])
    unhappyUsers = [x for x in unhappyUsers if len(x) == appsLeft] # purge all unhappy users with more than the minimal apps left
    thisUser = random.choice(unhappyUsers) # thisUser is a list of apps
    limitedApps = dict( (x,apps[x]) for x in thisUser ) # Make a new "apps" that is only this user's apps
    return pick_random_app(bugsSolved, limitedApps) # Then just use our existing pick_random_app function    

# TODO: decorate as strategy; check for speed
def pick_first_least_unhappy_user(bugsSolved, apps, users):
    """Returns an unsolved bug from an application from a the first user among those closest to being happy (has the fewest nonworking applications).  Note that this is based on applications - it doesn't analyze total bugs left or difficulty.  Code here is duplicated from pickLeastUnhappyUser, with the exception that the lowest-numbered user is chosen rather than randomly.
    """
    unhappyUsers = [users[a] for a in users.keys() if not users[a] is True] # here we are converting users from a dictionary to a list of lists of apps
    if unhappyUsers == []: #happens when all users are happy
        return pick_random_app(bugsSolved, apps)
    unhappyUsers = [ [x for x in y if not x is True] for y in unhappyUsers ] # purge all solved apps from the unhappy users
    appsLeft = min([len(x) for x in unhappyUsers])
    unhappyUsers = [x for x in unhappyUsers if len(x) == appsLeft] # purge all unhappy users with more than the minimal apps left
    thisUser = unhappyUsers[0] # thisUser is a list of apps
    limitedApps = dict( (x,apps[x]) for x in thisUser ) # Make a new "apps" that is only this user's apps
    return pick_random_app(bugsSolved, limitedApps) # Then just use our existing pick_random_app function    

# TODO: consider apps[x] is True --> is SOLVED, define solved as True
# TODO: decorate as strategy; check for speed
def pick_first_unsolved_app(bugsSolved, apps):
    """Picks a new bug not in the bugsSolved list by randomly selecting an unsolved bug from the first app that has any open bugs.  We assume that there is at leat one app and that there are no apps with zero unsolved bugs which haven't yet been cleaned by check_apps.
    """
    lowest = False
    for x in apps:
        if not apps[x] is True: #Note that this will only occur when apps[x] is a list (Or frozenset), which is what we want
            lowest = x
    if lowest: return random.choice([x for x in apps[lowest] if not x in bugsSolved])
    else: return pick_next_bug(bugsSolved) # occurs when all apps are solved

# TODO: decorate as strategy; check for speed
def pick_nearest_done_app(bugsSolved, apps):
    """Picks a new bug not in the bugsSolved list by randomly selecting an unsolved bug from the app with the least bugs remaining.  We assume that there is at least one app and that there are no apps with zero unsolved bugs.
    """
    openBugCount = dict ( (len([y for y in apps[x] if y not in bugsSolved]), x) for x in apps if not apps[x] == True) # Create a dictionary of key: open bugs to value: app number, excluding ones that are solved
    if openBugCount: # in case all bugs are solved already
        bestApp = apps[openBugCount[min(openBugCount)]] # The one with the smallest open bugs is the best app
        return random.choice([x for x in bestApp if not x in bugsSolved])
    else: # openBugCount is empty, all apps are solved, so any bug will work fine:
        return pick_next_bug(bugsSolved)

# TODO: this smells like it can be a smoother list construction
def sum_bugs (numberOfBugs, apps):
    """Returns a list of the frequency of each bug.  This will then be handled by bug_popularity
    inputs:
        numberOfBugs, an integer showing total number of bugs
        apps: here we assume that everything in x is an iterable
    """
    bugSums = [0 for x in range(numberOfBugs)]
    for x in apps:
        if apps[x] != True:  # In case apps has been passed to check_apps before summing here
            for y in apps[x]: # for every bug this app affects...
                bugSums[y] += 1 # add 1 to the sum of this bug
    return bugSums

# TODO: this smells rather inefficient
def bug_popularity (bugCount):
    """Returns a dictionary of affected app count:which bugs that affect that many apps
    Thus if 3 apps are affected by bugs 10, 11, and 12, then bug_popularity[3]=[10,11,12]
    
    input: bugCount, a list of how many apps each bug number affects
     bugcount is destroyed in this function
    """
    bug_popularity = dict( (x,[]) for x in bugCount )
    while(True):
        bug_popularity[bugCount.pop()].append(len(bugCount)) #iterate through bugCount and add the particular bugs to the list
        if len(bugCount) == 0: # once bugCount is out of items, then we are done building the dictionary
            return bug_popularity

# TODO: decorate as strategy; check for speed
def pick_most_common_bug(bugsSolved, numberOfBugs, apps, priority=False):
    """Picks a new bug not in bugsSolved based on which has the most frequency amongst nonworking apps
    If all apps have all their bugs solved, picks the next unsolved bug not associated with an app.

    inputs:
        bugsSolved: a set of all the current bugs solved
        numberOfBugs: an integer for the total number of bugs
        apps: TODO
        priority: TODO
    """
    if not priority:
        bugSums = bug_popularity (sum_bugs (numberOfBugs, apps)) #bugSums is a dictionary of the number of times a bug occurs[which bug it is]
    else: # priority was supplied, so we just use that rather than recounting the priority each time
        bugSums = priority
    x = max(bugSums)
    newMaxNeeded = False
    while(True):
        y = bugSums[x].pop() # we've already solved this bug, or are about to solve it, so remove it from further consideration
        if bugSums[x] == []:
            del bugSums[x]   # clean up so an empty list won't be "max" next time
            newMaxNeeded = True  # then we need to find a new max for this time
        if not y in bugsSolved:
            return y
        if newMaxNeeded: # only calculate that new max if we actually had to loop
            if bugSums == {}: # occurs when all apps are solved
                return pick_next_bug(bugsSolved)
            x = max(bugSums)
            newMaxNeeded = False

# TODO: decorate as strategy; check for speed
def pick_most_popular_app(bugsSolved, apps, users):
    """Returns an unsolved bug from the most popular application with open bugs.
        If no used application has open bugs, returns a bug from an unused application. If all apps are solved, returns pick_next_bug
    """
    solvedApps = [x for x in apps if apps[x] is True]
    if len(solvedApps) == len(apps): #all apps are solved
        return pick_next_bug(bugsSolved)
    try:
        mostPopularApp = pick_most_common_bug(solvedApps, len(apps), users) # pretend the apps are bugs here; "solved" bugs are then the apps that equal True because check_apps set them that way when all their bugs were solved.
    except ValueError: # This occurs when all apps somewhere in users are solved but there are still applications remaining
        return pick_first_unsolved_app(bugsSolved, apps)
    return random.choice([x for x in apps[mostPopularApp] if not x in bugsSolved])

# TODO: decorate as strategy; check for speed
def pick_easiest(bugsSolved, reverseBugDifficulty):
    """Returns the unsolved bug with the lowest difficulty
    Input:
        bugsSolved
        reverseBugDifficulty, a dictionary with key:difficulty to value: set(bugs with this difficulty)
    """
    easiestDifficulty = min(reverseBugDifficulty)
    while(True):
        easiest = reverseBugDifficulty[easiestDifficulty]
        if easiest <= bugsSolved: #if every bug in easiest set is in the bugsSolved set then we've already solved the apparent minimum, erase it and try again
            del(reverseBugDifficulty[easiestDifficulty])
            easiestDifficulty = min(reverseBugDifficulty)
        else: # there is some element in easiest not in bugsSolved
            return random.choice(list(easiest - bugsSolved))

# TODO: recheck for slowness
def check_apps(apps, bugsSolved):
    """Checks the applications dictionary for newly working applications, 
    Inputs:
        apps: a dictionary with key: app number to value: frozenset[bugs affecting it].  This dictionary is not preserved: if all bugs are working then a key will be reassigned to True to speedup future lookup.
        bugsSolved: a set of integers. Preserved.
    TODO: we might speed things up here by purging the bug from the app once it's solved.  Then we can recheck assumptions in randomApp and randomUser.  However this isn't doable since we currently use frozensets for apps[x],
    """
    solved = 0
    for x in apps: # For every app...
        if not apps[x] is True:
             apps[x] = apps[x] - bugsSolved # Remove all solved bugs from x
        if apps[x] is True or apps[x] == set([]): # All bugs are solved in this app:
            apps[x] = True
            solved += 1

    return solved

# TODO: use a with statement here
def append_to_log(entry):
    """Writes entry to the log.  Make sure input has a line break at the end.
    """
    try:
        logfile = open('wine-model.log', 'a')
        try:
            logfile.write(entry)
        finally:
            logfile.close()
    except IOError: #cannot open file, don't log
        pass

###
### Simulation setup
###

if useAlternativeAppMaker:
    apps = dict( (x, alternative_make_app(bugProbability, random.randint(minAppBugs,maxAppBugs))) for x in range(numberOfApps) ) #applications will have from minAppbugs to maxAppBugs, uniformly distributed
else:
    apps = dict( (x, make_app(bugProbability, random.randint(minAppBugs,maxAppBugs))) for x in range(numberOfApps) ) #applications will have from minAppbugs to maxAppBugs, uniformly distributed

users = dict( (x, make_app(appProbability, random.randint(minUserApps,maxUserApps))) for x in range(numberOfUsers) ) #Users will have from minUserApps to maxUserApps, uniformly distributed

priority = bug_popularity(sum_bugs(numberOfBugs, apps)) #for speeding up the pickMostPopular function
reverseBugDifficulty = {} #for speeding up the pick_easiest function.  This is a dictionary of key bugdifficulty to value set(apps that have tha difficulty)
for x in bugDifficulty:
    if bugDifficulty[x] not in reverseBugDifficulty:
        reverseBugDifficulty[bugDifficulty[x]]=set([x])
    else:
        reverseBugDifficulty[bugDifficulty[x]].add(x)

averageBugsPerApp = sum([len(apps[x]) for x in apps]) / numberOfApps
averageAppsPerUser = sum([len(users[x]) for x in users]) / numberOfUsers
print("Applications and users generated, averaging", averageBugsPerApp ,"bugs per app and", averageAppsPerUser ,"apps per user.  Starting simulation...")

###
### Simulation begins here
###

bugsSolved = set([]) # an (ordered?) list of integers of all bugs solved so far
day = 0

timespent = time.clock()
averageApps = 0
averageHappy = 0
hitHalfway = False
hitFirst = False
lastWorkedBug = False

append_to_log("Day, % Bugs Solved, % Working Apps, % Happy Users \n")
chartData = {"Working Apps" : [], "Happy Users" : []} # TODO: numpy.array!

progressIndicator = 0.10 # When to first show 'working on day' (x) progress indicators

while(True): 
    # Check for newly working apps every day we solved a bug in the previous day (otherwise, we might try and work on an already working app):
    if lastWorkedBug is False:
        workingApps = check_apps(apps,bugsSolved)
        happyUsers = check_apps(users,set([x for x in apps if apps[x] == True])) # The check_apps function can be used for users because it does the same thing

    #print("Day:",day,"Working apps:",workingApps,"Bugs solved:",len(bugsSolved))
    if workingApps >= 1 and not hitFirst:
        print("First app working on day", day)
        hitFirst = True
    if day >= totalTimeToSolve*progressIndicator:
        print("%i%% complete on day: " % (progressIndicator*100), day)
        progressIndicator += 0.10

    if not dontLog: # Log every day
        append_to_log("%f, %f, %f, %f \n" % (float(day), len(bugsSolved)/numberOfBugs, workingApps/numberOfApps, happyUsers/numberOfUsers) )
        #chartData["Bugs Solved"].append(len(bugsSolved)*100/numberOfBugs)
        chartData["Working Apps"].append(workingApps*100/numberOfApps)
        chartData["Happy Users"].append(happyUsers*100/numberOfUsers)

    # Pick a bug from those not yet solved:
    ### TODO: This could be more elegant if they could all be called in the same way, then we could just do
    # strategy(bugsSolved, apps, numberOfBugs, priority) ###
    strategy, backupStrategy = pick_two_strategies(day)
    if strategy == "pickPrevious": # If the strategy is pickPrevious, then by default we just use the previous bug to solve; otherwise we need to use the backup strategy
        if lastWorkedBug is False: # note we do not test lastWorkedBug == 0 in case we just solved bug number 0
            strategy = backupStrategy
        else:
            bugToSolve = lastWorkedBug
    if strategy == pick_random_app:
        bugToSolve = pick_random_app(bugsSolved, apps)
    if strategy == pick_first_unsolved_app:
        bugToSolve = pick_first_unsolved_app(bugsSolved, apps)
    if strategy == pick_nearest_done_app:
        bugToSolve = pick_nearest_done_app(bugsSolved, apps)
    if strategy == pick_random_bug:
        bugToSolve = pick_random_bug(bugsSolved, numberOfBugs)
    if strategy == pick_most_common_bug:
        bugToSolve = pick_most_common_bug(bugsSolved, numberOfBugs, apps, priority)
    if strategy == pick_next_bug:
        bugToSolve = pick_next_bug(bugsSolved)
    if strategy == pick_most_popular_app:
        bugToSolve = pick_most_popular_app(bugsSolved, apps, users)
    if strategy == pick_easiest:
        bugToSolve = pick_easiest(bugsSolved, reverseBugDifficulty)
    if strategy == pick_random_user:
        bugToSolve = pick_random_user(bugsSolved, apps, users)
    if strategy == pick_first_unhappy_user:
        bugToSolve = pick_first_unhappy_user(bugsSolved, apps, users)
    if strategy == pick_random_least_unhappy_user:
        bugToSolve = pick_random_least_unhappy_user(bugsSolved, apps, users)
    if strategy == pick_first_least_unhappy_user:
        bugToSolve = pick_first_least_unhappy_user(bugsSolved, apps, users)

    # And take bugDifficulty days to solve it
    day += 1 
    lastWorkedBug = bugToSolve # In case we want to work on it again tomorrow

    # Increment the counters for happiness and working apps.  We multiply by bugDifficulty since we are going that many days before solving the next bug.  We don't update the bugsSolved list with check_apps until after this time.
    averageApps += workingApps # average apps is actually a sum of apps working * days they've been working -- divide by day to get the actual average
    averageHappy += happyUsers # TODO: not an average, rename

    bugDifficulty[bugToSolve] -= 1
    if bugDifficulty[bugToSolve] <= 0:
        bugsSolved.add(bugToSolve)
        lastWorkedBug = False # this is the number of the bug we last worked on, in case we want to do it again

    if len(bugsSolved) == numberOfBugs:
        print("All bugs solved on day", day)
        break

print("CPU time taken for simulation:", (time.clock() - timespent))
print("Apps Working * Days:", averageApps, ", average", averageApps/day, "per day.")
print("Happy Users * Days:", averageHappy, ", average", averageHappy/day, "per day.")
# Final log entry - everything is done here
if not dontLog: append_to_log("%f, 1.0, 1.0, 1.0 \n" % (float(day)) )

print("Now making chart.")

chart = pandas.DataFrame(chartData)
chart.plot()
plt.ylabel("Percentage")
plt.xlabel("Man-hours invested")
plt.savefig('wine-model-results.svg')
