#!/usr/bin/env python3

###
### Wine-Model v 1.3
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
### Version 1.3 changes:
###  * Python 3
###  * Massive refactor
###  * New license
### Version 1.2 changes:
###  * Use Pandas instead of cairoplot
### Version 1.1 changes:
###  * Relative bug probability by default

import random
import time
from math import sqrt
from operator import itemgetter
import pandas
import matplotlib.pyplot as plt

LOGFILE = 'wine-model.log'
SOLVED = True

random.seed(a=12345) #TODO: allow command-line pass to declare this (otherwise real random)

### Basic setup
# TODO: constants == caps or make them defined by arguments parser
numberOfBugs, numberOfApps, numberOfUsers = 10000, 2500, 5000
numberOfBugs, numberOfApps, numberOfUsers = 1000, 250, 500 # TODO: remove, temporary fast for dev mode
minAppBugs, maxAppBugs = 150, 900 # Applications pick from between these two numbers using a uniform distribution
# Set this to True if you want to ignore the above pre-set number of bugs and instead use the alternative App Maker (see bug probability below)
useAlternativeAppMaker = True
# Number of apps a user uses, not the number of users an app has
minUserApps, maxUserApps = 1, 10 
# Set this to True to prevent making the wine-model.log file (the chart will still be made)
enable_log = True
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
#bugDifficulty = {x:1 for x in range(numberOfBugs)}
## Here, a positive, almost normally distributed number of days per bug.  Average is just under 5 days per bug, with about 10% taking only 1 day.
bugDifficulty = {x:abs(int(random.normalvariate(4,3))) + 1 for x in range(numberOfBugs)}
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
    #return pick_from_specific_unsolved_app
    #return pick_nearest_done_app
    #return pick_random_bug
    #return pick_specific_from_all_bugs
    #return pick_specific_from_most_popular_app
    #return pick_random_from_most_popular_app
    #return pick_from_most_common_by_feature
    #return pick_easiest
    #return pick_random_user
    #return pick_first_unhappy_user
    #return pick_random_least_unhappy_user
    #return pick_first_least_unhappy_user

    ### You can select the strategy based on the day
    if day < 300: # eg do nothing but this strategy for the first 300 days
        return pick_from_most_common_by_feature
    ### "Realistic" model: do different plausible strategies for each day of the week
    if day %7 == 6: return pick_random_from_most_popular_app
    if day %7 == 5: return pick_from_most_common_by_feature
    if day %7 == 4: return pick_nearest_done_app
    if day %7 == 3: return pick_from_specific_unsolved_app
    if day %7 == 2: return pick_first_unhappy_user
    if day %7 == 1: return pick_easiest
    if day %7 == 0: return pick_first_least_unhappy_user


### ----------------------------------------------------------------------------
###  You shouldn't need to modify anything below here to just run a simulation
### ----------------------------------------------------------------------------


# TODO: remove this function; all strategies should fallback to other ones
def pick_two_strategies(day):
    """Returns a pair of strategies.  The first is done unless it's impossible, then the second is used as a backup.
    """
    return pick_strategy(day), pick_strategy(day, allowPrevious=False)

# Start the log
if enable_log:
    with open(LOGFILE, 'w') as logfile:
        logfile.write("Bugs %i Apps %i Users %i Min App Bugs %i Max App Bugs %i Min User Apps %i Max User Apps %i \n" % 
            (numberOfBugs, numberOfApps, numberOfUsers, minAppBugs, maxAppBugs, minUserApps, maxUserApps) )

def append_to_log(entry):
    if enable_log:
        with open(LOGFILE, 'a') as logfile:
            logfile.write(entry)

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

# TODO: can be sped up by making it a generator
def pick_specific_from_all_bugs():
    """Picks the smallest bug number not in the bugsSolved list"""
    for x in range(numberOfBugs):
        if x not in bugsSolved: return x

# TODO: decorate as strategy; check for speed
# pick_random_from_all_bugs():
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
# TODO: fallback to random
def pick_random_app(bugsSolved, apps):
    """Picks a new bug not in the bugsSolved list by randomly selecting an app and selecting a random unsolved bug in it.  We assume that apps has been passed through check_apps.
    """
    unsolvedApps = [a for a in apps if not apps[a] is True] # Exclude apps that are set to True because already solved 
    if unsolvedApps == []: # all apps are solved
        return pick_specific_from_all_bugs()
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
    limitedApps = {x:apps[x] for x in users[thisUser]} # Make a new "apps" that is only this user's apps
    return pick_random_app(bugsSolved, limitedApps) # Then just use our existing pick_random_app function

# TODO: decorate as strategy; check for speed
# TODO: implement as partial of above
def pick_first_unhappy_user(bugsSolved, apps, users):
    """Returns an unsolved bug from an application from the first unhappy user.  If all users are happy, returns a bug from an unused application.
    """
    unhappyUsers = [a for a in users.keys() if not users[a] is True]
    if unhappyUsers == []: #happens when all users are happy
        return pick_random_app(bugsSolved, apps)
    thisUser = unhappyUsers[0] # thisUser is a list of apps
    limitedApps = {x:apps[x] for x in users[thisUser]} # Make a new "apps" that is only this user's apps
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
    limitedApps = {x:apps[x] for x in thisUser} # Make a new "apps" that is only this user's apps
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
    limitedApps = {x:apps[x] for x in thisUser} # Make a new "apps" that is only this user's apps
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
    else: return pick_specific_from_all_bugs() # occurs when all apps are solved

#@strategy TODO
def pick_from_specific_unsolved_app(bugsSolved: set, apps: dict) -> int:
    for app,bugs in apps.items():
        if bugs is not SOLVED:
            return random.choice(list(bugs))
    return pick_specific_from_all_bugs() # occurs when all apps are solved

# TODO: decorate as strategy; check for speed
def pick_nearest_done_app(bugsSolved, apps):
    """Picks a new bug not in the bugsSolved list by randomly selecting an unsolved bug from the app with the least bugs remaining.  We assume that there is at least one app and that there are no apps with zero unsolved bugs.
    """
    openBugCount = dict ( (len([y for y in apps[x] if y not in bugsSolved]), x) for x in apps if not apps[x] == True) # Create a dictionary of key: open bugs to value: app number, excluding ones that are solved
    if openBugCount: # in case all bugs are solved already
        bestApp = apps[openBugCount[min(openBugCount)]] # The one with the smallest open bugs is the best app
        return random.choice([x for x in bestApp if not x in bugsSolved])
    else: # openBugCount is empty, all apps are solved, so any bug will work fine:
        return pick_specific_from_all_bugs()

def prioritize(goals: dict, total_tasks: int):
    count = {task:0 for task in range(total_tasks)}
    for goal, tasks in goals.items():
        if tasks is not SOLVED:
            for task in tasks:
                count[task] += 1
    yield from (task for (task, frequency) in sorted(count.items(), key=itemgetter(1), reverse=True))

def bugs_by_frequency_in_features_generator():
    for bug in prioritize(goals=apps, total_tasks=numberOfBugs):
        while bug not in bugsSolved: yield bug

def apps_by_popularity_generator():
    for app in prioritize(goals=users, total_tasks=numberOfApps):
        while apps[app] is not SOLVED: yield app

# These are nonlocal instances of the generators in order to preserve their state
bugs_by_frequency_in_features = bugs_by_frequency_in_features_generator()
apps_by_popularity = apps_by_popularity_generator()

def pick_from_most_common_by_feature(): # TODO: label Specific?
    """Picks the bug that is the most common among all the unfinished features"""
    return next(bugs_by_frequency_in_features)

def pick_specific_from_most_popular_app():
    """Picks a specific bug from the most popular app"""
    app = next(apps_by_popularity)
    return list(apps[app])[0]

def pick_random_from_most_popular_app():
    """Picks a random bug from the most popular app"""
    app = next(apps_by_popularity)
    return random.choice(list(apps[app]))

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

def check_apps(apps: dict, bugsSolved: set) -> int:
    """Checks the applications dictionary for newly working applications"""
    solved = 0
    for app, bugs in apps.items():
        if bugs is not SOLVED:
            remaining_bugs = bugs - bugsSolved
            if remaining_bugs:
                apps[app] = remaining_bugs
                continue
            else:
                apps[app] = SOLVED
        solved += 1
    return solved

###
### Simulation setup
###

if useAlternativeAppMaker:
    apps = {x:alternative_make_app(bugProbability, random.randint(minAppBugs,maxAppBugs)) for x in range(numberOfApps)} #applications will have from minAppbugs to maxAppBugs, uniformly distributed
else:
    apps = {x:make_app(bugProbability, random.randint(minAppBugs,maxAppBugs)) for x in range(numberOfApps)} #applications will have from minAppbugs to maxAppBugs, uniformly distributed

users = {x:make_app(appProbability, random.randint(minUserApps,maxUserApps)) for x in range(numberOfUsers)} #Users will have from minUserApps to maxUserApps, uniformly distributed

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
working_app_days = 0
happy_user_days = 0
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
    if strategy == pick_from_specific_unsolved_app:
        bugToSolve = pick_from_specific_unsolved_app(bugsSolved, apps)
    if strategy == pick_nearest_done_app:
        bugToSolve = pick_nearest_done_app(bugsSolved, apps)
    if strategy == pick_random_bug:
        bugToSolve = pick_random_bug(bugsSolved, numberOfBugs)
    if strategy == pick_from_most_common_by_feature:
        bugToSolve = pick_from_most_common_by_feature()
    if strategy == pick_specific_from_all_bugs:
        bugToSolve = pick_specific_from_all_bugs()
    if strategy == pick_specific_from_most_popular_app:
        bugToSolve = pick_specific_from_most_popular_app()
    if strategy == pick_random_from_most_popular_app:
        bugToSolve = pick_random_from_most_popular_app()
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

    working_app_days += workingApps
    happy_user_days += happyUsers

    bugDifficulty[bugToSolve] -= 1
    if bugDifficulty[bugToSolve] <= 0:
        bugsSolved.add(bugToSolve)
        lastWorkedBug = False # this is the number of the bug we last worked on, in case we want to do it again

    if len(bugsSolved) == numberOfBugs:
        print("All bugs solved on day", day)
        append_to_log("%f, 1.0, 1.0, 1.0 \n" % (float(day)) )
        break

print("CPU time taken for simulation:", (time.clock() - timespent))
print("Apps Working * Days:", working_app_days, ", average", working_app_days/day, "per day.")
print("Happy Users * Days:", happy_user_days, ", average", happy_user_days/day, "per day.")

print("Now making chart.")

chart = pandas.DataFrame(chartData)
chart.plot()
plt.ylabel("Percentage")
plt.xlabel("Man-hours invested")
plt.savefig('wine-model-results.svg')
