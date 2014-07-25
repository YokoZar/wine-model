#!/usr/bin/env python3

###
### Wine-Model:
###  A python script to model Wine development created by Scott Ritchie
###
### Copyright (c) 2009-2014 Scott Ritchie <scottritchie@ubuntu.com>
### Licensed under the MIT License.  See the LICENSE file for details.
###
### Hosted on GitHub here: https://github.com/YokoZar/wine-model
###

import random
import time
from math import sqrt
from operator import itemgetter
import pandas
import matplotlib.pyplot as plt

DEBUG = False
LOGFILE = 'wine-model-results.csv'
ENABLE_LOG_DEFAULT = True
CHARTFILE = 'wine-model-results.svg' # You can rename to .png for a png file
CHART_BUGS = "Tasks Complete"
CHART_APPS = "Working Features"
CHART_USERS = "Happy Users"
CHART_LABEL_X = "Time Invested"
CHART_LABEL_Y = "Percentage"
CHART_TITLE = "Development Model"
RANDOM_SEED = 123456 # set to False to randomize every time
FINISH_TASKS_BEFORE_CHANGING_STRATEGY = True

###
### Basic setup
###

# TODO: make the following defined by arguments parser
numberOfBugs, numberOfApps, numberOfUsers = 10000, 2500, 5000
numberOfBugs, numberOfApps, numberOfUsers = 1000, 250, 500 # TODO: remove, temporary fast for dev mode
minAppBugs, maxAppBugs = 150, 900 # Applications pick from between these two numbers using a uniform distribution
# Set this to True if you want to ignore the above pre-set number of bugs and instead use the alternative App Maker (see bug probability below)
useAlternativeAppMaker = True
# Number of apps a user uses, not the number of users an app has
minUserApps, maxUserApps = 1, 10 
enable_log = ENABLE_LOG_DEFAULT
if RANDOM_SEED: 
    random.seed(a=RANDOM_SEED)

###
###
###

SOLVED = True

print("Modeling with", numberOfBugs, "bugs,", numberOfApps, "apps, and", numberOfUsers, "users")
if not useAlternativeAppMaker:
    print("From", minAppBugs, "to", maxAppBugs, "bugs per app and from", minUserApps, "to", maxUserApps, "apps per user")
else:
    print("Using relative probabilities for individual bugs and from", minUserApps, "to", maxUserApps, "apps per user")

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
def pick_strategy():
    """ Returns a strategy function based on the day.  This is meant to be modified by user."""
    #return random.choice(strategies)
    # Available strategies:
    # pick_specific_from_all_bugs pick_random_from_all_bugs
    # pick_specific_from_specific_app pick_random_from_specific_app
    # pick_specific_from_random_apps pick_random_from_random_apps TODO: name inconsistent
    # pick_specific_from_specific_user pick_random_from_specific_user
    # pick_specific_from_random_user pick_random_from_random_user
    # pick_specific_from_easiest_app pick_random_from_easiest_app
    # pick_specific_from_easiest_user pick_random_from_easiest_user
    # pick_specific_from_most_common_by_feature #TODO: pick_random_from_most_common_by_feature
    # pick_specific_from_most_popular_app pick_random_from_most_popular_app
    # pick_random_from_easiest_bugs pick_specific_from_easiest_bugs

    # You can select the strategy based on the day
    if day < 300: # eg do nothing but this strategy for the first 300 days
        return pick_specific_from_most_common_by_feature
    # "Realistic" model: rotate through different strategies
    if day %5 == 4: return pick_random_from_most_popular_app
    if day %5 == 3: return pick_random_from_random_user
    if day %5 == 2: return pick_random_from_easiest_app
    if day %5 == 1: return pick_random_from_easiest_bugs
    if day %5 == 0: return pick_random_from_easiest_user


### ----------------------------------------------------------------------------
###  You shouldn't need to modify anything below here to just run a simulation
### ----------------------------------------------------------------------------

###
### Setup functions
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

###
### Strategies
###

strategies = []
def strategy(function):
    strategies.append(function)
    return function

@strategy
def pick_specific_from_all_bugs():
    """Picks the smallest bug number not in the bugsSolved list"""
    return next(bugs_by_number)

@strategy
def pick_random_from_all_bugs():
    """Picks a random unsolved bug"""
    return next(random_bugs)

@strategy
def pick_specific_from_random_apps():
    """Picks the smallest bug from a random app"""
    try:
        app = next(random_apps)
    except StopIteration:
        return pick_specific_from_all_bugs()
    return min(apps[app])

@strategy
def pick_random_from_random_apps():
    """Picks a random bug from a random app"""
    try:
        app = next(random_apps)
    except StopIteration:
        return pick_random_from_all_bugs()
    return random.sample(apps[app],1)[0]

@strategy
def pick_specific_from_specific_app():
    try:
        app = next(apps_by_number)
        return min(apps[app])
    except StopIteration: 
        return pick_specific_from_all_bugs() 

@strategy
def pick_random_from_specific_app():
    try:
        app = next(apps_by_number)
        return random.sample(apps[app],1)[0]
    except StopIteration: 
        return pick_random_from_all_bugs() 

@strategy
def pick_specific_from_specific_user():
    """Picks the smallest bug from the smallest app from the smallest user"""
    try:
        user = next(users_by_number)
    except StopIteration:
        return pick_specific_from_specific_app()
    app = min(users[user])
    return min(apps[app])

@strategy
def pick_random_from_specific_user():
    """Picks a random bug from a random app from the smallest user"""
    try:
        user = next(users_by_number)
    except StopIteration:
        return pick_random_from_random_apps()
    app = random.sample(users[user],1)[0]
    return random.sample(apps[app],1)[0]

@strategy
def pick_specific_from_random_user():
    """Picks the smallest bug in the smallest app from a random user"""
    try:
        user = next(random_users)
    except StopIteration:
        return pick_specific_from_random_app()
    app = min(users[user])
    return min(apps[app])

@strategy
def pick_random_from_random_user():
    """Picks a random bug from a random app from a random user"""
    try:
        user = next(random_users)
    except StopIteration:
        return pick_random_from_random_app()
    app = random.choice(tuple(users[user]))
    return random.choice(tuple(apps[app]))

@strategy
def pick_specific_from_easiest_app():
    try:
        app = next(apps_by_easiest)
    except StopIteration: 
        return pick_specific_from_all_bugs() 
    return min(apps[app])

@strategy
def pick_random_from_easiest_app():
    try:
        app = next(apps_by_easiest)
    except StopIteration: 
        return pick_random_from_all_bugs() 
    return random.sample(apps[app],1)[0]

@strategy
def pick_specific_from_easiest_user():
    try:
        user = next(users_by_easiest)
    except StopIteration: 
        return pick_specific_from_easiest_app() 
    app = min(users[user])
    return min(apps[app])

@strategy
def pick_random_from_easiest_user():
    try:
        user = next(users_by_easiest)
    except StopIteration: 
        return pick_random_from_easiest_app()
    app = random.sample(users[user],1)[0]
    return random.sample(apps[app],1)[0] 

@strategy
def pick_specific_from_most_common_by_feature():
    """Picks the bug that is the most common among all the unfinished features"""
    return next(bugs_by_popularity_in_apps)

@strategy
def pick_specific_from_most_popular_app():
    """Picks a specific bug from the most popular app"""
    app = next(apps_by_popularity_in_users)
    return list(apps[app])[0]

@strategy
def pick_random_from_most_popular_app():
    """Picks a random bug from the most popular app"""
    app = next(apps_by_popularity_in_users)
    return random.sample(apps[app],1)[0]

@strategy
def pick_random_from_easiest_bugs():
    easiest_difficulty = None
    for bug, difficulty in bugDifficulty.items():
        if 0 < difficulty and (easiest_difficulty == None or difficulty < easiest_difficulty):
            candidates = {bug}
            easiest_difficulty = difficulty
        elif 0 < difficulty == easiest_difficulty:
            candidates.add(bug)
    return random.sample(candidates,1)[0]

@strategy
def pick_specific_from_easiest_bugs():
    easiest_difficulty = None
    for bug, difficulty in bugDifficulty.items():
        if 0 < difficulty <= 1: # Doesn't get any easier
            return bug
        if difficulty > 1 and (easiest_difficulty is None or difficulty < easiest_difficulty):
            easiest_difficulty = difficulty
            easiest_bug = bug
    return easiest_bug

###
### Helper functions
###

def append_to_log(entry):
    if enable_log:
        with open(LOGFILE, 'a') as logfile:
            logfile.write(entry)

def check_done(goals: dict, solved_tasks: set) -> int:
    """Checks a dictionary (eg users, apps) for solved things (eg apps, bugs) and marks them"""
    solved = 0
    for goal, tasks in goals.items():
        if tasks is not SOLVED:
            remaining_tasks = tasks - solved_tasks
            if remaining_tasks:
                goals[goal] = remaining_tasks
                continue
            else:
                goals[goal] = SOLVED
        solved += 1
    return solved

###
### Simulation setup
###

if enable_log:
    with open(LOGFILE, 'w'): pass
append_to_log("Bugs %i Apps %i Users %i Min App Bugs %i Max App Bugs %i Min User Apps %i Max User Apps %i \n" % 
            (numberOfBugs, numberOfApps, numberOfUsers, minAppBugs, maxAppBugs, minUserApps, maxUserApps) )

# TODO: compress
if useAlternativeAppMaker:
    apps = {x:alternative_make_app(bugProbability, random.randint(minAppBugs,maxAppBugs)) for x in range(numberOfApps)} #applications will have from minAppbugs to maxAppBugs, uniformly distributed
else:
    apps = {x:make_app(bugProbability, random.randint(minAppBugs,maxAppBugs)) for x in range(numberOfApps)} #applications will have from minAppbugs to maxAppBugs, uniformly distributed

users = {x:make_app(appProbability, random.randint(minUserApps,maxUserApps)) for x in range(numberOfUsers)} #Users will have from minUserApps to maxUserApps, uniformly distributed

averageBugsPerApp = sum([len(apps[x]) for x in apps]) / numberOfApps
averageAppsPerUser = sum([len(users[x]) for x in users]) / numberOfUsers
print("Applications and users generated, averaging", averageBugsPerApp ,"bugs per app and", averageAppsPerUser ,"apps per user.  Starting simulation...")

###
### Generators, helper functions, and state variables for strategies
###

def prioritize(goals: dict, total_tasks: int):
    """Generator to yield tasks within a dict of goals based on their frequency"""
    count = {task:0 for task in range(total_tasks)}
    for goal, tasks in goals.items():
        if tasks is not SOLVED:
            for task in tasks:
                count[task] += 1
    yield from (task for (task, frequency) in sorted(count.items(), key=itemgetter(1), reverse=True))

def goals_by_number_generator(goals: dict):
    for goal in goals:
        while goals[goal] is not SOLVED: yield goal

def goals_by_easiest_generator(goals: dict):
    """Generator to yield goals based on which has the fewest tasks remaining"""
    goalsizes = [(goal, len(tasks)) for goal, tasks in goals.items() if tasks is not SOLVED]
    while goalsizes:
        yield min(goalsizes, key=itemgetter(1))[0]
        goalsizes = [(goal, len(tasks)) for goal, tasks in goals.items() if tasks is not SOLVED]

def goals_by_random_generator(goals: dict):
    """Generator to yield unsolved goals at random"""
    unfinished_goals = set(goals)
    while unfinished_goals:
        goal = random.choice(tuple(unfinished_goals))
        if goals[goal] is SOLVED:
            unfinished_goals.remove(goal)
        else:
            yield goal

def bugs_by_popularity_in_apps_generator():
    for bug in prioritize(goals=apps, total_tasks=numberOfBugs):
        while bug not in bugsSolved: yield bug

def apps_by_popularity_in_users_generator():
    for app in prioritize(goals=users, total_tasks=numberOfApps):
        while apps[app] is not SOLVED: yield app

def bugs_by_number_generator():
    for bug in range(numberOfBugs):
        while bug not in bugsSolved: yield bug

def random_bugs_generator():
    open_bugs = set(range(numberOfBugs)) - bugsSolved
    while True:
        open_bugs -= bugsSolved
        yield random.sample(open_bugs,1)[0]

# These are nonlocal instances of the generators in order to preserve their state
bugs_by_popularity_in_apps = bugs_by_popularity_in_apps_generator()
bugs_by_number = bugs_by_number_generator()
apps_by_popularity_in_users = apps_by_popularity_in_users_generator()
apps_by_number = goals_by_number_generator(apps)
apps_by_easiest = goals_by_easiest_generator(apps)
users_by_number = goals_by_number_generator(users)
users_by_easiest = goals_by_easiest_generator(users)
random_bugs = random_bugs_generator()
random_apps = goals_by_random_generator(apps)
random_users = goals_by_random_generator(users)

###
### Simulation begins here
###

bugsSolved = set([]) # an (ordered?) list of integers of all bugs solved so far
day = 0

timespent = time.clock()
working_app_days = 0
happy_user_days = 0
bug_in_progress = None
reported_first_app, reported_first_user = False, False
reported_all_apps, reported_all_users = False, False

append_to_log("Time, % Work Items Completed, % Features Completed, % Happy Users \n")
chartData = {CHART_BUGS: [], CHART_APPS : [], CHART_USERS : []}

# TODO: make this an int to avoid 0.89 from float weirdness
progressIndicator = 0.10 # When to first show 'working on day' (x) progress indicators

while(True): 
    # Check for newly working apps every day we solved a bug in the previous day
    if bug_in_progress is None:
        workingApps = check_done(apps,bugsSolved)
        happyUsers = check_done(users,set(x for x in apps if apps[x] is SOLVED))
        # TODO: refactor above to maybe not reconstruct solved apps every time

    if not reported_first_app and workingApps >= 1:
        print("First app working on day", day)
        reported_first_app = True
    if not reported_all_apps and workingApps == numberOfApps:
        print("All apps working on day", day)
        reported_all_apps = True
    if not reported_first_user and happyUsers >= 1:
        print("First user happy on day", day)
        reported_first_user = True
    if not reported_all_users and happyUsers == numberOfUsers:
        print("All users happy on day", day)
        reported_all_users = True

    if day >= totalTimeToSolve*progressIndicator:
        print("%i%% complete on day: " % (progressIndicator*100), day)
        progressIndicator += 0.10

    append_to_log("%f, %f, %f, %f \n" % (float(day), len(bugsSolved)/numberOfBugs, workingApps/numberOfApps, happyUsers/numberOfUsers) )
    chartData[CHART_BUGS].append(len(bugsSolved)*100/numberOfBugs)
    chartData[CHART_APPS].append(workingApps*100/numberOfApps)
    chartData[CHART_USERS].append(happyUsers*100/numberOfUsers)

    if bug_in_progress is None or not FINISH_TASKS_BEFORE_CHANGING_STRATEGY:
        bug_in_progress = pick_strategy()()
        if DEBUG and bug_in_progress in bugsSolved:
            error = "Working already complete task: " + str(bug_in_progress) + " on day " + str(day)
            raise ValueError(error)

    day += 1 
    working_app_days += workingApps
    happy_user_days += happyUsers

    bugDifficulty[bug_in_progress] -= 1
    if DEBUG: print("worked bug:", bug_in_progress)
    if bugDifficulty[bug_in_progress] <= 0:
        bugsSolved.add(bug_in_progress)
        if DEBUG: print("solved bug:", bug_in_progress)
        bug_in_progress = None

    if len(bugsSolved) == numberOfBugs:
        print("All bugs solved on day", day)
        append_to_log("%f, 1.0, 1.0, 1.0 \n" % (float(day)) )
        break

# TODO: make optional command line output
if DEBUG: print("Available strategies:", " ".join(f.__name__ for f in strategies))

print("Time spent running simulation:", (time.clock() - timespent))
print("Average features working:", working_app_days/day)
print("Average happy users:", happy_user_days/day)

print("Now making chart.")

chart = pandas.DataFrame(chartData)
chart.plot()
plt.title(CHART_TITLE)
plt.ylabel(CHART_LABEL_Y)
plt.xlabel(CHART_LABEL_X)
plt.savefig(CHARTFILE)
