#!/usr/bin/env python3
#
# Wine-Model:
#  A python script to model Wine development created by Scott Ritchie
#
# Copyright (c) 2009-2014 Scott Ritchie <scottritchie@ubuntu.com>
# Licensed under the MIT License.  See the LICENSE file for details.
#
# Hosted on GitHub here: https://github.com/YokoZar/wine-model
#

import random
import time
from math import sqrt
from operator import itemgetter
from functools import partial
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
RANDOM_SEED = False # Set to a constant to directly compare strategies from one run to the next
FINISH_TASKS_BEFORE_CHANGING_STRATEGY = True

MIN_APPS_PER_USER = 1
MAX_APPS_PER_USER = 10

###
### Basic setup -- meant to be modified by user
###

# Note that internally "features" == "apps" and "work items" == "bugs"
number_of_bugs, number_of_apps, number_of_users = 10000, 2500, 5000

def setup_functions():
    global bug_difficulty_function, bug_probability_function 
    global app_frequency_function, apps_per_user_function
    bug_difficulty_function = lambda: abs(int(random.normalvariate(4,3))) + 1
    bug_probability_function = partial(probability_list_from_zipfs_law, number_of_bugs)
    app_frequency_function = partial(frequency_list_from_pareto_distribution, number_of_apps)
    apps_per_user_function = partial(random.randint, MIN_APPS_PER_USER, MAX_APPS_PER_USER)

enable_log = ENABLE_LOG_DEFAULT
if RANDOM_SEED: 
    random.seed(a=RANDOM_SEED)

###
### Strategy -- meant to be modified by user
###

def pick_strategy():
    """ Returns a strategy function based on the day.  This is meant to be modified by user."""
    #return random.choice(strategies)
    # Available strategies:
    # pick_specific_from_all_bugs pick_random_from_all_bugs
    # pick_specific_from_specific_app pick_random_from_specific_app
    # pick_specific_from_random_app pick_random_from_random_app
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
### App and User Setup
###

def probability_list_from_zipfs_law(size: int):
    """Returns a list of floats from 0 to 1 based on a Zipfian distribution"""
    item_probability = [1.0/sqrt(x+1) for x in range(size)]
    random.shuffle(item_probability) # Prevent "smallest number" from implying "more likely"
    return item_probability

def set_from_fixed_probabilities(probability: list):
    """Returns a set of numbers by randomly testing to include each one based on probability."""
    return {item for (item, chance) in enumerate(probability) if random.uniform(0,1) <= chance}

def frequency_list_from_pareto_distribution(size: int):
    """Returns a set of relative probabilities based on a pareto distribution"""
    return [random.paretovariate(2.2) for x in range(size)]

def set_from_relative_frequencies(frequency: list, quantity: int, mutate_list=False):
    """Returns a set of quantity numbers based on the frequency list. An item of frequency 2 is
    twice as likely to appear as an item of frequency 1, 4 is 4 times as likely, and so on.
    """
    assert quantity <= len(frequency)
    if quantity == 0:
        return set()
    if not mutate_list:
        frequency = frequency.copy() 
    length_of_ruler = sum(frequency)
    point_on_line = random.uniform(0,length_of_ruler)
    for (index, length_of_segment) in enumerate(frequency):
        point_on_line -= length_of_segment
        if point_on_line < 0:
            frequency[index] = 0
            return {index} | set_from_relative_frequencies(frequency, quantity - 1, True)
    assert False

###
### Strategies
###

SOLVED = True

strategies = []
def strategy(function):
    strategies.append(function)
    return function

@strategy
def pick_specific_from_all_bugs():
    """Picks the smallest bug number not in the solved_bugs list"""
    return next(bugs_by_number)

@strategy
def pick_random_from_all_bugs():
    """Picks a random unsolved bug"""
    return next(random_bugs)

@strategy
def pick_specific_from_random_app():
    """Picks the smallest bug from a random app"""
    for app in random_apps:
        return min(apps[app])
    return pick_specific_from_all_bugs()

@strategy
def pick_random_from_random_app():
    """Picks a random bug from a random app"""
    for app in random_apps:
        return random.sample(apps[app],1)[0]
    return pick_random_from_all_bugs()

@strategy
def pick_specific_from_specific_app():
    for app in apps_by_number:
        return min(apps[app])
    return pick_specific_from_all_bugs() 

@strategy
def pick_random_from_specific_app():
    for app in apps_by_number:
        return random.sample(apps[app],1)[0]
    return pick_random_from_all_bugs() 

@strategy
def pick_specific_from_specific_user():
    """Picks the smallest bug from the smallest app from the smallest user"""
    for user in users_by_number:
        app = min(users[user])
        return min(apps[app])
    return pick_specific_from_specific_app()

@strategy
def pick_random_from_specific_user():
    """Picks a random bug from a random app from the smallest user"""
    for user in users_by_number:
        app = random.sample(users[user],1)[0]
        return random.sample(apps[app],1)[0]
    return pick_random_from_random_app()

@strategy
def pick_specific_from_random_user():
    """Picks the smallest bug in the smallest app from a random user"""
    for user in random_users:
        app = min(users[user])
        return min(apps[app])
    return pick_specific_from_random_app()

@strategy
def pick_random_from_random_user():
    """Picks a random bug from a random app from a random user"""
    for user in random_users:
        app = random.choice(tuple(users[user]))
        return random.choice(tuple(apps[app]))
    return pick_random_from_random_app()

@strategy
def pick_specific_from_easiest_app():
    for app in apps_by_easiest:
        return min(apps[app])
    return pick_specific_from_all_bugs() 

@strategy
def pick_random_from_easiest_app():
    for app in apps_by_easiest:
        return random.sample(apps[app],1)[0]
    return pick_random_from_all_bugs() 

@strategy
def pick_specific_from_easiest_user():
    for user in users_by_easiest:
        app = min(users[user])
        return min(apps[app])
    return pick_specific_from_easiest_app() 

@strategy
def pick_random_from_easiest_user():
    for user in users_by_easiest:
        app = random.sample(users[user],1)[0]
        return random.sample(apps[app],1)[0]
    return pick_random_from_easiest_app()

@strategy
def pick_specific_from_most_common_by_feature():
    """Picks the bug that is the most common among all the unfinished features"""
    return next(bugs_by_popularity_in_apps)

@strategy
def pick_specific_from_most_popular_app():
    """Picks a specific bug from the most popular app"""
    for app in (apps_by_popularity_in_users):
        return list(apps[app])[0]
    return pick_specific_from_all_bugs()

@strategy
def pick_random_from_most_popular_app():
    """Picks a random bug from the most popular app"""
    for app in (apps_by_popularity_in_users):
        return random.sample(apps[app],1)[0]
    return pick_random_from_all_bugs()

@strategy
def pick_random_from_easiest_bugs():
    easiest_difficulty = None
    for bug, difficulty in bug_difficulty.items():
        if 0 < difficulty and (easiest_difficulty == None or difficulty < easiest_difficulty):
            candidates = {bug}
            easiest_difficulty = difficulty
        elif 0 < difficulty == easiest_difficulty:
            candidates.add(bug)
    return random.sample(candidates,1)[0]

@strategy
def pick_specific_from_easiest_bugs():
    easiest_difficulty = None
    for bug, difficulty in bug_difficulty.items():
        if 0 < difficulty <= 1: # Doesn't get any easier
            return bug
        if difficulty > 1 and (easiest_difficulty is None or difficulty < easiest_difficulty):
            easiest_difficulty = difficulty
            easiest_bug = bug
    return easiest_bug

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
    for bug in prioritize(goals=apps, total_tasks=number_of_bugs):
        while bug not in solved_bugs: yield bug

def apps_by_popularity_in_users_generator():
    for app in prioritize(goals=users, total_tasks=number_of_apps):
        while apps[app] is not SOLVED: yield app

def bugs_by_number_generator():
    for bug in range(number_of_bugs):
        while bug not in solved_bugs: yield bug

def random_bugs_generator():
    open_bugs = set(range(number_of_bugs)) - solved_bugs
    while True:
        open_bugs -= solved_bugs
        yield random.sample(open_bugs,1)[0]

###
### Helper functions for running simulation
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

def setup():
    """Creates apps and users and erases the log"""
    global apps, users, bug_difficulty
    if enable_log:
        with open(LOGFILE, 'w'): pass

    setup_functions()
    bug_difficulty = {bug: bug_difficulty_function() for bug in range(number_of_bugs)}

    bug_probability = bug_probability_function()
    apps = {app:set_from_fixed_probabilities(bug_probability) for app in range(number_of_apps)}
    average_bugs_per_app = sum([len(apps[x]) for x in apps]) / number_of_apps
    print("Features generated, averaging", average_bugs_per_app, "items per feature.")

    app_frequency = app_frequency_function()
    users = {user:set_from_relative_frequencies(app_frequency, apps_per_user_function())
             for user in range(number_of_users)}
    average_apps_per_user = sum([len(users[x]) for x in users]) / number_of_users
    print("Users generated, averaging", average_apps_per_user, "features per user.")

###
### Simulation begins here
###

print("Modeling with", number_of_bugs, "bugs,", number_of_apps, "apps, and", number_of_users, "users")
setup()

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

total_time_to_solve = sum(bug_difficulty.values())
print(total_time_to_solve, "total time units to finish every work item.")

solved_bugs = set()
day = 0

timespent = time.clock()
working_app_days = 0
happy_user_days = 0
bug_in_progress = None
reported_first_app, reported_first_user = False, False
reported_all_apps, reported_all_users = False, False

append_to_log("Time, % Work Items Completed, % Features Completed, % Happy Users \n")
chart_data = {CHART_BUGS: [], CHART_APPS : [], CHART_USERS : []}

show_at_percent_done = 10 # When to first show 'working on day' (x) progress indicators

while(True): 
    # Check for newly working apps every day we solved a bug in the previous day
    if bug_in_progress is None:
        working_apps = check_done(apps,solved_bugs)
        happy_users = check_done(users,set(x for x in apps if apps[x] is SOLVED))
        # TODO: refactor above to maybe not reconstruct solved apps every time

    if not reported_first_app and working_apps >= 1:
        print("First feature working at time", day)
        reported_first_app = True
    if not reported_all_apps and working_apps == number_of_apps:
        print("All features working at time", day)
        reported_all_apps = True
    if not reported_first_user and happy_users >= 1:
        print("First user happy at time", day)
        reported_first_user = True
    if not reported_all_users and happy_users == number_of_users:
        print("All users happy at time", day)
        reported_all_users = True

    if day >= total_time_to_solve*(show_at_percent_done/100):
        print("%i%% complete at time: " % (show_at_percent_done), day)
        show_at_percent_done += 10

    append_to_log("%f, %f, %f, %f \n" % (float(day), len(solved_bugs)/number_of_bugs, working_apps/number_of_apps, happy_users/number_of_users) )
    chart_data[CHART_BUGS].append(len(solved_bugs)*100/number_of_bugs)
    chart_data[CHART_APPS].append(working_apps*100/number_of_apps)
    chart_data[CHART_USERS].append(happy_users*100/number_of_users)

    if bug_in_progress is None or not FINISH_TASKS_BEFORE_CHANGING_STRATEGY:
        bug_in_progress = pick_strategy()()
        if DEBUG and bug_in_progress in solved_bugs:
            error = "Working already complete task: " + str(bug_in_progress) + " on day " + str(day)
            raise ValueError(error)

    day += 1 
    working_app_days += working_apps
    happy_user_days += happy_users

    bug_difficulty[bug_in_progress] -= 1
    if DEBUG: print("worked bug:", bug_in_progress)
    if bug_difficulty[bug_in_progress] <= 0:
        solved_bugs.add(bug_in_progress)
        if DEBUG: print("solved bug:", bug_in_progress)
        bug_in_progress = None

    if len(solved_bugs) == number_of_bugs:
        print("100% complete at time: ", day)
        append_to_log("%f, 1.0, 1.0, 1.0 \n" % (float(day)) )
        break

# TODO: make optional command line output
if DEBUG: print("Available strategies:", " ".join(f.__name__ for f in strategies))

print("Time spent running simulation:", (time.clock() - timespent))
print("Average features working:", working_app_days/day)
print("Average happy users:", happy_user_days/day)

print("Now making chart.")

chart = pandas.DataFrame(chart_data)
chart.plot()
plt.title(CHART_TITLE)
plt.ylabel(CHART_LABEL_Y)
plt.xlabel(CHART_LABEL_X)
plt.savefig(CHARTFILE)
