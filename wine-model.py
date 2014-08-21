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
CHART_TITLE = "Comparing Development Models"
CHART_TASKS_COMPLETE = False # This is often not helpful when comparing
RANDOM_SEED = False # Set to a constant to directly compare strategies from one run to the next
FINISH_TASKS_BEFORE_CHANGING_STRATEGY = True

PROJECT_NAMES = ["Most popular feature", "Easiest feature", "Satisfy arbitrary user"]

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

# Available pick methods:
# pick_specific_from_all_bugs pick_random_from_all_bugs
# pick_specific_from_specific_app pick_random_from_specific_app
# pick_specific_from_random_app pick_random_from_random_app
# pick_specific_from_specific_user pick_random_from_specific_user
# pick_specific_from_random_user pick_random_from_random_user
# pick_specific_from_easiest_app pick_random_from_easiest_app
# pick_specific_from_easiest_user pick_random_from_easiest_user
# pick_specific_from_most_common_by_feature #TODO: pick_random_from_most_common_by_feature
# pick_specific_from_most_popular_app pick_random_from_most_popular_app
# pick_specific_from_easiest_bugs pick_random_from_easiest_bugs


def strategy_chooser(name):
    """Returns a function that returns a pick method based on the current state"""
    if name == "Rotate reasonably": return rotate_strategy
    if name == "Easiest task": return lambda: pick_specific_from_easiest_bugs
    if name == "Easiest feature": return lambda: pick_specific_from_easiest_app
    if name == "Most popular feature": return lambda: pick_specific_from_most_popular_app
    if name == "Satisfy arbitrary user": return lambda: pick_specific_from_specific_user
    return default_strategy

def rotate_strategy():
    """Returns a pick method based on the day"""
    # You can select the strategy based on the day
    if day < 300: # eg do nothing but this strategy for the first 300 days
        return pick_specific_from_most_common_by_feature
    # "Realistic" model: rotate through different reasonable pick methods
    if day %5 == 4: return pick_random_from_most_popular_app
    if day %5 == 3: return pick_random_from_random_user
    if day %5 == 2: return pick_random_from_easiest_app
    if day %5 == 1: return pick_random_from_easiest_bugs
    if day %5 == 0: return pick_random_from_easiest_user

def default_strategy():
    return random.choice(pick_methods)

### ----------------------------------------------------------------------------
###  You shouldn't need to modify anything below here to just run a simulation
### ----------------------------------------------------------------------------

DONE = True

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
### Pick Methods
###

pick_methods = []
def pick_method(function):
    pick_methods.append(function)
    return function

@pick_method
def pick_specific_from_all_bugs(project):
    """Picks the smallest bug number not in the solved_bugs list"""
    return next(project.bugs_by_number)

@pick_method
def pick_random_from_all_bugs(project):
    """Picks a random unsolved bug"""
    return next(project.random_bugs)

@pick_method
def pick_specific_from_random_app(project):
    """Picks the smallest bug from a random app"""
    for app in project.random_apps:
        return min(project.apps[app])
    return pick_specific_from_all_bugs(project)

@pick_method
def pick_random_from_random_app(project):
    """Picks a random bug from a random app"""
    for app in project.random_apps:
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_all_bugs(project)

@pick_method
def pick_specific_from_specific_app(project):
    for app in project.apps_by_number:
        return min(project.apps[app])
    return pick_specific_from_all_bugs(project) 

@pick_method
def pick_random_from_specific_app(project):
    for app in project.apps_by_number:
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_all_bugs(project) 

@pick_method
def pick_specific_from_specific_user(project):
    """Picks the smallest bug from the smallest app from the smallest user"""
    for user in project.users_by_number:
        app = min(project.users[user])
        return min(project.apps[app])
    return pick_specific_from_specific_app(project)

@pick_method
def pick_random_from_specific_user(project):
    """Picks a random bug from a random app from the smallest user"""
    for user in project.users_by_number:
        app = random.choice(tuple(project.users[user]))
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_random_app(project)

@pick_method
def pick_specific_from_random_user(project):
    """Picks the smallest bug in the smallest app from a random user"""
    for user in project.random_users:
        app = min(project.users[user])
        return min(project.apps[app])
    return pick_specific_from_random_app(project)

@pick_method
def pick_random_from_random_user(project):
    """Picks a random bug from a random app from a random user"""
    for user in project.random_users:
        app = random.choice(tuple(project.users[user]))
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_random_app(project)

@pick_method
def pick_specific_from_easiest_app(project):
    for app in project.apps_by_easiest:
        return min(project.apps[app])
    return pick_specific_from_all_bugs(project) 

@pick_method
def pick_random_from_easiest_app(project):
    for app in project.apps_by_easiest:
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_all_bugs(project)

@pick_method
def pick_specific_from_easiest_user(project):
    for user in project.users_by_easiest:
        app = min(project.users[user])
        return min(project.apps[app])
    return pick_specific_from_easiest_app(project) 

@pick_method
def pick_random_from_easiest_user(project):
    for user in project.users_by_easiest:
        app = random.choice(tuple(project.users[user]))
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_easiest_app(project)

@pick_method
def pick_specific_from_most_common_by_feature(project):
    """Picks the bug that is the most common among all the unfinished features"""
    return next(project.bugs_by_popularity_in_apps)

@pick_method
def pick_specific_from_most_popular_app(project):
    """Picks a specific bug from the most popular app"""
    for app in (project.apps_by_popularity_in_users):
        return list(project.apps[app])[0]
    return pick_specific_from_all_bugs(project)

@pick_method
def pick_random_from_most_popular_app(project):
    """Picks a random bug from the most popular app"""
    for app in (project.apps_by_popularity_in_users):
        return random.choice(tuple(project.apps[app]))
    return pick_random_from_all_bugs(project)

@pick_method
def pick_random_from_easiest_bugs(project):
    easiest_difficulty = None
    for bug, difficulty in project.bug_difficulty.items():
        if 0 < difficulty and (easiest_difficulty == None or difficulty < easiest_difficulty):
            candidates = {bug}
            easiest_difficulty = difficulty
        elif 0 < difficulty == easiest_difficulty:
            candidates.add(bug)
    return random.choice(tuple(candidates))

@pick_method
def pick_specific_from_easiest_bugs(project):
    easiest_difficulty = None
    for bug, difficulty in project.bug_difficulty.items():
        if 0 < difficulty <= 1: # Doesn't get any easier
            return bug
        if difficulty > 1 and (easiest_difficulty is None or difficulty < easiest_difficulty):
            easiest_difficulty = difficulty
            easiest_bug = bug
    return easiest_bug

###
### Project class
###

class Project:
    def __init__(self, users, apps, bug_difficulty, name):
        self.users = users
        self.apps = apps
        self.bug_difficulty = bug_difficulty
        self.solved_bugs = set()
        self.name = name
        self.method_selector = strategy_chooser(name)

        # Class-wide generators to preserve state
        self.bugs_by_number = bugs_by_number_generator(self.solved_bugs)
        self.random_bugs = random_bugs_generator(self.solved_bugs)
        self.apps_by_number = goals_by_number_generator(self.apps)
        self.random_apps = goals_by_random_generator(self.apps)
        self.users_by_number = goals_by_number_generator(self.users)
        self.random_users = goals_by_random_generator(self.users)
        self.bugs_by_popularity_in_apps = bugs_by_popularity_in_apps_generator(self.apps, self.solved_bugs)
        self.apps_by_popularity_in_users = apps_by_popularity_in_users_generator(self.users, self.apps)
        self.apps_by_easiest = goals_by_easiest_generator(self.apps)
        self.users_by_easiest = goals_by_easiest_generator(self.users)
        
        self.working_app_days = 0
        self.happy_user_days = 0
        self.bug_in_progress = None
        self.reported_first_app, self.reported_first_user = False, False
        self.reported_all_apps, self.reported_all_users = False, False

    def make_log_item(self):
        log = str(self.name) + ", "
        log += str(day) + ", "
        log += str(len(self.solved_bugs)/number_of_bugs) + ", "
        log += str(self.working_apps/number_of_apps) + ", "
        log += str(self.happy_users/number_of_users) + "\n"
        return log

    def choose_bug(self):
        """Finds a bug to work on and sets bug_in_progress to it"""
        if self.bug_in_progress is not None and FINISH_TASKS_BEFORE_CHANGING_STRATEGY:
            return
        else:
            pick_method = self.method_selector()
            self.bug_in_progress = pick_method(self)
            assert self.bug_in_progress not in self.solved_bugs

    def work_bug(self):
        """Works on the bug_in_progress"""
        self.working_app_days += self.working_apps
        self.happy_user_days += self.happy_users

        self.bug_difficulty[self.bug_in_progress] -= 1
        if DEBUG: print("worked bug:", self.bug_in_progress)
        if self.bug_difficulty[self.bug_in_progress] <= 0:
            self.solved_bugs.add(self.bug_in_progress)
            if DEBUG: print("solved bug:", self.bug_in_progress)
            self.bug_in_progress = None


###
### Generators and helper functions for pick methods
###

def prioritize(goals: dict, total_tasks: int):
    """Generator to yield tasks within a dict of goals based on their frequency"""
    count = {task:0 for task in range(total_tasks)}
    for goal, tasks in goals.items():
        if tasks is not DONE:
            for task in tasks:
                count[task] += 1
    yield from (task for (task, frequency) in sorted(count.items(), key=itemgetter(1), reverse=True))

def goals_by_number_generator(goals: dict):
    for goal in goals:
        while goals[goal] is not DONE: yield goal

def goals_by_easiest_generator(goals: dict):
    """Generator to yield goals based on which has the fewest tasks remaining"""
    goalsizes = [(goal, len(tasks)) for goal, tasks in goals.items() if tasks is not DONE]
    while goalsizes:
        yield min(goalsizes, key=itemgetter(1))[0]
        goalsizes = [(goal, len(tasks)) for goal, tasks in goals.items() if tasks is not DONE]

def goals_by_random_generator(goals: dict):
    """Generator to yield unsolved goals at random"""
    unfinished_goals = set(goals)
    while unfinished_goals:
        goal = random.choice(tuple(unfinished_goals))
        if goals[goal] is DONE:
            unfinished_goals.remove(goal)
        else:
            yield goal

def bugs_by_popularity_in_apps_generator(apps: dict, solved_bugs: set):
    for bug in prioritize(goals=apps, total_tasks=number_of_bugs):
        while bug not in solved_bugs: yield bug

def apps_by_popularity_in_users_generator(users: dict, apps: dict):
    for app in prioritize(goals=users, total_tasks=number_of_apps):
        while apps[app] is not DONE: yield app

def bugs_by_number_generator(solved_bugs: set):
    for bug in range(number_of_bugs):
        while bug not in solved_bugs: yield bug

def random_bugs_generator(solved_bugs: set):
    open_bugs = set(range(number_of_bugs))
    while True:
        open_bugs -= solved_bugs
        yield random.choice(tuple(open_bugs))

###
### Helper functions for running simulation
###

def append_to_log(entry: str):
    if enable_log:
        with open(LOGFILE, 'a') as logfile:
            logfile.write(entry)

def check_done(goals: dict, solved_tasks: set) -> int:
    """Checks a dictionary (eg users, apps) for solved things (eg apps, bugs) and marks them"""
    solved = 0
    for goal, tasks in goals.items():
        if tasks is not DONE:
            remaining_tasks = tasks - solved_tasks
            if remaining_tasks:
                goals[goal] = remaining_tasks
                continue
            else:
                goals[goal] = DONE
        solved += 1
    return solved

def setup():
    """Creates apps and users and erases the log"""
    global projects
    global total_time_to_solve
    if enable_log:
        with open(LOGFILE, 'w'): pass

    setup_functions()

    bug_difficulty = {bug: bug_difficulty_function() for bug in range(number_of_bugs)}
    total_time_to_solve = sum(bug_difficulty.values())
    print("Work items generated, with", total_time_to_solve, "total time to finish every work item.")

    bug_probability = bug_probability_function()
    apps = {app:set_from_fixed_probabilities(bug_probability) for app in range(number_of_apps)}
    average_bugs_per_app = sum([len(apps[x]) for x in apps]) / number_of_apps
    print("Features generated, averaging", average_bugs_per_app, "items per feature.")

    app_frequency = app_frequency_function()
    users = {user:set_from_relative_frequencies(app_frequency, apps_per_user_function())
             for user in range(number_of_users)}
    average_apps_per_user = sum([len(users[x]) for x in users]) / number_of_users
    print("Users generated, averaging", average_apps_per_user, "features per user.")

    assert PROJECT_NAMES
    projects = [Project(users.copy(), apps.copy(), bug_difficulty.copy(), name) for name in PROJECT_NAMES]

###
### Simulation begins here
###

print("Modeling with", number_of_bugs, "bugs,", number_of_apps, "apps, and", number_of_users, "users")
setup()

day = 0

timespent = time.clock()

append_to_log("Strategy, Time, % Work Items Completed, % Features Completed, % Happy Users \n")
chart_data = {}
for project in projects:
    name = project.name
    chart_data.update({name+": "+CHART_BUGS: [], name+": "+CHART_APPS: [], name+": "+CHART_USERS: []})

show_at_percent_done = 10 # When to first show 'working on day' (x) progress indicators
bugs_remaining = True

while(bugs_remaining): # TODO: just use the inner for loop to cycle over projects
    for project in projects:
        # Check for newly working apps every day we solved a bug in the previous day
        if project.bug_in_progress is None:
            project.working_apps = check_done(project.apps,project.solved_bugs)
            finished_apps = set(x for x in project.apps if project.apps[x] is DONE)
            project.happy_users = check_done(project.users,finished_apps)
            # TODO: refactor above to maybe not reconstruct finished apps every time

        if not project.reported_first_app and project.working_apps >= 1:
            print("First feature working for", project.name, "at time", day)
            project.reported_first_app = True
        if not project.reported_all_apps and project.working_apps == number_of_apps:
            print("All features working for", project.name, " at time", day)
            project.reported_all_apps = True
        if not project.reported_first_user and project.happy_users >= 1:
            print("First user happy for", project.name, " at time", day)
            project.reported_first_user = True
        if not project.reported_all_users and project.happy_users == number_of_users:
            print("All users happy for", project.name, " at time", day)
            project.reported_all_users = True

        if day >= total_time_to_solve*(show_at_percent_done/100):
            print("%i%% complete at time: " % (show_at_percent_done), day)
            show_at_percent_done += 10

        append_to_log(project.make_log_item())
        if CHART_TASKS_COMPLETE:
            chart_data[project.name + ": " + CHART_BUGS].append(len(project.solved_bugs)*100/number_of_bugs)
        chart_data[project.name + ": " + CHART_APPS].append(project.working_apps*100/number_of_apps)
        chart_data[project.name + ": " + CHART_USERS].append(project.happy_users*100/number_of_users)

        project.choose_bug()
        project.work_bug()

        if len(project.solved_bugs) == number_of_bugs:
            print("100% complete at time: ", day + 1)
            append_to_log("%s, %i, 1.0, 1.0, 1.0 \n" % (project.name, day + 1))
            bugs_remaining = False

    day += 1 



# TODO: make optional command line output
if DEBUG: print("Available pick methods:", " ".join(f.__name__ for f in pick_methods))

print("Time spent running simulation:", (time.clock() - timespent))

for project in projects:
    print("---", project.name, "---")
    print("Average features working:", project.working_app_days/day)
    print("Average happy users:", project.happy_user_days/day)

print("Now making chart.")

chart_data = {k:v for (k,v) in chart_data.items() if len(v) > 0} # Remove uncharted things
chart = pandas.DataFrame(chart_data)
params = {'legend.fontsize': 10,
          'legend.linewidth': 2}
plt.rcParams.update(params)
chart.plot()
plt.title(CHART_TITLE)
plt.ylabel(CHART_LABEL_Y)
plt.xlabel(CHART_LABEL_X)
plt.savefig(CHARTFILE)
