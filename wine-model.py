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

import csv
import matplotlib.pyplot as plt
import pandas
import random
import time
from collections import Counter
from functools import partial
from math import sqrt
from operator import itemgetter

DEBUG = False
LOGFILE = 'wine-model-results.csv'
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


MIN_APPS_PER_USER = 1
MAX_APPS_PER_USER = 10

###
### Basic setup -- meant to be modified by user
###

# Note that internally "features" == "apps" and "work items" == "bugs"
number_of_bugs, number_of_apps, number_of_users = 10000, 2500, 5000
# For additional options, see pick_methods below
STRATEGIES_TO_TEST = ["Most popular feature", "Easiest feature", "Satisfy easiest user first", "Rotate strategies"]

# TODO: document!
def setup_functions():
    global bug_difficulty_function, bug_probability_function 
    global app_frequency_function, apps_per_user_function
    bug_difficulty_function = lambda: abs(int(random.normalvariate(4,3))) + 1
    bug_probability_function = partial(probability_list_from_zipfs_law, number_of_bugs)
    app_frequency_function = partial(frequency_list_from_pareto_distribution, number_of_apps)
    apps_per_user_function = partial(random.randint, MIN_APPS_PER_USER, MAX_APPS_PER_USER)

enable_log = False
if RANDOM_SEED: 
    random.seed(a=RANDOM_SEED)

###
### Strategy -- meant to be modified by user
###

# TODO: just get rid of this and use pick method directly
def strategy_chooser(name: str) -> "function":
    """Returns a function that returns a pick method based on the current state"""
    x = pick_methods.get(name)
    if x:
        return lambda: x
    
    raise ValueError("Unrecognized strategy: %s" % name)

###
### Pick Methods
###

pick_methods = {}
def pick_method(name):
    def setup_pick_method(function):
        pick_methods[name] = function
        return function
    return setup_pick_method

@pick_method("Rotate strategies")
def rotate_strategy(project):
    """Returns a pick method based on the day"""
    if day <= total_time_to_solve / 20: # Do nothing but this strategy for the first 5%
        return pick_specific_from_most_common_by_feature(project)
    elif day <= total_time_to_solve / 10:
        return pick_specific_from_easiest_user(project) # MVP! Early Adopters!
    elif day % 2 == 1:
        return pick_specific_from_most_common_by_feature(project)
    else:
        return pick_specific_from_easiest_user(project)

@pick_method("First item on list")
def pick_specific_from_all_bugs(project):
    """Picks the smallest bug number not in the solved_bugs list"""
    return next(project.bugs_by_number)

@pick_method("Random item")
def pick_random_from_all_bugs(project):
    """Picks a random unsolved bug"""
    return next(project.random_bugs)

@pick_method("Random feature")
def pick_specific_from_random_app(project):
    """Picks the smallest bug from a random app"""
    for app in project.random_apps:
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_all_bugs(project)

@pick_method("Random item in random feature")
def pick_random_from_random_app(project):
    """Picks a random bug from a random app"""
    for app in project.random_apps:
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_all_bugs(project)

@pick_method("First feature on list")
def pick_specific_from_specific_app(project):
    for app in project.apps_by_number:
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_all_bugs(project) 

@pick_method("Random item in first feature on list")
def pick_random_from_specific_app(project):
    for app in project.apps_by_number:
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_all_bugs(project) 

@pick_method("First user on list")
def pick_specific_from_specific_user(project):
    """Picks the smallest bug from the smallest app from the smallest user"""
    for user in project.users_by_number:
        app = min(project.users[user] - project.solved_apps)
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_specific_app(project)

@pick_method("Random feature from first user on list")
def pick_random_from_specific_user(project):
    """Picks a random bug from a random app from the smallest user"""
    for user in project.users_by_number:
        app = random.choice(tuple(project.users[user] - project.solved_apps))
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_random_app(project)

@pick_method("First item on list for random user")
def pick_specific_from_random_user(project):
    """Picks the smallest bug in the smallest app from a random user"""
    for user in project.random_users:
        app = min(project.users[user] - project.solved_apps)
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_random_app(project)

@pick_method("Random feature from random user")
def pick_random_from_random_user(project):
    """Picks a random bug from a random app from a random user"""
    for user in project.random_users:
        app = random.choice(tuple(project.users[user] - project.solved_apps))
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_random_app(project)

@pick_method("Easiest feature")
def pick_specific_from_easiest_app(project):
    """Picks the smallest bug from the smallest app with the fewest bugs remaining"""
    easy_apps = project.easiest_apps()
    if easy_apps:
        app = min(easy_apps)
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_all_bugs(project) 

@pick_method("Random item among easiest features")
def pick_random_from_easiest_app(project):
    """Picks a random bug from a random app with the fewest bugs remaining"""
    easy_apps = project.easiest_apps()
    if easy_apps:
        app = random.choice(easy_apps)
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_all_bugs(project)

@pick_method("Satisfy easiest user first")
def pick_specific_from_easiest_user(project):
    easy_users = project.easiest_users()
    if easy_users:
        user = min(easy_users)
        app = min(project.users[user] - project.solved_apps)
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_easiest_app(project) 

@pick_method("Random feature among almost happy users")
def pick_random_from_easiest_user(project):
    easy_users = project.easiest_users()
    if easy_users:
        user = random.choice(easy_users)
        app = random.choice(tuple(project.users[user] - project.solved_apps))
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_easiest_app(project)

@pick_method("Most common item among features")
def pick_specific_from_most_common_by_feature(project):
    """Picks the bug that is the most common among all the unfinished features"""
    return next(project.bugs_by_popularity_in_apps)

@pick_method("Most popular feature")
def pick_specific_from_most_popular_app(project):
    """Picks a specific bug from the most popular app"""
    for app in (project.apps_by_popularity_in_users):
        return min(project.apps[app] - project.solved_bugs)
    return pick_specific_from_all_bugs(project)

@pick_method("Random item from most popular feature")
def pick_random_from_most_popular_app(project):
    """Picks a random bug from the most popular app"""
    for app in (project.apps_by_popularity_in_users):
        return random.choice(tuple(project.apps[app] - project.solved_bugs))
    return pick_random_from_all_bugs(project)

@pick_method("Easiest work item (first in list)")
def pick_specific_from_easiest_bugs(project):
    easiest_difficulty = None
    for bug, difficulty in project.bug_difficulty.items():
        if 0 < difficulty <= 1: # Doesn't get any easier
            return bug
        if difficulty > 1 and (easiest_difficulty is None or difficulty < easiest_difficulty):
            easiest_difficulty = difficulty
            easiest_bug = bug
    return easiest_bug

@pick_method("Easiest work item (random choice)")
def pick_random_from_easiest_bugs(project):
    easiest_difficulty = None
    for bug, difficulty in project.bug_difficulty.items():
        if 0 < difficulty and (easiest_difficulty is None or difficulty < easiest_difficulty):
            candidates = {bug}
            easiest_difficulty = difficulty
        elif 0 < difficulty == easiest_difficulty:
            candidates.add(bug)
    return random.choice(tuple(candidates))

###
### Generators and helper functions for pick methods
###

def goals_requiring_tasks(goals: tuple, total_tasks: int) -> dict: # TODO: make this return a list, refactor
    """Traverses a goals:tasks tuple and creates a task:goals-needing-that-task dictionary"""
    goals_by_tasks = {task: set() for task in range(total_tasks)} # TODO: frozenset?
    for goal, tasks in enumerate(goals):
        for task in tasks:
            goals_by_tasks[task].add(goal)
    return goals_by_tasks

def prioritize(goals: tuple, total_tasks: int):
    """Generator to yield tasks within a tuple of goals based on their frequency"""
    count = Counter()
    for tasks in goals:
        for task in tasks:
            count[task] += 1
    yield from (task for (task, frequency) in sorted(count.items(), key=itemgetter(1), reverse=True))

def goals_by_number_generator(number_of_goals: int, solved_goals: set):
    for goal in range(number_of_goals):
        while goal not in solved_goals: yield goal

def goals_by_random_generator(number_of_goals: int, solved_goals: set):
    """Generator to yield unsolved goals at random"""
    unfinished_goals = set(range(number_of_goals)) - solved_goals
    while unfinished_goals:
        goal = random.choice(tuple(unfinished_goals))
        if goal in solved_goals:
            unfinished_goals.remove(goal)
        else:
            yield goal

def bugs_by_popularity_in_apps_generator(apps: tuple, solved_bugs: set):
    for bug in prioritize(goals=apps, total_tasks=number_of_bugs):
        while bug not in solved_bugs: yield bug

def apps_by_popularity_in_users_generator(users: tuple, solved_apps: set):
    for app in prioritize(goals=users, total_tasks=number_of_apps):
        while app not in solved_apps: yield app

###
### App and User Setup
###

def probability_list_from_zipfs_law(size: int) -> list:
    """Returns a list of floats from 0 to 1 based on a Zipfian distribution"""
    item_probability = [1.0/(x+1)**.33 for x in range(size)]
    random.shuffle(item_probability) # Prevent "smallest number" from implying "more likely"
    return item_probability

def set_from_fixed_probabilities(probability: list) -> frozenset:
    """Returns a set of numbers by randomly testing to include each one based on probability."""
    return frozenset(item for (item, chance) in enumerate(probability) if random.uniform(0,1) <= chance)

def frequency_list_from_pareto_distribution(size: int) -> list:
    """Returns a set of relative probabilities based on a pareto distribution"""
    return [random.paretovariate(2.2) for x in range(size)]

def set_from_relative_frequencies(frequency: list, quantity: int, mutate_list=False) -> frozenset:
    """Returns a set of quantity numbers based on the frequency list. An item of frequency 2 is
    twice as likely to be picked on each selection as an item of frequency 1, 4 is 4 times as likely,
    and so on.
    """
    assert quantity <= len(frequency)
    frequency=frequency.copy()
    results = []
    for x in range(quantity):
        length_of_ruler = sum(frequency)
        point_on_line = random.uniform(0,length_of_ruler)
        for (index, length_of_segment) in enumerate(frequency):
            point_on_line -= length_of_segment
            if point_on_line < 0:
                results.append(index)
                frequency[index] = 0
                break
        else:
            assert False
    return frozenset(results)

class Project:
    setup_functions()
    global total_time_to_solve

    initial_bug_difficulty = {bug: bug_difficulty_function() for bug in range(number_of_bugs)}
    total_time_to_solve = sum(initial_bug_difficulty.values())
    print("Work items generated, with", total_time_to_solve, "total time to finish every work item.")

    apps = tuple(set_from_fixed_probabilities(bug_probability_function()) for app in range(number_of_apps))
    average_bugs_per_app = sum(len(app) for app in apps) / number_of_apps
    print("Features generated, averaging", average_bugs_per_app, "items per feature.")

    users = tuple(set_from_relative_frequencies(app_frequency_function(), apps_per_user_function())
                  for user in range(number_of_users))
    average_apps_per_user = sum(len(user) for user in users) / number_of_users
    print("Users generated, averaging", average_apps_per_user, "features per user.")

    apps_affected_by_bug = goals_requiring_tasks(apps, number_of_bugs)
    users_affected_by_app = goals_requiring_tasks(users, number_of_apps)

    # TODO: make these lists
    initial_app_bugs_remaining = {app_id: len(bugs) for app_id, bugs in enumerate(apps)} 
    initial_user_apps_remaining = {user_id: len(apps) for user_id, apps in enumerate(users)}

    def __init__(self, name):
        self.bug_difficulty = self.initial_bug_difficulty.copy()
        self.solved_bugs = set()
        self.solved_apps = set()
        self.solved_users = set()
        self.name = name
        self.method_selector = strategy_chooser(name)

        # Class-wide generators to preserve state
        self.bugs_by_number = goals_by_number_generator(number_of_bugs, self.solved_bugs)
        self.random_bugs = goals_by_random_generator(number_of_bugs, self.solved_bugs)
        self.apps_by_number = goals_by_number_generator(number_of_apps, self.solved_apps)
        self.random_apps = goals_by_random_generator(number_of_apps, self.solved_apps)
        self.users_by_number = goals_by_number_generator(number_of_users, self.solved_users)
        self.random_users = goals_by_random_generator(number_of_users, self.solved_users)
        self.bugs_by_popularity_in_apps = bugs_by_popularity_in_apps_generator(self.apps, self.solved_bugs)
        self.apps_by_popularity_in_users = apps_by_popularity_in_users_generator(self.users, self.solved_apps)
        
        self.working_app_days = 0
        self.working_app_count = 0
        self.happy_user_days = 0
        self.happy_user_count = 0
        self.bug_in_progress = None
        self.reported_first_app, self.reported_first_user = False, False
        self.reported_all_apps, self.reported_all_users = False, False

        # TODO: these should be generated only once and copied on class instantiation
        self.app_bugs_remaining = self.initial_app_bugs_remaining.copy()
        self.user_apps_remaining = self.initial_user_apps_remaining.copy()

    def easiest_apps(self) -> tuple:
        unsolved = [(app, bugs) for app, bugs in self.app_bugs_remaining.items() if bugs > 0]
        smallest_bug_count = min(unsolved, key=itemgetter(1), default=(0,0))[1]
        return tuple(app for app, bugs in unsolved if bugs == smallest_bug_count)

    def easiest_users(self) -> tuple:
        unsolved = [(user, apps) for user, apps in self.user_apps_remaining.items() if apps > 0]
        smallest_app_count = min(unsolved, key=itemgetter(1), default=(0,0))[1]
        return tuple(user for user, apps in unsolved if apps == smallest_app_count)

    def make_log_item(self) -> list:
        log = [self.name, day, len(self.solved_bugs)/number_of_bugs,
               self.working_app_count/number_of_apps, self.happy_user_count/number_of_users]
        return log

    def choose_bug(self):
        """Finds a bug to work on and sets bug_in_progress to it"""
        if self.bug_in_progress is not None and FINISH_TASKS_BEFORE_CHANGING_STRATEGY:
            # TODO: should also be in here if the pick method always returns the same, so we don't recompute
            return
        else:
            pick_method = self.method_selector()
            self.bug_in_progress = pick_method(self)
            assert self.bug_in_progress not in self.solved_bugs

    def work_bug(self):
        """Works on the bug_in_progress"""
        self.working_app_days += self.working_app_count
        self.happy_user_days += self.happy_user_count

        self.bug_difficulty[self.bug_in_progress] -= 1
        if DEBUG: print("worked bug:", self.bug_in_progress)
        if self.bug_difficulty[self.bug_in_progress] <= 0:
            self.solve_bug(self.bug_in_progress)
            self.bug_in_progress = None

    def solve_bug(self, bug):
        self.solved_bugs.add(bug)
        if DEBUG: print("solved bug:", bug)
        for app in self.apps_affected_by_bug[bug]:
            self.app_bugs_remaining[app] -= 1
            if self.app_bugs_remaining[app] == 0:
                self.solve_app(app)

    def solve_app(self, app):
        self.solved_apps.add(app)
        self.working_app_count += 1
        if DEBUG: print("solved app:", app)
        for user in self.users_affected_by_app[app]:
            self.user_apps_remaining[user] -= 1
            if self.user_apps_remaining[user] == 0:
                self.solved_users.add(user)
                self.happy_user_count += 1
                if DEBUG: print("happy user:", user)
                

###
### Helper functions for running simulation
###

def append_to_log(entry: list):
    assert enable_log
    with open(LOGFILE, 'a', newline='') as logfile:
        logger = csv.writer(logfile)
        logger.writerow(entry)

def setup():
    """Creates apps and users and erases the log"""
    global projects

    if enable_log: # Erase logfile
        with open(LOGFILE, 'w'): pass

    assert STRATEGIES_TO_TEST
    projects = [Project(strategy) for strategy in STRATEGIES_TO_TEST]

###
### Simulation begins here
###

print("Modeling with", number_of_bugs, "bugs,", number_of_apps, "apps, and", number_of_users, "users")
setup()

day = 0

timespent = time.clock()

if enable_log:
    append_to_log(["Strategy","Time","% Work Items Completed","% Features Completed","% Happy Users"])
chart_data = {}
for project in projects:
    name = project.name
    chart_data.update({name+": "+CHART_BUGS: [], name+": "+CHART_APPS: [], name+": "+CHART_USERS: []})

show_at_percent_done = 10 # When to first show 'working on day' (x) progress indicators
bugs_remaining = True

while(bugs_remaining): # TODO: just use the inner for loop to cycle over projects
    for project in projects:
        if not project.reported_first_app and project.working_app_count >= 1:
            print("First feature working for", project.name, "at time", day)
            project.reported_first_app = True
        if not project.reported_all_apps and project.working_app_count == number_of_apps:
            print("All features working for", project.name, " at time", day)
            project.reported_all_apps = True
        if not project.reported_first_user and project.happy_user_count >= 1:
            print("First user happy for", project.name, " at time", day)
            project.reported_first_user = True
        if not project.reported_all_users and project.happy_user_count == number_of_users:
            print("All users happy for", project.name, " at time", day)
            project.reported_all_users = True

        if day >= total_time_to_solve*(show_at_percent_done/100):
            print("%i%% complete at time: " % (show_at_percent_done), day)
            show_at_percent_done += 10

        if enable_log:
            append_to_log(project.make_log_item())
        if CHART_TASKS_COMPLETE:
            chart_data[project.name + ": " + CHART_BUGS].append(len(project.solved_bugs)*100/number_of_bugs)
        chart_data[project.name + ": " + CHART_APPS].append(project.working_app_count*100/number_of_apps)
        chart_data[project.name + ": " + CHART_USERS].append(project.happy_user_count*100/number_of_users)

        project.choose_bug()
        project.work_bug()

        if len(project.solved_bugs) == number_of_bugs:
            if enable_log:
                append_to_log([project.name, day + 1, 1.0, 1.0, 1.0])
            bugs_remaining = False

    day += 1 

print("100% complete at time:", day)
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
