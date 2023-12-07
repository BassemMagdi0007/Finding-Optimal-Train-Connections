# Assignment 1.1: Find Train Connections

This repository contains a Python implementation for optimizing transportation scheduling problem using Dijkstra's algorithm for finding the shortest path in a directed graph. The nodes in the graph represent train stations, edges represent train connections, and weights correspond to different cost functions, such as stops, distance, price, and arrival time.

## Table of Contents

- [Introduction](#introduction)
  - Key Features
- [Setup](#setup)
  - Repository content
  - How to run the code
  - Used libraries
- [Code Structure](#code-structure)
- [Self Evaluation and Design Decisions](#design-decision)
- [Output Format](#output-format)
- [Important Note](#important-note)
- [Use Cases](#use-cases)
- [License](#license)

## Introduction

The primary purpose of this script is to address the challenges associated with optimizing train journeys in a network of interconnected stations. By leveraging Dijkstra's algorithm, the script identifies the most efficient routes based on different criteria, offering valuable insights into the intricacies of train schedules.

### Key Features 
- **Dynamic Cost Functions:** The script supports multiple cost functions, allowing users to customize the optimization criteria based on their specific requirements. Whether prioritizing minimal stops, shortest distance, cost-effective pricing, or efficient arrival times, the script adapts accordingly.
- **Comprehensive Graph Representation:** The train schedule is represented as a weighted directed graph, where stations serve as nodes, and the edges between these nodes carry diffrent labels such as: train number, arrival and departure times, station numbers, and various cost functions as weights. This comprehensive representation enables a detailed analysis of the entire train network.
- **Efficient Path Exploration:** Dijkstra's algorithm, a widely used algorithm for finding the shortest paths in weighted graphs, is employed to ensure efficient and accurate exploration of potential routes. The priority queue implementation further enhances the speed of path discovery.
- **Input Flexibility:** The script seamlessly integrates with CSV files containing train schedule data. This allows users to apply the algorithm to different scenarios by providing their own datasets.

## Setup
### This repository contains:
 1) **`train_connections.py`**: That contains the implementation for solving the train connection problem. It includes functions for reading train schedule data from CSV files, applying Dijkstra's algorithm with various cost functions, and writing the results to a CSV file.
 2) **`solutions.csv`**: solutions for the "problem.csv" file

### How to run the code: 

1. Import pandas as pd, import heapq, from datetime import datetime.
2. **problem.csv** and **`train_connections.py`** must be in the same folder.
3. Run **`train_connections.py`**.
### Used libraries:

**_heapq:_**
Description: The heapq library in Python provides an implementation of the heap queue algorithm, which is a priority queue algorithm. It is used for efficiently maintaining a priority queue, allowing for quick retrieval of the smallest element.
Purpose in Code: In the script, heapq is utilized to manage a priority queue during the execution of Dijkstra's algorithm. This enables the efficient exploration of potential paths in the weighted graph.

**_pandas:_**
Description: pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrame for efficient handling and analysis of structured data.
Purpose in Code: The script employs pandas to read and process data from CSV files containing train schedule information. It simplifies the manipulation of tabular data, making it easier to extract relevant details for further analysis.

**_datetime:_**
Description: The datetime module is part of the Python standard library and provides classes for working with dates and times.
Purpose in Code: In the script, datetime is used to handle and manipulate time-related information. It aids in calculating time differences, parsing time strings, and performing operations related to arrival and departure times in the train schedule.
## Code Structure
1) **Library imports**
```python
import heapq
import pandas as pd 
from datetime import datetime
```


2) **Parsing CSV input**
```python
# INPUT: mini-schedule.csv or schedule.csv
# OUTPUT: adjust lists used to create the data dictionary 
def readCSV(csvName):
# ...
```
- **_First_**, The readCSV function parses the CSV file whose name is passed to it as an argument 'csvPath', and assigns it into a Pandas DataFrame called 'csv'.

- **_Second_**, It iterates over each row in the CSV file, extracting the contents of each column in the CSV input file and appending them to diffrenet lists.

- **_Third_**, It finds distinct trains using 
```python
# Iterate over the trains 'lst' fetched from the CSV file and extracts the distinct trains
def find_distinct_elements(lst):
# ...
```
- **_Fourth_**, based on the previous generated lists, it creates new lists containing all possible information for each single train,  such as its 'arrival time', 'departure time', 'source stations', 'target stations', etc.. .

- **_Finally_**, the function returns a tuple containing various lists that have been adjusted based on the information extracted from the CSV file. 

3) **Time Manipulation Functions**
```python
def time_diff_in_minutes(time1, time2):
    # ...
```
This function calculates the time difference in minutes between two time strings in the format '%H:%M:%S', converts them to datetime objects, and return the diffrence in minutes
```python
def add_times(time_str1, time_str2_minutes):
    # ...
```
This function takes a time string (time_str1) _which in our scenario is the givenArrivalTime in the CSV file for the arrivaltime Cost function problems_, and a duration in minutes (time_str2_minutes) and adds them together in order to find the time spended since the arrival to the train station till the first train departure. 

4) **Dijkstra Algorithm**
```python
def dijkstra_shortest_path(data, start, end, costFunction, givenArrivalTime):
    # ...
```
- **Graph Initialization:**
The graph dictionary is initialized to represent the adjacency list of the graph. It is constructed from the input data, which includes source (src), target (tgt), weight (wt), train number (train_num), source sequence (src_elem), target sequence (tar_elem), arrival and departure times at the start and end stations.

- **Priority Queue Initialization:**
The priority_queue is a min-heap containing tuples with the format (current_cost, current_node, path, train_numbers, src_seq_list, tar_seq_list, last_train_number, lastArrivalBeforeTrainChange, lastDepartureBeforeTrainChange). It starts with a single entry representing the start node with cost 0.

- **Main Loop:**
The function enters a loop that continues until the priority queue is empty.
In each iteration, it pops the node with the lowest cost from the priority queue.
The function checks if the node has been visited before based on the combination of the current node and the last train number. If not, it adds the node to the visited set and continues.

- **Neighbor Exploration:**
For each neighbor of the current node, it calculates the new cost based on the chosen cost function (costFunction). The cost is updated according to the specified conditions for different cost functions.
The neighbor is pushed into the priority queue with the updated cost and other relevant information.

- **Termination and Result:**
If the current node is equal to the destination node (end), the function returns the path, cost, and other relevant information.
If the priority queue becomes empty and the destination is not reached, the function returns None.

- **Cost Functions:**
The function supports different cost functions such as 'stops', 'distance', 'price' and 'arrivaltime'. The cost is updated accordingly based on the chosen cost function.

## Self Evaluation and Design Decisions
Check if a certain station was visited on which train before marking it as visited

Before: Parse problem examples and formulate the data dectionary (read stations csv file) with every problem.

After: 
- Calculate the data dectionary for mini-schedule and schedule once  
- saved 200 seconds run time

SCORE:

Total points scored on the 'example-problems.csv'
```python 
TOTAL POINTS: 79
```

## Output Format

| ProblemNo     | Connection | Cost
| ------------- | ------------- | ----------
| 0  | 19269 : 27 -> 28         | 1
| 1  | 19269 : 7 -> 22 ; 14266 : 19 -> 20 ; 19269 : 23 -> 24 ; 23010 : 19 -> 23  | 21
| 2 | 23010 : 64 -> 68          | 4
|.  |        .                  | .

The script generates the output in a table format where:

- **First column:** Problem number.

- **Second column:** The trains used to reach from the start station to the desired station, and the staions they passed on the way.

- **Thied column:** The cost score for each problem, it varies according to diffrent 'cost functions'.

## Important Note
problem 75

## Use Cases

- **Transportation Optimization:** Transportation authorities can use the script to optimize train routes, minimizing travel time, reducing costs, or improving passenger convenience.

- **Research and Analysis:** Researchers and analysts can leverage the script to study the dynamics of train schedules, exploring the impact of different cost factors on the overall network.

- **Educational Tool:** The script serves as an educational tool for computer science students studying algorithms, graph theory, and optimization techniques. It provides practical insights into the application of Dijkstra's algorithm in real-world scenarios.
