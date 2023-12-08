import heapq
import pandas as pd 
from datetime import datetime


# Iterate over the trains 'lst' fetched from the CSV file and extracts the distinct trains
def find_distinct_elements(lst):
    seen = set()
    result = []

    for element in lst:
        if element not in seen:
            seen.add(element)
            result.append(element)

    return result   


def find_indices(lst, element):
    return [index for index, value in enumerate(lst) if value == element]

# INPUT: mini-schedule.csv or schedule.csv
# OUTPUT: adjust lists used to create the data dictionary 
def readCSV(csvName):

    csvFile = csvName
    csvPath = 'assignment/' + csvFile
    csv = pd.read_csv(csvPath)

    sequence = []
    stations = []
    distance = []
    train = []
    arrivalTime = []
    departureTime = []

    # Read values from CSV file columns
    for _, row in csv.iterrows():
        stations.append(row['station Code'].strip())
        sequence.append(row['islno'])
        train.append(row['Train No.'])
        distance.append(row['Distance'])
        arrivalTime.append(row['Arrival time'])
        departureTime.append(row['Departure time'])

    distinct = find_distinct_elements(train)


    usedTrains = []
    source = []
    srcSequence = []
    arrivalTime_start = []
    departureTime_start = []
    arrivalTime_end = []
    departureTime_end = []
    target = []
    trgSequence = []
    distanceDiff = []

    # Adjust the lists for each train
    for zug in distinct:
        indices = find_indices(train, zug)
        for i in range(len(indices) - 1):
            source.append(stations[indices[i]])  
            srcSequence.append(sequence[indices[i]])  
            target.append(stations[indices[i+1]])
            trgSequence.append(sequence[indices[i+1]]) 
            usedTrains.append(train[indices[i]])
            arrivalTime_start.append(arrivalTime[indices[i]])
            departureTime_start.append(departureTime[indices[i]])
            arrivalTime_end.append(arrivalTime[indices[i+1]])
            departureTime_end.append(departureTime[indices[i+1]])
            
            # distance cost function
            distanceDiff.append(distance[indices[i+1]] - distance[indices[i]])
    
    return source, target, usedTrains, srcSequence, trgSequence, arrivalTime_start, departureTime_start, arrivalTime_end, departureTime_end, distanceDiff


def time_diff_in_minutes(time1, time2):
    format_str = '%H:%M:%S'

    # Convert time strings to datetime objects
    datetime1 = datetime.strptime(time1, format_str)
    datetime2 = datetime.strptime(time2, format_str)

    # Calculate time difference in minutes
    #diff = abs((datetime1 - datetime2).total_seconds()) // 60
    diff = ((datetime1 - datetime2).total_seconds()) // 60
    return diff


# Add resulting arrival time cost to the arrival time given in the excel sheet to produce the final arrive time at end station
def add_times(time_str1, time_str2_minutes):
    # Step 1: Parse the time strings
    time1 = datetime.strptime(time_str1, "%H:%M:%S")
    time2_minutes = int(time_str2_minutes)

    # Step 2: Convert time in minutes to hours and add to the hours from time1
    total_hours = time1.hour + time2_minutes // 60
    remaining_minutes = time1.minute + time2_minutes % 60
    remaining_seconds = time1.second

    # Step 3: Handle overflow for minutes and seconds
    if remaining_minutes >= 60:
        total_hours += 1
        remaining_minutes -= 60

    # Step 4: Calculate the number of days
    days = total_hours // 24
    remaining_hours = total_hours % 24

    # Step 5: Format the result
    result_str = "{:02d}:{:02d}:{:02d}:{:02d}".format(days, remaining_hours, remaining_minutes, remaining_seconds)

    return result_str  


def dijkstra_shortest_path(data, start, end, costFunction, givenArrivalTime):
    graph = {}
    
    for src, tgt, wt, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end in zip(data['source'], data['target'], data['weight'], data['train_number'], data['src_seq'], data['tar_seq'], data['arrivalTime_start'], data['departureTime_start'], data['arrivalTime_end'], data['departureTime_end']):
        if src not in graph:
            graph[src] = []

        # ['SGRL' : ('OBR', 0, "'13346'", 1, 2, "'00:00:00'", "'05:45:00'", "'07:09:00'", "'07:10:00'")] # Iteration 1
        graph[src].append((tgt, wt, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end))
    
    # (new_cost, neighbor, path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end)
    priority_queue = [(0, start, [], [], [], [], None, None)]  # Adding last_train_number to track the last chosen train number
    visited = set()

    #pop: (current_cost, current_node, path, train_numbers, src_seq_list, tar_seq_list, last_train_number, lastArrivalBeforeTrainChange, lastDepartureBeforeTrainChange)
    #push: (new_cost,     neighbor,    path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end)
    
    # Algorithm Starts 
    while priority_queue:

        # POP
        # when the priority queue pops a tuple, each of value in the tuple is assigned to one of the variable on the left of '=' 
        (current_cost, current_node, path, train_numbers, src_seq_list, tar_seq_list, last_train_number, lastArrivalBeforeTrainChange) = heapq.heappop(priority_queue)

        # path variable holds the nodes visited until it gets pushed to the queue if it has neighbors
        #if current_node not in visited:
        if (current_node, last_train_number) not in visited:
            visited.add((current_node, last_train_number))
            path = path + [current_node]

            # If ite reaches the desired node
            if current_node == end:
                return path, current_cost, train_numbers, src_seq_list, tar_seq_list

            # CHECK NEIGHBORS for current node , update cost from current node to each neighbor (Get those info from graph dictionary)
            for neighbor, weight, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end in graph.get(current_node, []):
                
                if costFunction == 'price':
                    if last_train_number != train_num:
                        new_cost = current_cost + weight + 1
                        
                    else: # Same train continues 
                        new_cost = current_cost + weight

                        # If the same train passed midnight                       
                        if (time_diff_in_minutes(dep_start.strip("'") ,arr_start.strip("'") ) < 0 or 
                            time_diff_in_minutes(arr_end.strip("'") ,dep_start.strip("'")) < 0):
                            new_cost = new_cost + 1
                            
                    
                elif costFunction == 'arrivaltime':
                    # for the very first train, the travel time is the difference between the arrival time at next station - the departure time from the very first station
                    # On first train __________________________(1)
                    if last_train_number == None:
                        
                        # Travel time between first two stations on the FIRST TRAIN: arrival time at first station - departure time at next station
                        travelTime = time_diff_in_minutes(arr_end.strip("'"), dep_start.strip("'"))
                        
                        if travelTime < 0:
                            travelTime = travelTime + 24 * 60 #accumulated travel time adjusted if the origianl value is negative. example 23010 PNME - GMO
                        
                        # the wait on the station since the arrival till the departure of the next train
                        firstArrivDef = time_diff_in_minutes(dep_start.strip("'"), givenArrivalTime.strip("'"))
                        
                        if firstArrivDef < 0:
                            firstArrivDef = firstArrivDef + 24*60
                        
                        new_cost = current_cost + weight + firstArrivDef + travelTime

                    # Change trains __________________________(2) 
                    elif last_train_number != train_num:
                        
                        #time to wait for next train
                        timeDiff = time_diff_in_minutes(dep_start.strip("'"), lastArrivalBeforeTrainChange.strip("'"))
                        if timeDiff < 10:
                            timeDiff = timeDiff + 24 * 60
              
                        trainChangeTravelTime = time_diff_in_minutes(arr_end.strip("'"), dep_start.strip("'"))
                        if trainChangeTravelTime < 0:
                            trainChangeTravelTime =  trainChangeTravelTime + 24 * 60 #accumulated travel time (adjusting for midnight change)
                        
                        new_cost = current_cost + weight + trainChangeTravelTime + timeDiff #accumulated travel time
                        
                        
                    # Same train continues __________________________(3) #arrival @ next station - arrival @ current station 
                    else:  
                        sameTrainTravelTime = time_diff_in_minutes(arr_end.strip("'"), arr_start.strip("'"))
   
                        # If midnight passed on the same train, adjust the negative by adding a day to it
                        if sameTrainTravelTime < 0:
                            sameTrainTravelTime = sameTrainTravelTime + 24 * 60 #accumulated travel time adjusted if the origianl value is negative. example 23010 PNME - GMO
                            
                        new_cost = current_cost + weight + sameTrainTravelTime                   
                
                # Stops or distance cost functions
                else:
                    new_cost = current_cost + weight
                
                # PUSH the path to the priority queue
                heapq.heappush(priority_queue, (new_cost, neighbor, path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end))

    return None

# Output formatting functions
# trains: [1,1,1,1], [5,5], [1], [6,6,6,6]
def split_by_sequence(input_list):
    result = []
    current_sequence = []

    for item in input_list:
        if not current_sequence or item == current_sequence[-1]:
            current_sequence.append(item)
        else:
            result.append(current_sequence.copy())
            current_sequence = [item]

    if current_sequence:
        result.append(current_sequence)

    return result

# Function to write the results to solutions.csv
def write_to_csv(problem_no, connections, cost, csv_path='solutions.csv'):
    df = pd.DataFrame({
        'ProblemNo': [problem_no],
        'Connection': [connections],
        'Cost': [cost]
    })
    
    try:
        existing_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # File doesn't exist, create a new DataFrame
        existing_df = pd.DataFrame(columns=['ProblemNo', 'Connection', 'Cost'])

    df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_path, index=False)



# Test code
test = pd.read_csv('problems.csv')

listIndex = 0
givenArrivalTime = 0

source, target, usedTrains, srcSequence, trgSequence, arrivalTime_start, departureTime_start, arrivalTime_end, departureTime_end, distanceDiffMini = readCSV('mini-schedule.csv')

# cost functions mini
n = len(source)
stopsMini = [1] * n
pricesMini = [0] * n 
arrivalCostMini = [0] * n
weightsMini = [0] * n

miniData = {}
miniData = {
        'source' : source,
        'target' : target,
        'train_number': usedTrains,
        'src_seq' : srcSequence ,
        'tar_seq': trgSequence,
        'arrivalTime_start': arrivalTime_start,
        'departureTime_start': departureTime_start,
        'arrivalTime_end': arrivalTime_end,
        'departureTime_end': departureTime_end,
        'weight' : weightsMini
    }
source, target, usedTrains, srcSequence, trgSequence, arrivalTime_start, departureTime_start, arrivalTime_end, departureTime_end, distanceDiffBig = readCSV('schedule.csv')

# cost functions schedule
n = len(source)
stopsBig = [1] * n
pricesBig = [0] * n 
arrivalCostBig = [0] * n
weightsBig = [0] * n

scheduleData = {}
scheduleData = {
        'source' : source,
        'target' : target,
        'train_number': usedTrains,
        'src_seq' : srcSequence,
        'tar_seq': trgSequence,
        'arrivalTime_start': arrivalTime_start,
        'departureTime_start': departureTime_start,
        'arrivalTime_end': arrivalTime_end,
        'departureTime_end': departureTime_end,
        'weight' : weightsBig
    }

# to only count once it finds the required problems  
for _, row in test.iterrows():

    print()
    print(row['ProblemNo'], row['FromStation'].strip(), row['ToStation'].strip())
    print()

    CostRowContent = row['CostFunction']

    if CostRowContent == 'stops':
        costFn = 'stops'
        if row['Schedule'] == 'mini-schedule.csv': 
            miniData['weight'] = stopsMini
        else:
            scheduleData['weight'] = stopsBig

    elif CostRowContent == 'distance':
        costFn = 'distance'
        if row['Schedule'] == 'mini-schedule.csv':
            miniData['weight'] = distanceDiffMini
        else:
            scheduleData['weight'] = distanceDiffBig

    elif CostRowContent == 'price':
        costFn = 'price'
        if row['Schedule'] == 'mini-schedule.csv': 
            miniData['weight'] = pricesMini
        else:
            scheduleData['weight'] = pricesBig

    elif 'arrivaltime' in CostRowContent:
        costFn = 'arrivaltime'
        givenArrivalTime = CostRowContent.split(" ")[1]
        if row['Schedule'] == 'mini-schedule.csv':
            miniData['weight'] = arrivalCostMini
        else:
            scheduleData['weight'] = arrivalCostBig


    if row['Schedule'] == 'mini-schedule.csv':
        dijkstra_result = dijkstra_shortest_path(miniData, row['FromStation'].strip(), row['ToStation'].strip(), costFn, givenArrivalTime)
    else:
        dijkstra_result = dijkstra_shortest_path(scheduleData, row['FromStation'].strip(), row['ToStation'].strip(), costFn, givenArrivalTime)


    result = split_by_sequence(dijkstra_result[2])
    sum = 0
    connections = ""
    for idx, lst in enumerate(result): 
        connections += "{} : {} -> {}".format(lst[0].strip("'"), dijkstra_result[3][sum], dijkstra_result[4][sum + len(lst) - 1])
        # Check if it's not the last iteration before adding the semicolon
        if idx < len(result) - 1:
            connections += " ; "
        ### print("{} : {} -> {} ; ".format(lst[0].strip("'"), dijkstra_result[3][sum], dijkstra_result[4][sum + len(lst) - 1]), end='')
        sum = sum + len(lst)

    cost = dijkstra_result[1]
    if 'arrivaltime' in CostRowContent:
        time_result = add_times(givenArrivalTime, dijkstra_result[1])
        print("TIME:", time_result)
        cost = time_result  # Update cost for arrivaltime
        ### print("TIME: ", add_times(givenArrivalTime ,dijkstra_result[1]))
    print("Cost:", cost)

    #Write results to CSV
    write_to_csv(row['ProblemNo'], connections, cost)

    # Increment the index only when the both conditions of the if condition is met
    listIndex += 1
    print('\n')