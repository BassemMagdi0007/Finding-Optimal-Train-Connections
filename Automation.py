import heapq
import pandas as pd 
from datetime import datetime

# #csvFile = 'schedule.csv'
# csvFile = 'mini-schedule.csv'
# csvPath = 'assignment/' + csvFile
# #csvPath =  csvFile
# costFn = 'arrivaltime'

# csv = pd.read_csv(csvPath)



# #___________________________________grab trains info________________________________________________
# def find_distinct_elements(lst):
#     seen = set()
#     result = []

#     for element in lst:
#         if element not in seen:
#             seen.add(element)
#             result.append(element)

#     return result   

# def find_indices(lst, element):
#     return [index for index, value in enumerate(lst) if value == element]



# sequence = []
# stations = []
# distance = []
# train = []
# arrivalTime = []
# departureTime = []

# # Append the content of the CSV file according to its column name 
# for _, row in csv.iterrows():
#     stations.append(row['station Code'].strip())
#     sequence.append(row['islno'])
#     train.append(row['Train No.'])
#     distance.append(row['Distance'])
#     arrivalTime.append(row['Arrival time'])
#     departureTime.append(row['Departure time'])

# distinct = find_distinct_elements(train)


# usedTrains = []
# source = []
# srcSequence = []
# arrivalTime_start = []
# departureTime_start = []
# arrivalTime_end = []
# departureTime_end = []
# target = []
# trgSequence = []
# distanceDiff = []


# for zug in distinct:
#     indices = find_indices(train, zug)
#     for i in range(len(indices) - 1):
#         source.append(stations[indices[i]])  
#         srcSequence.append(sequence[indices[i]])  
#         target.append(stations[indices[i+1]])
#         trgSequence.append(sequence[indices[i+1]]) 
#         usedTrains.append(train[indices[i]])
#         arrivalTime_start.append(arrivalTime[indices[i]])
#         departureTime_start.append(departureTime[indices[i]])
#         arrivalTime_end.append(arrivalTime[indices[i+1]])
#         departureTime_end.append(departureTime[indices[i+1]])
        
#         # distance cost function
#         distanceDiff.append(distance[indices[i+1]] - distance[indices[i]])
#___________________________________________________________________________________________

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

    #print('GRAPH: ', graph['SGRL'])
    
    # (new_cost, neighbor, path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end)
    priority_queue = [(0, start, [], [], [], [], None, None, None)]  # Adding last_train_number to track the last chosen train number
    visited = set()

    #pop: (current_cost, current_node, path, train_numbers, src_seq_list, tar_seq_list, last_train_number, lastArrivalBeforeTrainChange, lastDepartureBeforeTrainChange)
    #push: (new_cost,     neighbor,    path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end)
    
    # Algorithm Starts 
    while priority_queue:

        # Print to show the Elements in the PriorityQueue__________________________
        # print("\nlen of priority_queue before pop", len(priority_queue))
        # sorted_list = sorted(priority_queue)  # This won't modify the original heap
        # # # # print("Elements in the PriorityQueue:")
        # for item in sorted_list:
        #     print(item, "\n")
        #___________________________________________________________________________


        # POP
        # when the priority queue pops a tuple, each of value in the tuple is assigned to one of the variable on the left of '=' 
        (current_cost, current_node, path, train_numbers, src_seq_list, tar_seq_list, last_train_number, lastArrivalBeforeTrainChange, lastDepartureBeforeTrainChange) = heapq.heappop(priority_queue)
        
        #print("len of priority_queue after pop", len(priority_queue), "\n")
        #print()
        
        # print("LAST TRAIN NUMBER", last_train_number)
        # print("CURRENT NODE ", current_node)
        # print ("GRAPH of CURRENT NODE ", graph.get(current_node, []))
        # #print(f"CURRENT TRAIN : {train_num}")
        # print()

        # path variable holds the nodes visited until it gets pushed to the queue if it has neighbors
        #if current_node not in visited:
        if (current_node, last_train_number) not in visited:
            #visited.add(current_node)
            visited.add((current_node, last_train_number))
            path = path + [current_node]
            # print()
            
            #print(f"Current Shortest Path: {path}")
            # print()
            
            if current_node == end:
                return path, current_cost, train_numbers, src_seq_list, tar_seq_list

            # CHECK NEIGHBORS for current node , update cost from current node to each neighbor (Get those info from graph dictionary)
            for neighbor, weight, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end in graph.get(current_node, []):
                
               
                if costFunction == 'price':
                    if last_train_number != train_num:
                        #print("ADD COST 1")
                        new_cost = current_cost + weight + 1
                        
                        # #-----WRONG!!---- all these timings are of two consequitive stations on the SAME TRAIN
                        # if ((time_to_minutes(dep_end.strip("'")) - time_to_minutes(arr_start.strip("'")) < 10) and
                        #     (time_to_minutes(dep_end.strip("'")) - time_to_minutes(arr_start.strip("'")) > 0)):
                        #     #print("22FSH!")
                        #     new_cost = 100 #infinite new cost i.e don't take this connection.

                    else: # Same train continues 
                        #print("SAME COST")
                        new_cost = current_cost + weight

                        # If the same train passed midnight
                        #print(arr_start, dep_start)
                        
                        if (time_diff_in_minutes(dep_start.strip("'") ,arr_start.strip("'") ) < 0 or 
                            time_diff_in_minutes(arr_end.strip("'") ,dep_start.strip("'")) < 0):
                            #print("Train:", train_num, " Passed midnight at station: ", current_node, neighbor )
                            new_cost = new_cost + 1
                            
                    
                elif costFunction == 'arrivaltime':
             
                    #for the very first train, the travel time is the differnce between the arrival time at next station - the departure time from the very first station
                    
                    # On first train __________________________(1)
                    if last_train_number == None:
                        
                        #print(givenArrivalTime, dep_start, train_num, neighbor)
                        # Travel time between first two stations on the FIRST TRAIN: arrival time at first station - departure time at next station
                        travelTime = time_diff_in_minutes(arr_end.strip("'"), dep_start.strip("'"))
                        
                        #print("NEW ", train_num, current_node, neighbor, travelTime)
                        

                        if travelTime < 0:
                            travelTime = travelTime + 24 * 60 #accumulated travel time adjusted if the origianl value is negative. example 23010 PNME - GMO
                        
                        # the wait on the station since the arrival till the departure of the next train
                        firstArrivDef = time_diff_in_minutes(dep_start.strip("'"), givenArrivalTime.strip("'"))
                        
                        if firstArrivDef < 0:
                            firstArrivDef = firstArrivDef + 24*60
                        
                            #new_cost = current_cost + weight + 24*60 - firstArrivDef + travelTime
                        new_cost = current_cost + weight + firstArrivDef + travelTime

                        #print(f"Checking edge from {current_node} to {neighbor} with cost {new_cost} and data tuple: {((current_node, neighbor, weight, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end, last_train_number))}")
                        # heapq.heappush(priority_queue, (new_cost, neighbor, path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end))
                        # continue
                    
                    # Change trains __________________________(2) 
                    elif last_train_number != train_num:
                        
                        """
                        check for less tha 10 minutes difference:
                        Two possible cases:
                        1) example arrival of train1 13:02:00, departure of train2 13:09:00 --> less than 10 min difference. New cost must be adjusted
                        2) example arrival of train1 23:55:00, departure of train2 00:03:00 --> less than 10 min difference. New cost must be adjusted
                        3)Original arrival time given in pdf. New cost must be adjusted
                        
                        check for passing midnight to update final result
                        1) current train passes midnight
                        2) midnight passes while waiting for the next train
                        """
                        #print(graph.get(path[-2]), last_train_number)
                        # for tp in graph.get(path[-2]):
                            # if path[-1] in tp and last_train_number in tp:
                                # tpfinal = tp
                         
                        # timeDiff = time_to_minutes(dep_end.strip("'")) - time_to_minutes(tpfinal[-2].strip("'"))
                        
                        
                        #time to wait for next train
                        
                        timeDiff = time_diff_in_minutes(dep_start.strip("'"), lastArrivalBeforeTrainChange.strip("'"))
                        if timeDiff < 10:
                            timeDiff = timeDiff + 24 * 60
                        #timeDiff = time_to_minutes(dep_start.strip("'")) - time_to_minutes(lastArrivalBeforeTrainChange.strip("'"))
                        #ie = 24*60 - time_to_minutes(lastArrivalBeforeTrainChange.strip("'")) + time_to_minutes(dep_end.strip("'"))
     
                        #Handling 'Case 1'
                        # if timeDiff < 10 and timeDiff > 0:
                            
                            #print("\n\nAMOMKEN?", last_train_number, lastArrivalBeforeTrainChange, current_node, neighbor, train_num, dep_end, timeDiff)
                            
                            # print("ew3a weshk?", lastArrivalBeforeTrainChange)
                            #print("\n\nAMOMKEN?", last_train_number, tpfinal[-2], current_node, neighbor, train_num, dep_end, timeDiff)
                        
                        
                        #new_cost = current_cost + weight + timeDiff
                            
                        #Handling 'Case 2'
                        
                        # elif ie < 10:
                        #     #print("\n\nAMOMKEN? 2222", last_train_number, lastArrivalBeforeTrainChange, current_node, neighbor, train_num, dep_end, timeDiff, ie)
                        #     new_cost = current_cost + weight + 24*60 + ie
                        
                        #print("\nlastArrivalBeforeTrainChange", lastArrivalBeforeTrainChange, "arr_start", arr_start, "dep_start", dep_start, "arr_end", arr_end, "dep_end", dep_end)
                        
                        trainChangeTravelTime = time_diff_in_minutes(arr_end.strip("'"), dep_start.strip("'"))
                        if trainChangeTravelTime < 0:
                            trainChangeTravelTime =  trainChangeTravelTime + 24 * 60 #accumulated travel time (adjusting for midnight change)
                        
                        new_cost = current_cost + weight + trainChangeTravelTime + timeDiff #accumulated travel time
                        
                        #new_cost = new_cost + 2000 #penalty for changing trains   
                        
                        """
                        WHAT ABOUT THE WAITING TIME AT EACH STATION? MUST BE ADDED TO TOTAL TRAVEL TIME
                        
                        FIXED: ARRIVAL @ end - ARRIVAL @ start
                        """
                    # Same train continues __________________________(3) #arrival @ next station - arrival @ current station 
                    else:  
                        #print("SAME COST", time_to_minutes(arr_end.strip("'")) - time_to_minutes(dep_start.strip("'")))
                        sameTrainTravelTime = time_diff_in_minutes(arr_end.strip("'"), arr_start.strip("'"))
                        
                        #print("SAME Train ", train_num, src_elem, current_node, neighbor, sameTrainTravelTime)
                        
                        # If midnight passed on the same train, adjust the negative by adding a day to it
                        if sameTrainTravelTime < 0:
                            sameTrainTravelTime = sameTrainTravelTime + 24 * 60 #accumulated travel time adjusted if the origianl value is negative. example 23010 PNME - GMO
                            
                        new_cost = current_cost + weight + sameTrainTravelTime                   
                
                # Stops or distance cost functions
                else:
                    new_cost = current_cost + weight
                
                #print(f"Checking edge from {current_node} to {neighbor} with cost {new_cost} and data tuple: {((current_node, neighbor, weight, train_num, src_elem, tar_elem, arr_start, dep_start, arr_end, dep_end, last_train_number))}")
                
                # PUSH the path to the priority queue
                heapq.heappush(priority_queue, (new_cost, neighbor, path, train_numbers + [train_num], src_seq_list + [src_elem], tar_seq_list + [tar_elem], train_num, arr_end, dep_end))

                #if neighbor in visited:
                    #print(f"Using {neighbor} as a cut node to change to a different path in the graph")

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


# Test code
test = pd.read_csv('assignment/example-problems.csv')

problemNo = []
fromSt = []
toSt = []
schedule = []
listIndex = 0
givenArrivalTime = 0

# to only count once it finds the required problems  
for _, row in test.iterrows():
    # if row['Schedule'] == csvFile and costFn in row['CostFunction']: # choose a specific problem --> and row['ProblemNo'] == 62 // row['ProblemNo'] == 1 for clarification
    #     problemNo.append(row['ProblemNo'])
    #     fromSt.append(row['FromStation'].strip())
    #     toSt.append(row['ToStation'].strip())
    #     schedule.append(row['CostFunction'].strip())


    print()
    print(row['ProblemNo'], row['FromStation'].strip(), row['ToStation'].strip())
    print()

    source, target, usedTrains, srcSequence, trgSequence, arrivalTime_start, departureTime_start, arrivalTime_end, departureTime_end, distanceDiff = readCSV(row['Schedule'])
    # Stops cost function
    n = len(source)
    stops = [1] * n
    prices = [0] * n 
    arrivalCost = [0] * n
    weights = []
    CostRowContent = row['CostFunction']

    if CostRowContent == 'stops':
        weights = stops
        costFn = 'stops'
    elif CostRowContent == 'distance':
        weights = distanceDiff
        costFn = 'distance'
    elif CostRowContent == 'price':
        weights = prices
        costFn = 'price'
    elif 'arrivaltime' in CostRowContent:
        weights = arrivalCost
        givenArrivalTime = CostRowContent.split(" ")[1]
        costFn = 'arrivaltime'

    data = {'source' : source,
            'target' : target,
            'train_number': usedTrains,
            'src_seq' : srcSequence ,
            'tar_seq': trgSequence,
            'arrivalTime_start': arrivalTime_start,
            'departureTime_start': departureTime_start,
            'arrivalTime_end': arrivalTime_end,
            'departureTime_end': departureTime_end,
            #'weight': distanceDiff
            #'weight': stops
            #'weight': prices
            'weight' : weights
        }
    
    dijkstra_result = dijkstra_shortest_path(data, row['FromStation'].strip(), row['ToStation'].strip(), costFn, givenArrivalTime)


    result = split_by_sequence(dijkstra_result[2])
    sum = 0
    for lst in result: 
        print("{}: {} -> {} ; ".format(lst[0].strip("'"), dijkstra_result[3][sum], dijkstra_result[4][sum + len(lst) - 1]), end='')

        sum = sum + len(lst)
    print("Cost:", dijkstra_result[1])
    
    if 'arrivaltime' in CostRowContent:
        print("TIME: ", add_times(givenArrivalTime ,dijkstra_result[1]))
    
    # Increment the index only when the both conditions of the if condition is met
    listIndex += 1
    print('\n')