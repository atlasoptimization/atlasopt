"""
The goal of this script is to demonstrate optimal vehicle routing. We want to 
derive a route traversing all points of interest in the least amount of 
cumulated time. Three vehicles can be used toward this purpose. The problem 
features 15 randomly generated  points in 2 D space which are used to generate 
the distance matrix needed to evaluate travel times.
For this, do the following:
    1. Imports and definitions
    2. Generate problem data
    3. Formulate the Optimization problem
    4. Assemble the solution
    5. Plots and illustratons

More Information can be found e.g. on pp. 151-156 in the Handbook on modelling 
for discrete optimization by G. Appa, L. Pitsoulis, and H. P. Williams,
Springer Science & Business Media (2006).
The problem is formulated and solved OR-Tools, a Python framework for posing and
solving problems from operations research by Google. More information on 
OR-Tools 7.2. by Laurent Perron and Vincent Furnon can be found on
https://developers.google.com/optimization/.


The script is meant solely for educational and illustrative purposes. Adapted 
from the OR-Tools tutorial https://developers.google.com/optimization/routing/vrp
on the vehicle routing problem. Written by Jemil Avers Butt, Atlas optimization
GmbH, www.atlasoptimization.ch.

"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ii) Definitions

n_locations=45                  # Should be multiple of 3 for cluster simulation
n_clusterpoints=np.round(n_locations/3).astype(int)


"""
    2. Generate problem data --------------------------------------------------
"""


# i) Random locations

# np.random.seed(3)                               # Activate line for reproducibility 

mu_1=np.array([[1],[1]])
mu_2=np.array([[-1],[1]])
mu_3=np.array([[0],[-1]])

x_1=np.random.multivariate_normal(mu_1.flatten(), 3.1*np.eye(2), size=[n_clusterpoints])
x_2=np.random.multivariate_normal(mu_2.flatten(), 3.1*np.eye(2), size=[n_clusterpoints])
x_3=np.random.multivariate_normal(mu_3.flatten(), 3.1*np.eye(2), size=[n_clusterpoints])

x=np.vstack((x_1,x_2,x_3))
x[0,:]=0

# ii) Distance matrix

dist_mat= distance_matrix(x,x)


# iii) Data model

data = {}
data['distance_matrix'] = dist_mat
data['num_vehicles'] = 3
data['depot'] = 0



"""
    3. Formulate the Optimization problem ------------------------------------
"""


# i) Set up the subroutines

manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]


transit_callback_index = routing.RegisterTransitCallback(distance_callback)


# ii) Add info and constraints

# Travel distances
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Distance constraint
dimension_name = 'Distance'
routing.AddDimension(transit_callback_index, 0, 100, True, dimension_name)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# Setting first solution heuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


# ii) Solve the problem

solution = routing.SolveWithParameters(search_parameters)




"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Assemble the tours

route_list=[]

for vehicle_id in range(data['num_vehicles']):
    temp_route=[0]
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    route_distance = 0
    
    
    # ii) Loop through the tour
    
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        temp_route+=[index]
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
        
        
    # iii) Print out results
    
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
    temp_route[-1]=0
    route_list+=[temp_route]
    print(plan_output)


# iv) Coordinaes of the routes

full_tour=[]

for k in range(data['num_vehicles']):
    temp_location=np.empty([0,2])
    for l in range(len(route_list[k])):
        temp_location=np.vstack((temp_location, x[route_list[k][l],:]))
    full_tour+=[temp_location]



"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Figure displaying distance marix

plt.figure(1,dpi=300)
plt.imshow(dist_mat)
plt.title('The distance matrix')
plt.xlabel('Point nr')
plt.ylabel('Point nr')


# ii) Figure displaying the final routes

plt.figure(2,dpi=300)
plt.scatter(x[:,0],x[:,1],color='k')
plt.title('Point distribution and routes')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

plt.plot(full_tour[0][:,0], full_tour[0][:,1], color='k',linestyle='-',label='Tour 1')
plt.plot(full_tour[1][:,0], full_tour[1][:,1], color='k',linestyle='--',label='Tour 2')
plt.plot(full_tour[2][:,0], full_tour[2][:,1], color='k',linestyle=':',label='Tour 3')
plt.legend()



