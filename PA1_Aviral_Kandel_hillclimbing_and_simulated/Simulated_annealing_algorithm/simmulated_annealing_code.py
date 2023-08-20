import random 
import math


def neighborhood(binary_string):                        # generate neighborhood bits by flipping each bit at a time
    list_a = list(binary_string)
    main_flipped = []
    #print(list_a)
    for i in range(len(list_a)-1):
        if list_a[i] == '0':
            flip_bin = list_a.copy()                    # copy to flip_bin if the bit is 0
            flip_bin[i] = '1'                           # copy but make that bit to 1
            main_flipped.append(''.join(flip_bin))      # append the rest of the bits
            
        else:  # same for the 0 bit
            flip_bin = list_a.copy()
            flip_bin[i] = '0'
            main_flipped.append(''.join(flip_bin)) 
    main_flipped.append(''.join(binary_string))
    return main_flipped
    
def fitness(binary):                                    # fitness function
    ones = binary.count('1')
    return abs(14 * ones - 190)

def simulated_annealing_algorithm(binary_bit):          # user defined simulated annealing algorithm
    temp = 20                                           # initial temperature
    cooling_rate = 0.01         
    current = binary_bit
    best = binary_bit
    
    while temp > 1:
        neighbors = neighborhood(current)               # all neighbors are found by calling user defined function
       # print(current)
    #    print(neighbors)
        best_neighbor = best
        best_fitness = fitness(best)
        for x in neighbors:                             # for loop used to find best fitness function in each neighbor
            neighbor_fitness = fitness(x)
            if neighbor_fitness > best_fitness:
                best_neighbor = x
                best_fitness = neighbor_fitness
            elif (random.uniform(0,1) < math.exp((neighbor_fitness - best_fitness) / temp)):   # checks again with probabilistic approach
                best_neighbor = x
                best_fitness = neighbor_fitness
         
        temp = temp * (1-cooling_rate)
        if best_fitness > fitness(best):
            current = best_neighbor
            best = best_neighbor
     

        
    return best

maximum = 0           
maximum_list = []
for i in range(200):
    binary_bit = "".join(random.choices(["0", "1"], k=50))    # generate random binary number with 50 bits
    result = simulated_annealing_algorithm(binary_bit)
    result_fitness = fitness(result)
    
   # print(result_fitness)
    maximum_list.append(result_fitness)
    
    if result_fitness > maximum:
        maximum = result_fitness
        best_result = result
with open("Output.txt", "w") as file:                          # writing the maximum list in the output text
    file.write(", ".join(str(x) for x in maximum_list))

print(maximum_list)