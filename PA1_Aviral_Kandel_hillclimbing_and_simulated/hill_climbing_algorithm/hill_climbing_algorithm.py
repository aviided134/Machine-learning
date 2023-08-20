import random
def neighborhood(binary_string)                             # neighbor function
    list_a = list(binary_string)
    main_flipped = []
    #print(list_a)
    for i in range(len(list_a)):
        if list_a[i] == '0':
            flip_bin = list_a.copy()                        # copy to flip_bin if the bit is 0
            flip_bin[i] = '1'                               # copy but make that bit to 1
            main_flipped.append(''.join(flip_bin))          # append the rest of the bits
            
        else:  # same for the 0 bit
            flip_bin = list_a.copy()
            flip_bin[i] = '0'
            main_flipped.append(''.join(flip_bin)) 
    main_flipped.append(''.join(binary_string))             # added all the neighboring strings into main_flipped list
    return main_flipped
    
def fitness(binary):
    ones = binary.count('1')
    return abs(13 * ones - 170)
    
def hill_climbing_algorithm(binary_bit):  
    current = binary_bit                                     # intitially both current and best bits are the binary bits
    best = binary_bit
    while True:
        #print(current)     
        neighbors = neighborhood(current)                    # found neighbors by calling the user defined function
        #print(neighbors)
        #print(neighbors)
        best_neighbor = current         
        best_fitness = fitness(best_neighbor)                #fitness function
        for x in neighbors:                                  # for loop used to find best function for each neighbors
            neighbor_fitness = fitness(x)
            if neighbor_fitness > best_fitness:
                best_neighbor = x
                best_fitness = neighbor_fitness
        if best_fitness> fitness(best):                      # used for improving fitness function to either global or local maxima
            current = best_neighbor
            best = best_neighbor
        else:
            return best
maximum = 0
maximum_list = []
for i in range(100):
    
    binary = "".join(random.choices(["0", "1"], k=40))      # to generate random binary number with 40 bits
    result = hill_climbing_algorithm(binary)
    result_fitness = fitness(result)
    maximum_list.append(result_fitness)
    if result_fitness > maximum:
        maximum = result_fitness
        best_result = result

with open("Output.txt", "w") as file:                       # writing the code into output text file
    file.write(", ".join(str(x) for x in maximum_list))
