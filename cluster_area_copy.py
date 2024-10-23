import random
import math
import time

start_time = time.time()


    

def area_optimisation(gate_dimensions,cluster):

    sorted_cluster = sorted(cluster, key=lambda gate: max(gate_dimensions[gate][0],gate_dimensions[gate][1]), reverse=True)
    # print("sorted cluster is ",sorted_cluster)
    def make_two_new_empty_spaces(L, index, key , w, h, bounding_box):
        temp_w = L[index][0]
        temp_h = L[index][1]
        (x, y) = (L[index][2], L[index][3])
        gates[key] = (x,y)
        if abs((temp_h * (temp_w - w)) - (w * (temp_h - h))) > abs(
                (temp_w * (temp_h - h)) - (h * (temp_w - w))):
            if (temp_h * temp_w - w) >= (w * temp_h - h):
                smaller_space = (w, temp_h - h,x,y+h)
                larger_space = ( temp_w - w, temp_h,x+w, y)

            else:
                larger_space = ( w, temp_h - h, x, y+h)
                smaller_space = ( temp_w - w, temp_h, x+w, y)
        else:
            if (temp_w * temp_h - h) >= (h * temp_w - w):
                smaller_space = (temp_w - w, h, x+w, y)
                larger_space = (temp_w, temp_h - h, x, y+h)
            else:
                larger_space = (temp_w - w, h, x+w, y)
                smaller_space = (temp_w, temp_h - h, x,y+h)
        i = index+1
        n = len(L)
        while i<n and L[i][0]*L[i][1] > larger_space[0]*larger_space[1]:
            L[i-1]= L[i]
            i+=1
        L[i-1] = larger_space
        L.append(smaller_space)
        i = n-1
        while i>=0 and L[i][0]*L[i][1] < smaller_space[0]*smaller_space[1]:
            L[i+1]= L[i]
            i-=1
        L[i+1] = smaller_space
    def add_new_block_up(L,key, width_block, height_block, bounding_box):
        width_reached = bounding_box[0]
        height_reached = bounding_box[1]
        gates[key] = (0, height_reached)
        if width_block<width_reached:
            new_block= (width_reached-width_block, height_block, width_block , height_reached)
        else :
            new_block= (-width_reached+width_block, height_reached, width_reached , 0)
        new_block_area= new_block[0]*new_block[1]
        n = len(L)
        i = n-1
        L.append(new_block)
        while i >= 0 and L[i][0]*L[i][1] < new_block_area:
            L[i+1]=L[i]
            i-=1
        L[i+1] = new_block
        bounding_box[1] = height_block+ height_reached 
        bounding_box[0]=max(width_block,width_reached)

    def add_new_block_right(L,key, width_block, height_block, bounding_box):
        width_reached= bounding_box[0]
        height_reached= bounding_box[1]
        gates[key] = (width_reached,0)
        if height_block<height_reached:
            new_block= (width_block, height_reached-height_block, width_reached , height_block)
        else :
            new_block= (width_reached, -height_reached+height_block, 0 , height_reached)
        new_block_area= new_block[0]*new_block[1]
        n = len(L)
        i = n-1
        L.append(new_block)
        while i >= 0 and L[i][0]*L[i][1] < new_block_area:
            L[i+1]=L[i]
            i-=1
        L[i+1] = new_block
        bounding_box[0] =  width_reached+width_block
        bounding_box[1] = max(height_block,height_reached)

    def add_new_block (L , key , width_block, height_block , bounding_box):
        width_reached = bounding_box[0]
        height_reached = bounding_box[1]
        width_small = (width_block <= width_reached)
        height_small = (height_block <= height_reached)
      
        if width_small:
            extra_space_if_added_above= height_block*(width_reached-width_block)
        else :
            extra_space_if_added_above= height_reached*(width_block-width_reached)
        if height_small:
            extra_space_if_added_right = width_block*(height_reached-height_block)
        else :
            extra_space_if_added_right= width_reached*(height_block-height_reached)
        
        if extra_space_if_added_above<extra_space_if_added_right:
            add_new_block_up(L,key,width_block,height_block,bounding_box)
        else :
            add_new_block_right(L,key,width_block,height_block,bounding_box)

    def find_fitting_node ( L, width_block, height_block):
        n = len(L)
        i = n-1
        while i >= 0 :
            if L[i][0] >= width_block and L[i][1] >= height_block :
                return i
            else :
                i -= 1
        if i < 0:
            return -1

    gates={}
    key = sorted_cluster[0]
    # print(gate_dimensions)
    L = []
    L.append((gate_dimensions[key][0],gate_dimensions[key][1],0,0))
    bounding_box = [gate_dimensions[key][0], gate_dimensions[key][1]]
    for key in sorted_cluster:
        w = gate_dimensions[key][0]
        h = gate_dimensions[key][1]
        fit_index= find_fitting_node(L , w, h)
        if fit_index == -1:
            add_new_block (L , key ,w , h, bounding_box)
        else :
            make_two_new_empty_spaces(L, fit_index , key , w, h, bounding_box )
    # print(gates)
    return gates,bounding_box



# Parse the input to extract gate details
def parse_input(input_data):
    gates = {}
    wires = []
    lines = input_data.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].split()
        if line[0] == "pins":
            gate_name = line[1]
            pin_coordinates = [(int(line[j]), int(line[j + 1])) for j in range(2, len(line), 2)]
            gates[gate_name]["pins"] = pin_coordinates
        elif line[0] == "wire":
            wires.append((line[1], line[2]))
        else:
            gate_name = line[0]
            width, height = int(line[1]), int(line[2])
            gates[gate_name] = {"width": width, "height": height, "pins": []}
        i += 1
    return gates, wires


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        # Path compression optimization
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by rank optimization
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x


def calculate_wire_length(gates, placements, pin_groups):
    """
    Takes a group from pin_groups, pick co-ordinates of gates present in that group from gates dictionary,
    fin max_x_coordinate, min_x_coordinate, max_y_coordinate, min_y_coordinate
    return max_x - min_x + max_y-min_y
    """
    semi_perimeter=0
    for group in pin_groups:
        
        max_x = max_y = 0
        min_x = min_y = 100000 #1e5
        for gate_pin in pin_groups[group]:
            gate=gate_pin.split('.')[0]
            # print(gate, 'abcd')
            # print()
            pin=gate_pin.split('.')[1]
            pin_index=int(pin[1:])
            x_gate, y_gate = placements[gate]
            x = x_gate + get_pin_coordinates(gates, gate, pin_index)[0] 
            y= y_gate + get_pin_coordinates(gates, gate, pin_index)[1]
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            max_y = max(max_y, y)
            min_y = min(min_y, y)
        semi_perimeter+=(max_x-min_x+max_y-min_y)
    return semi_perimeter

def get_pin_coordinates(gates, gate, pin_index):
    return gates[gate]["pins"][pin_index - 1]

def group_connected_pins(wires):
    uf = UnionFind()

    # Add all pins to the Union-Find structure and unite connected pins
    for (pin1, pin2) in wires:
        uf.add(pin1)
        uf.add(pin2)
        uf.union(pin1, pin2)

    # Group pins by their root
    groups = {}
    for pin in uf.parent:
        root = uf.find(pin)
        if root not in groups:
            groups[root] = []
        groups[root].append(pin)

    return groups

# Initial placement strategy
def calculate_gate_degrees(gates, wires):
    degrees = {gate: 0 for gate in gates}
    for wire in wires:
        g1_pin, g2_pin = wire[0].split('.'), wire[1].split('.')
        g1, g2 = g1_pin[0], g2_pin[0]
        degrees[g1] += 1
        degrees[g2] += 1
    return degrees


# Group connected gates together for prioritizing them in the sorted list
def group_connected_gates(wires,gates):
    adjacency_list = {gate: set() for gate in gates}

    # Build adjacency list from the wires
    for wire in wires:
        g1_pin, g2_pin = wire[0].split('.'), wire[1].split('.')
        g1, g2 = g1_pin[0], g2_pin[0]
        adjacency_list[g1].add(g2)
        adjacency_list[g2].add(g1)

    # Perform a DFS/BFS to group connected gates together
    visited = set()
    connected_groups = []

    def dfs(gate, group):
        visited.add(gate)
        group.append(gate)
        for neighbor in adjacency_list[gate]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for gate in gates:
        if gate not in visited:
            group = []
            dfs(gate, group)
            connected_groups.append(group)

    return connected_groups

def split_connected_groups(connected_groups, max_size=10):
    """
    Split large connected groups into smaller pieces, each with at most `max_size` gates.
    
    Args:
    - connected_groups: List of lists, where each sublist contains connected gates.
    - max_size: Maximum size of each group (default is 10).
    
    Returns:
    - List of smaller groups with each group containing at most `max_size` gates.
    """
    smaller_groups = []
    
    for group in connected_groups:
        # Split the group into chunks of size `max_size`
        for i in range(0, len(group), max_size):
            smaller_groups.append(group[i:i + max_size])
    
    return smaller_groups


# Improved initial placement strategy based on connections and degrees
def initial_placement(gates, wires):


    # Calculate degrees of each gate (number of connections)
    degrees = calculate_gate_degrees(gates, wires)

    # Group gates based on direct connections (to place them closer)
    connected_groups = group_connected_gates(wires,gates)
    small_clusters = split_connected_groups(connected_groups)
    x_reached = 100
    y_reached = 0
    placements = {}
    area_placements=[]
    cluster_as_gate = {}
    gate_dimensions = {}
    placements_of_cluster_as_gate = {}
    for cc in gates:
        gate_dimensions[cc] = (gates[cc]["width"],gates[cc]["height"])
    i = 0
    for cluster in small_clusters:
        placements_cluster,bb = area_optimisation(gate_dimensions,cluster)
        cluster_as_gate[len(area_placements)] = (bb[0],bb[1])
        area_placements.append(placements_cluster)
        
    placements_of_cluster_as_gate , bounding_box = area_optimisation(cluster_as_gate,cluster_as_gate)
    for gg in cluster_as_gate:
        x_offset = placements_of_cluster_as_gate[gg][0]
        y_offset = placements_of_cluster_as_gate[gg][1]
        for actual_gate in area_placements[gg]:
            placements[actual_gate] = (area_placements[gg][actual_gate][0]+x_offset,area_placements[gg][actual_gate][1]+y_offset)
    return placements_of_cluster_as_gate,area_placements,placements,cluster_as_gate


# Check if two gates overlap
def check_overlap(gate1, pos1, gate2, pos2, gates):
    g1_x, g1_y = pos1
    g2_x, g2_y = pos2
    g1_width, g1_height = gates[gate1]["width"], gates[gate1]["height"]
    g2_width, g2_height = gates[gate2]["width"], gates[gate2]["height"]

    if g1_x+g1_width<=g2_x or g2_x+g2_width<=g1_x or g1_y+g1_height<=g2_y or g2_y+g2_height<=g1_y:
        return False
    return True

def check_overlap_clusters(gate1, pos1, gate2, pos2, gates):
    g1_x, g1_y = pos1
    g2_x, g2_y = pos2
    g1_width, g1_height = gates[gate1][0], gates[gate1][1]
    g2_width, g2_height = gates[gate2][0], gates[gate2][1]
    if g1_x+g1_width<=g2_x or g2_x+g2_width<=g1_x or g1_y+g1_height<=g2_y or g2_y+g2_height<=g1_y:
        return False
    return True


# Ensure no overlaps after moving a gate
def is_valid_move(gate_to_move, new_position, placements, gates):
    for other_gate, other_position in placements.items():
        if other_gate != gate_to_move:
            if check_overlap(gate_to_move, new_position, other_gate, other_position, gates):
                return False
    return True
def is_valid_move_clusters(gate_to_move, new_position, placements, gates):
    for other_gate, other_position in placements.items():
        if other_gate != gate_to_move:
            if check_overlap_clusters(gate_to_move, new_position, other_gate, other_position, gates):
                return False
    return True


# Simulated Annealing with overlap check
def optimize_placement(gates, wires, initial_placement,pin_groups,it_inner):
    current_placement = initial_placement
    current_cost = calculate_wire_length(gates, current_placement,pin_groups)
    best_placement = current_placement.copy()
    best_cost = current_cost
    T = 1000  # Initial temperature
    alpha = 0.95  # Cooling rate

    pertubation_size = 30
    for iteration in range(it_inner):
        if iteration>1000:
            pertubation_size=26
        if iteration>2000:
            pertubation_size=22
        if iteration>3000:
            pertubation_size=18
        if iteration>4000:
            pertubation_size=14
        if iteration>5000:
            pertubation_size=10
        if iteration>6000:
            pertubation_size=8
        if iteration>7000:
            pertubation_size=6
        if iteration>8000:
            pertubation_size=4
        if iteration>9000:
            pertubation_size=2
        new_placement = current_placement.copy()

        # Randomly move one gate to a new position
        gate_to_move = random.choice(list(gates.keys()))
        delta_x, delta_y = random.randint(0, pertubation_size), random.randint(0, pertubation_size)
        new_x = current_placement[gate_to_move][0]+delta_x-pertubation_size//2
        new_y = current_placement[gate_to_move][1]+delta_y-pertubation_size//2

        # Check for overlap before accepting the new position
        if is_valid_move(gate_to_move, (new_x, new_y), new_placement, gates):
            new_placement[gate_to_move] = (new_x, new_y)

            # Calculate new cost
            new_cost = calculate_wire_length(gates, new_placement,pin_groups)
            delta_cost = new_cost - current_cost

            if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / T):
            # if delta_cost < 0 :
                current_placement = new_placement
                current_cost = new_cost
            if new_cost<best_cost:
                best_cost=new_cost
                best_placement=new_placement.copy()
        T *= alpha  # Cool down the temperature

    return best_placement, best_cost


def optimize_placement_of_clusters(gates,cluster_as_gate, wires, placements_of_cluster_as_gate,area_placements,initial_placement,pin_groups,it_inner):
    current_placement = initial_placement.copy()
    current_cost = calculate_wire_length(gates, current_placement,pin_groups)
    current_placement_cluster = placements_of_cluster_as_gate.copy()
    best_placement = current_placement.copy()
    best_placement_cluster = current_placement_cluster.copy()
    best_cost = current_cost
    n = len(cluster_as_gate)
    T = 1000  # Initial temperature
    alpha = 0.95  # Cooling rate
    pertubation_size = 30
    # change_after_every = 500
    # change_amount = 2
    for iteration in range(it_inner):
        if iteration>1000:
            pertubation_size=26
        if iteration>2000:
            pertubation_size=22
        if iteration>3000:
            pertubation_size=18
        if iteration>4000:
            pertubation_size=14
        if iteration>5000:
            pertubation_size=10
        if iteration>6000:
            pertubation_size=8
        if iteration>7000:
            pertubation_size=6
        if iteration>8000:
            pertubation_size=4
        if iteration>9000:
            pertubation_size=2
        new_placement = current_placement.copy()
        new_placement_cluster = current_placement_cluster.copy()
        # Randomly move one gate to a new position
        gate_to_move = random.choice(list(cluster_as_gate.keys()))
        delta_x, delta_y = random.randint(0, pertubation_size), random.randint(0, pertubation_size)
        new_x = delta_x-pertubation_size//2
        new_y = delta_y-pertubation_size//2
        gg = gate_to_move
        # print("area placements gg ",area_placements[gg])
        for gate_name in area_placements[gg]:
            
                new_placement[gate_name] = (current_placement[gate_name][0]+new_x,current_placement[gate_name][1]+new_y)
        new_placement_cluster[gg] = (current_placement_cluster[gg][0]+new_x,current_placement_cluster[gg][1]+new_y)
        if is_valid_move_clusters(gate_to_move, new_placement_cluster[gg], new_placement_cluster, cluster_as_gate):
            # new_placement[gate_to_move] = (new_x, new_y)

            # Calculate new cost
            new_cost = calculate_wire_length(gates, new_placement,pin_groups)
            delta_cost = new_cost - current_cost

            if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / T):
            # if delta_cost < 0 :
                current_placement = new_placement
                current_placement_cluster=new_placement_cluster
                current_cost = new_cost

            if new_cost<best_cost:
                best_cost=new_cost
                best_placement=new_placement.copy()
                best_placement_cluster=new_placement_cluster.copy()

        T *= alpha  # Cool down the temperature

    return best_placement,best_placement_cluster, best_cost



def final_optimisation(gates, wires, initial_placement,pin_groups,it_inner):
    current_placement = initial_placement
    current_cost = calculate_wire_length(gates, current_placement,pin_groups)
    best_placement = current_placement.copy()
    best_cost = current_cost
    T = 1000  # Initial temperature
    alpha = 0.95  # Cooling rate
    for iteration in range(it_inner):
        new_placement = current_placement.copy()

        # Randomly move one gate to a new position
        gate_to_move = random.choice(list(gates.keys()))
        delta_x, delta_y = random.randint(0, 2), random.randint(0, 2)
        new_x = current_placement[gate_to_move][0]+delta_x-1
        new_y = current_placement[gate_to_move][1]+delta_y-1

        # Check for overlap before accepting the new position
        if is_valid_move(gate_to_move, (new_x, new_y), new_placement, gates):
            new_placement[gate_to_move] = (new_x, new_y)

            # Calculate new cost
            new_cost = calculate_wire_length(gates, new_placement,pin_groups)
            delta_cost = new_cost - current_cost

            if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / T):
            # if delta_cost < 0 :
                current_placement = new_placement
                current_cost = new_cost
            if new_cost < best_cost:
                best_cost=new_cost
                best_placement=new_placement.copy()
        T *= alpha  # Cool down the temperature

    return best_placement, best_cost


# Main function
def main(input_data):
    gates, wires = parse_input(input_data)
    pin_groups = group_connected_pins(wires)

    number_of_gates = len(gates)
    number_of_pins = len(wires)

    if number_of_gates<=500 and number_of_pins<=2000:
        it_inner = 10000
        it_outer = 10
    else :
        it_inner = 1000
        it_outer = 10
    placements_of_cluster_as_gate,area_placements,placements,cluster_as_gate=initial_placement(gates,wires)
   
    placements,placements_of_cluster_as_gate,cost = optimize_placement_of_clusters(gates,cluster_as_gate,wires,placements_of_cluster_as_gate,area_placements,placements,pin_groups,it_inner)
    for i in range(0,it_outer):
        new_placement_g,new_placement_c,new_cost_c=optimize_placement_of_clusters(gates,cluster_as_gate,wires,placements_of_cluster_as_gate,area_placements,placements,pin_groups,it_inner)
        if new_cost_c<cost:
            cost = new_cost_c
            placements=new_placement_g.copy()
            placements_of_cluster_as_gate=new_placement_c.copy()

    optimized_placements1, final_cost1 = optimize_placement(gates, wires, placements,pin_groups,it_inner)
    x = 10
    for i in range(0,it_outer):
        new_placement,new_cost = optimize_placement(gates,wires,placements,pin_groups,it_inner)
        if new_cost<final_cost1:
            final_cost1= new_cost
            optimized_placements1=new_placement.copy()

    optimized_placements = optimized_placements1
    final_cost = final_cost1
    x = 10
    number_of_gates = len(gates)
    if (number_of_gates<=10):
        x = 1000
    if (number_of_gates <= 100):
        x = 100
    if (number_of_gates <= 1000):
        x = 10
    for i in range(0,it_outer):
        new_placement,new_cost = final_optimisation(gates,wires,optimized_placements1,pin_groups,it_inner)
        if new_cost<final_cost:
            final_cost= new_cost
            optimized_placements=new_placement.copy()

    key = list(optimized_placements.keys())[0]
    min_x = optimized_placements[key][0]
    min_y = optimized_placements[key][1]
    max_x = optimized_placements[key][0]
    max_y = optimized_placements[key][1]
    for gate, (x, y) in optimized_placements.items():
        # print(f"{gate} {x} {y}")
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y )

    for key in optimized_placements:
        optimized_placements[key] = (optimized_placements[key][0]-min_x,optimized_placements[key][1]-min_y)
    # Output the final placements
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    for gate, (x, y) in optimized_placements.items():
        # print(f"{gate} {x} {y}")
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + gates[gate]["width"])
        max_y = max(max_y, y + gates[gate]["height"])
    bounding_box1 = (max_x , max_y )
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    for gate, (x, y) in placements.items():
        # print(f"{gate} {x} {y}")
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + gates[gate]["width"])
        max_y = max(max_y, y + gates[gate]["height"])
    bounding_box2 = (max_x, max_y)
    with open("output.txt", "w") as file:
        file.write("bounding_box " + str(bounding_box1[0]) + " " + str(bounding_box1[1]) + "\n")
        for key in optimized_placements:
            file.write(key + " " + str(optimized_placements[key][0]) + " " + str(optimized_placements[key][1]) + "\n")
        file.write("wire_length " + str(final_cost)+"\n")
    # for gate, (x, y) in optimized_placements.items():
        # print(f"{gate} {x} {y}")

    with open("output1.txt", "w") as file:
        file.write("bounding_box " + str(bounding_box2[0]) + " " + str(bounding_box2[1]) + "\n")
        for key in placements:
            file.write(key + " " + str(int(placements[key][0])) + " " + str(int(placements[key][1])) + "\n")

    print(final_cost)

if __name__ == "__main__":
    # Open and read input.txt
    with open("input.txt", "r") as file:
        input_data = file.read()

    # Call the main function with the input data
    main(input_data)

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")