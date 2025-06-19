# Minimizing Wire Length in VLSI Circuit Design

This project implements algorithms for optimizing gate placement in VLSI circuit design with the goal of minimizing wire length while efficiently utilizing chip area.

## Project Overview

In VLSI (Very Large Scale Integration) circuit design, placing gates optimally is critical for performance, power consumption, and area utilization. This project focuses on:

1. **Area Optimization**: Efficiently placing gates to minimize the total chip area required
2. **Wire Length Minimization**: Arranging gates to minimize the total wire length between connected components
3. **Visualization**: Providing a visual representation of the gate placement and wire routing

## Data Structures Used

- **Lists and Dictionaries**: For storing gate dimensions, positions, and connectivity information
- **Sorted Collections**: For organizing gates by size during area optimization
- **Geometric Representations**: For tracking available spaces and bounding boxes

## Algorithm Flow

### Area Optimization

1. **Gate Sorting Strategy**:
   - Gates are sorted by their maximum dimension (width or height) in descending order
   - This "largest-first" approach helps minimize wasted space by placing larger, harder-to-fit gates first
   - Sorting uses a custom comparator function to prioritize gates with larger area requirements

2. **Space Management Algorithm**:
   - **Free Space Tracking**: Maintain a list of available rectangular spaces sorted by area
   - **First Fit Strategy**: For each gate, find the first available space where it fits
   - **Space Division**: When a gate is placed in a space:
     - The original space is removed from the available spaces list
     - The remaining area is split into two new rectangular spaces
     - New spaces are inserted back into the sorted list based on their areas
   - **Space Merging Heuristic**: Adjacent spaces are merged when possible to create larger contiguous areas

3. **Intelligent Space Division**:
   - When placing a gate in a space, the algorithm decides how to split the remaining area:
     - Compare different splitting options (horizontal vs. vertical cuts)
     - Choose the option that minimizes wasted area
     - Prioritize creating more useful rectangle shapes over thin, hard-to-use strips

4. **Boundary Extension**:
   - If no suitable space exists for a gate, the algorithm extends the boundary:
     - **Add Block Up**: Extends the layout upward and creates new spaces
     - **Add Block Right**: Extends the layout rightward and creates new spaces
     - Chooses extension direction based on which minimizes wasted area
   - Updates bounding box coordinates to reflect the new circuit dimensions

5. **Bounding Box Optimization**:
   - Continuously tracks the minimum bounding box that contains all placed gates
   - Updates dimensions incrementally as gates are placed
   - Calculates the total area utilization as a performance metric

### Wire Length Optimization

1. **Connectivity Graph Construction**:
   - Build an adjacency matrix/list representing connections between gates
   - Weight connections based on the number of wires between gates
   - Identify highly connected gate clusters

2. **Gate Clustering Algorithm**:
   - **Connectivity Analysis**: Calculate connectivity scores between all gate pairs
   - **Cluster Formation**: Group gates with strong connectivity using hierarchical clustering
   - **Cluster Assignment**: Prioritize placing gates from the same cluster in proximity

3. **Wire Length Calculation**:
   - For each wire connection:
     - Determine the pin positions on connected gates
     - Calculate Manhattan distance between connected pins
     - Sum all wire lengths to get total wire length metric

4. **Placement Refinement**:
   - **Iterative Improvement**: Use local search to optimize positions
     - Swap positions of gate pairs and evaluate impact on wire length
     - Accept swaps that reduce total wire length
   - **Simulated Annealing Approach**: 
     - Initially accept some sub-optimal swaps to escape local minima
     - Gradually decrease acceptance probability of worse solutions
     - Continue until convergence criteria are met

5. **Multi-objective Optimization**:
   - Balance trade-offs between area utilization and wire length
   - Use weighted cost function to guide optimization process
   - Apply constraints to ensure manufacturability requirements are met

## How to Run

### Prerequisites

- Python 3.x
- Required libraries: 
  - PIL (Pillow)
  - tkinter
  - math
  - random

### Execution

1. **Run the area optimization and gate placement algorithm**:
   ```
   python cluster_area_copy.py
   ```

2. **Visualize the results**:
   ```
   python visualization.py
   ```

## Input Format

The program expects input data describing:
- Gate dimensions (width and height)
- Pin coordinates on each gate
- Wire connections between pins

## Output

- Coordinates for each gate's placement
- Total area utilized
- Total wire length
- Visual representation of the circuit layout

## Performance Metrics

The solution is evaluated based on:
- Total chip area used
- Total wire length
- Runtime efficiency

## Notes

- The algorithm uses heuristics to approximate the optimal solution as the problem is NP-hard
- Visualization helps in understanding the quality of placement and identifying areas for improvement
- Various trade-offs between area utilization and wire length can be explored by adjusting parameters

## Future Improvements

- Implement more sophisticated placement algorithms (e.g., simulated annealing)
- Add support for timing constraints
- Integrate with standard cell libraries
- Consider thermal issues in placement decisions