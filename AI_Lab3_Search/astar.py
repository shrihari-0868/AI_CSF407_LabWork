"""
A* search for the warehouse robot navigation problem.

Artificial Intelligence -- Laboratory Exercise: Search and A*

Search problem formulation
--------------------------
    S   states        (row, col) integer coordinates of a free cell
    A   actions       Up, Down, Left, Right
    T   transition    (r, c) -> (r + dr, c + dc) if the target cell is
                      inside the grid and is not an obstacle
    s0  initial       the cell marked 'S'
    G   goal          the cell marked 'G'
    c   cost          1 for every action

Running this file executes the full test suite required by Tasks 3 and 6.
"""

import heapq
import math

# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

# Test 2: goal immediately adjacent to the start.
TRIVIAL = """\
#####
#SG##
#####"""

# Test 3: the goal is walled off, so no solution exists.
NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

# Test 4: an open room, so many distinct shortest paths of length 6 exist.
ALTERNATIVE_PATHS = """\
#######
#S....#
#.....#
#....G#
#######"""

# Task 6 supplementary map A: an open room split by a central barrier.
# Open space lets a heuristic prune, which the warehouse maze does not.
BARRIER_ROOM = """\
###############
#.............#
#.....###.....#
#.....#.#.....#
#..S..#.#..G..#
#.....#.#.....#
#.....###.....#
#.............#
###############"""

# Task 6 supplementary map B: the direct-looking route is a trap.
# An inadmissible heuristic commits to it and returns a sub-optimal path.
TRAP_ROOM = """\
###########
#S......#.#
#.........#
#.#.......#
#.#.....#.#
#......#.G#
###########"""

# ---------------------------------------------------------------------------
# Environment representation
# ---------------------------------------------------------------------------

# Fixed action order, so runs are reproducible.
ACTIONS = [
    ("Up", (-1, 0)),
    ("Down", (1, 0)),
    ("Left", (0, -1)),
    ("Right", (0, 1)),
]


def parse_map(text):
    """Return (grid, start, goal) for an ASCII warehouse map.

    grid  -- list of strings, grid[row][col] is one cell
    start -- (row, col) of 'S'
    goal  -- (row, col) of 'G'
    """
    grid = [line for line in text.splitlines() if line]
    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)
    if start is None:
        raise ValueError("map has no start cell 'S'")
    if goal is None:
        raise ValueError("map has no goal cell 'G'")
    return grid, start, goal


def is_free(grid, state):
    """True if `state` is inside the grid and is not an obstacle."""
    r, c = state
    if r < 0 or r >= len(grid):
        return False
    if c < 0 or c >= len(grid[r]):
        return False
    return grid[r][c] != "#"


def successors(grid, state):
    """Yield (action_name, next_state) for every *valid* action in `state`."""
    r, c = state
    for name, (dr, dc) in ACTIONS:
        nxt = (r + dr, c + dc)
        if is_free(grid, nxt):
            yield name, nxt


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def manhattan(state, goal):
    """h(n) = |x - x_G| + |y - y_G|.  Admissible for 4-way unit-cost movement."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero(state, goal):
    """h(n) = 0.  A* degenerates to uniform-cost search."""
    return 0


def euclidean(state, goal):
    """Straight-line distance.  Admissible but weaker than Manhattan here."""
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def double_manhattan(state, goal):
    """h(n) = 2 x Manhattan.  Inadmissible: it can overestimate."""
    return 2 * manhattan(state, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "zero": zero,
    "euclidean": euclidean,
    "double_manhattan": double_manhattan,
}


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------


def reconstruct_path(came_from, goal):
    """Walk the parent pointers back from the goal and reverse the result."""
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar(text, heuristic=manhattan):
    """Run A* on an ASCII map.

    Returns a result dictionary with the keys required by Task 1:
        found, path, path_length, states_expanded, actions
    `path_length` counts *actions*, i.e. len(path) - 1.
    """
    grid, start, goal = parse_map(text)

    # The frontier is a min-heap ordered by f(n) = g(n) + h(n).
    # `counter` breaks ties in FIFO order so that runs are deterministic.
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]

    g_score = {start: 0}
    came_from = {}
    action_taken = {}
    closed = set()          # states already expanded
    states_expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        # A state can sit in the heap more than once; skip stale copies.
        if current in closed:
            continue
        closed.add(current)
        states_expanded += 1

        # Goal test on expansion, not on generation: this is what makes the
        # returned path optimal for an admissible heuristic.
        if current == goal:
            path = reconstruct_path(came_from, goal)
            return {
                "found": True,
                "path": path,
                "path_length": len(path) - 1,
                "states_expanded": states_expanded,
                "actions": [action_taken[s] for s in path[1:]],
            }

        for name, nxt in successors(grid, current):
            if nxt in closed:
                continue
            tentative_g = g_score[current] + 1      # every move costs 1
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                action_taken[nxt] = name
                counter += 1
                heapq.heappush(
                    frontier, (tentative_g + heuristic(nxt, goal), counter, nxt)
                )

    # Frontier exhausted: every reachable state was expanded and none was the
    # goal, so the problem has no solution.
    return {
        "found": False,
        "path": None,
        "path_length": None,
        "states_expanded": states_expanded,
        "actions": None,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def format_path(path, per_line=10):
    """Render a coordinate path over several lines so it stays readable."""
    if path is None:
        return "  (none)"
    chunks = []
    for i in range(0, len(path), per_line):
        chunks.append("  " + " -> ".join(str(p) for p in path[i:i + per_line]))
    return "\n".join(chunks)


def overlay(text, path):
    """Return the map with the solution path drawn using '*'."""
    grid = [list(line) for line in text.splitlines() if line]
    if path:
        for r, c in path[1:-1]:
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def report(name, text, result, show_map=True, show_path=True):
    print("=" * 62)
    print(name)
    print("=" * 62)
    print("Solution found : %s" % ("yes" if result["found"] else "NO"))
    print("Path length    : %s" % result["path_length"])
    print("States expanded: %d" % result["states_expanded"])
    if result["found"] and show_path:
        print("Path:")
        print(format_path(result["path"]))
    if result["found"] and show_map:
        print("Map with path:")
        print(overlay(text, result["path"]))
    print()


# ---------------------------------------------------------------------------
# Task 3 and Task 6 experiments
# ---------------------------------------------------------------------------


def run_tests():
    print("### TASK 3: TESTING THE GENERATED PROGRAM ###\n")
    report("Test 1 -- Original warehouse", WAREHOUSE, astar(WAREHOUSE))
    report("Test 2 -- Trivial case (goal adjacent to start)",
           TRIVIAL, astar(TRIVIAL))
    report("Test 3 -- No solution (goal walled off)",
           NO_SOLUTION, astar(NO_SOLUTION))
    report("Test 4 -- Alternative paths (open room)",
           ALTERNATIVE_PATHS, astar(ALTERNATIVE_PATHS))

    print("### TASK 6: HEURISTIC INVESTIGATION ###\n")
    maps = [
        ("Original warehouse", WAREHOUSE),
        ("Barrier room (supplementary)", BARRIER_ROOM),
        ("Trap room (supplementary)", TRAP_ROOM),
    ]
    for map_name, text in maps:
        print(map_name)
        print("  %-18s %-8s %-13s %s" % ("heuristic", "found", "path length",
                                         "states expanded"))
        print("  " + "-" * 56)
        for label in ("manhattan", "zero", "euclidean", "double_manhattan"):
            res = astar(text, HEURISTICS[label])
            print("  %-18s %-8s %-13s %s" % (
                label,
                "yes" if res["found"] else "NO",
                res["path_length"],
                res["states_expanded"],
            ))
        print()

    # The inadmissible heuristic is the interesting case: show the two paths.
    print("Trap room -- path returned by admissible Manhattan (length %d):"
          % astar(TRAP_ROOM, manhattan)["path_length"])
    print(overlay(TRAP_ROOM, astar(TRAP_ROOM, manhattan)["path"]))
    print("Trap room -- path returned by inadmissible 2h (length %d):"
          % astar(TRAP_ROOM, double_manhattan)["path_length"])
    print(overlay(TRAP_ROOM, astar(TRAP_ROOM, double_manhattan)["path"]))
    print()


if __name__ == "__main__":
    run_tests()
