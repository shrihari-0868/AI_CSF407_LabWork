"""
Breadth-first search for the warehouse robot navigation problem.

Artificial Intelligence -- Laboratory Exercise: Search and A*

This is the blind-search counterpart to astar.py, used for the Task 5
comparison.  The search problem is identical -- same map, same states,
same actions, same unit costs.  Only the frontier discipline changes:

    A*   min-heap ordered by f(n) = g(n) + h(n)
    BFS  FIFO queue, no cost or heuristic information at all

Because every action costs 1, BFS is also guaranteed to return a
shortest path, which makes it a useful oracle for checking A*.

Running this file prints the BFS results and the Task 5 comparison table.
"""

from collections import deque

from astar import (
    WAREHOUSE,
    TRIVIAL,
    NO_SOLUTION,
    ALTERNATIVE_PATHS,
    BARRIER_ROOM,
    TRAP_ROOM,
    astar,
    parse_map,
    successors,
    reconstruct_path,
    report,
)


def bfs(text):
    """Run breadth-first graph search on an ASCII map.

    Returns the same result dictionary shape as astar(), so the two can be
    compared directly.
    """
    grid, start, goal = parse_map(text)

    frontier = deque([start])
    visited = {start}            # states generated (BFS never needs to revisit)
    came_from = {}
    action_taken = {}
    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        # BFS may safely test for the goal on expansion; with unit costs the
        # first time the goal is dequeued it is already at minimum depth.
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
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                action_taken[nxt] = name
                frontier.append(nxt)

    return {
        "found": False,
        "path": None,
        "path_length": None,
        "states_expanded": states_expanded,
        "actions": None,
    }


def compare(name, text):
    """Print one row of the Task 5 comparison table."""
    b = bfs(text)
    a = astar(text)
    print("%-26s | %-9s %-9s | %-9s %-9s | %-9s %-9s" % (
        name,
        "yes" if b["found"] else "NO",
        "yes" if a["found"] else "NO",
        b["path_length"], a["path_length"],
        b["states_expanded"], a["states_expanded"],
    ))
    return b, a


def run_comparison():
    print("### BFS ON THE ORIGINAL WAREHOUSE ###\n")
    report("BFS -- Original warehouse", WAREHOUSE, bfs(WAREHOUSE))

    print("### TASK 5: BFS vs A* ###\n")
    header = "%-26s | %-19s | %-19s | %-19s" % (
        "map", "solution found", "path length", "states expanded")
    print(header)
    print("%-26s | %-9s %-9s | %-9s %-9s | %-9s %-9s" % (
        "", "BFS", "A*", "BFS", "A*", "BFS", "A*"))
    print("-" * len(header))
    results = {
        "Original warehouse": compare("Original warehouse", WAREHOUSE),
        "Trivial": compare("Trivial", TRIVIAL),
        "No solution": compare("No solution", NO_SOLUTION),
        "Alternative paths": compare("Alternative paths", ALTERNATIVE_PATHS),
        "Barrier room": compare("Barrier room", BARRIER_ROOM),
        "Trap room": compare("Trap room", TRAP_ROOM),
    }
    print()

    # Cross-check: on unit-cost maps BFS is optimal, so A* must match it.
    print("### OPTIMALITY CROSS-CHECK (A* length == BFS length?) ###\n")
    for name, (b, a) in results.items():
        ok = (b["found"] == a["found"]) and (b["path_length"] == a["path_length"])
        print("%-26s %s" % (name, "OK" if ok else "MISMATCH"))
    print()


if __name__ == "__main__":
    run_comparison()
