import sys

# Faster I/O
readline = sys.stdin.readline
write = sys.stdout.write

# Increase recursion depth for find, although path compression helps significantly
# Set based on max possible grid size (1000*1000 = 1,000,000) + buffer
try:
    # Setting recursion depth can fail in some environments, wrap in try-except
    sys.setrecursionlimit(1001 * 1001 + 50)
except Exception:
    pass # Continue if setting limit fails

# Global DSU arrays (or pass them around if preferred)
N, M = 0, 0
parent = []
rank = []
has_S = []
has_X = []
is_ship_cell = [] # Optimization: Track if a cell is part of *any* ship

def find(i):
    # Find with path compression
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    # Union by rank with metadata merging
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        # Determine which root becomes the parent
        if rank[root_i] < rank[root_j]:
            root_i, root_j = root_j, root_i # Ensure root_i is the new parent (or equal rank)

        # Merge j into i
        parent[root_j] = root_i

        # Merge metadata: new parent inherits properties of both children
        has_S[root_i] = has_S[root_i] or has_S[root_j]
        has_X[root_i] = has_X[root_i] or has_X[root_j]

        # Update rank if they were equal
        if rank[root_i] == rank[root_j]:
            rank[root_i] += 1
        return True # Indicates a merge happened
    return False # Already in the same set

def solve():
    global N, M, parent, rank, has_S, has_X, is_ship_cell
    try:
        line1 = readline().split()
        if len(line1) != 2: raise ValueError("Format error")
        n_str, m_str = line1
        if not n_str.isdigit() or not m_str.isdigit(): raise ValueError("N, M not digits")
        N, M = int(n_str), int(m_str)

        if not (1 <= N <= 1000 and 1 <= M <= 1000): raise ValueError("N, M bounds")

        # --- Attempt to read grid efficiently ---
        # Reading line by line might be slightly better than list comprehension
        grid_lines = []
        for _ in range(N):
            grid_lines.append(readline().strip())
        # If grid_lines itself causes MLE, this problem is likely unsolvable
        # in standard Python within typical competitive programming limits.

    except Exception:
        exit(1) # Exit silently on input error

    # --- DSU Initialization ---
    num_cells = N * M
    parent = list(range(num_cells))
    rank = [0] * num_cells
    has_S = [False] * num_cells
    has_X = [False] * num_cells
    is_ship_cell = [False] * num_cells # Track ship cells

    # --- Populate initial DSU data (First Pass) ---
    for r in range(N):
        row = grid_lines[r] # Access the read line
        for c in range(M):
            idx = r * M + c
            char = row[c]
            if char == 'S':
                has_S[idx] = True
                is_ship_cell[idx] = True
            elif char == 'X':
                has_X[idx] = True
                is_ship_cell[idx] = True
            # '-' cells have all flags False

    # --- Perform Unions (Second Pass) ---
    for r in range(N):
        for c in range(M):
            idx1 = r * M + c
            if is_ship_cell[idx1]: # Only process ship cells
                # Check neighbor below (if exists and is a ship cell)
                if r + 1 < N:
                    idx2 = (r + 1) * M + c
                    if is_ship_cell[idx2]:
                        union(idx1, idx2)
                # Check neighbor right (if exists and is a ship cell)
                if c + 1 < M:
                    idx2 = r * M + (c + 1)
                    if is_ship_cell[idx2]:
                        union(idx1, idx2)

    # --- Count Components (Third Pass) ---
    whole_count = 0
    damaged_count = 0
    destroyed_count = 0
    # Use the parent array to track processed roots efficiently
    # Mark roots as processed by setting parent[root] = -1 (or similar)
    # Or use a separate set if preferred, but modifying parent saves memory
    processed_roots = set()

    for r in range(N):
        for c in range(M):
            idx = r * M + c
            if is_ship_cell[idx]: # Only consider ship cells
                root = find(idx)
                if root not in processed_roots:
                    processed_roots.add(root)
                    is_S = has_S[root]
                    is_X = has_X[root]

                    # Classify based on the merged properties at the root
                    if is_S and not is_X:
                        whole_count += 1
                    elif is_S and is_X:
                        damaged_count += 1
                    elif not is_S and is_X: # Only X
                        destroyed_count += 1
                    # else: # Not S and not X - shouldn't happen if is_ship_cell[idx] is true

    # --- Вывод результата ---
    try:
        write(f"{whole_count} {damaged_count} {destroyed_count}\n")
    except Exception:
        exit(1) # Exit silently on output error

# --- Вызов основной функции ---
solve()