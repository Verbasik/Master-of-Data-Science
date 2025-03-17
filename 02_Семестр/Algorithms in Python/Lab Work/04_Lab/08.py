def solve():
    n, m, s, A, B = map(int, input().split())
    a_costs = list(map(int, input().split()))
    b_costs = list(map(int, input().split()))

    a_costs.sort(reverse=True)
    b_costs.sort(reverse=True)

    prefix_sum_a = [0] * (n + 1)
    for i in range(n):
        prefix_sum_a[i+1] = prefix_sum_a[i] + a_costs[i]

    prefix_sum_b = [0] * (m + 1)
    for i in range(m):
        prefix_sum_b[i+1] = prefix_sum_b[i] + b_costs[i]

    max_total_cost = 0

    max_count_a_possible = min(n, s // A) if A > 0 else n
    if A == 0 and n > 0:
        current_cost_a = prefix_sum_a[n]
        remaining_weight = s
        max_count_b = remaining_weight // B if B > 0 else m
        count_b = min(max_count_b, m)
        current_cost_b = prefix_sum_b[count_b]
        max_total_cost = max(max_total_cost, current_cost_a + current_cost_b)

    for count_a in range(min(n, s // A) + 1 if A > 0 else 1):
        current_weight_a = count_a * A
        if current_weight_a > s:
            continue

        current_cost_a = prefix_sum_a[count_a]

        remaining_weight = s - current_weight_a
        max_count_b = remaining_weight // B if B > 0 else m
        count_b = min(max_count_b, m)

        current_cost_b = prefix_sum_b[count_b]

        total_cost = current_cost_a + current_cost_b
        max_total_cost = max(max_total_cost, total_cost)

    print(max_total_cost)

if __name__ == "__main__":
    solve()