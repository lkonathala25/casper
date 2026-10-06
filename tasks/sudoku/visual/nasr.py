def compute_reward(solution_board,final_solution,ground_truth_board):
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    if final_solution == ground_truth_board:
        reward = 10
    else:
        reward = 0
    partial_reward = 0.0
    for i,j in zip(final_solution,ground_truth_board):
        if i == j:
            partial_reward+=1
    reward += partial_reward/81
    return reward