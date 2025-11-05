include "globals.mzn";
% Use this editor as a MiniZinc scratch book
array[1..13] of int: cost = [0, 1, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 1]; %cp-sat,1 cp-sat,8 gecode,1 gecode,2 gecode,4 gecode,8 CPLEX,1 CPLEX,2 CPLEX,4 CPLEX,8 chuffed,1, Picat,1

array[1..12] of var 1..13: choice;

constraint all_different_except(choice, {1});
constraint increasing(choice);

constraint sum(i in 1..12)(cost[choice[i]]) <= 8;
constraint sum(i in 1..12)(cost[choice[i]]) >= 1;

solve satisfy;
