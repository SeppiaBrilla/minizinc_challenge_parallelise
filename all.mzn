include "globals.mzn";
array[1..14] of int: cost = [0, 1, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 1, 1]; %pass, cp-sat,1 cp-sat,8 choco,1 choco,2 choco,4 choco,8 CPLEX,1 CPLEX,2 CPLEX,4 CPLEX,8 chuffed,1, Picat,1, Huub,1

array[1..14] of var 1..14: choice;

constraint all_different_except(choice, {1});
constraint increasing(choice);

constraint sum(i in 1..14)(cost[choice[i]]) <= 8;
constraint sum(i in 1..14)(cost[choice[i]]) >= 1;

solve satisfy;
