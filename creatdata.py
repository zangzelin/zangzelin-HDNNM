from __future__ import print_function
from itertools import combinations, permutations
# Import Python wrapper for or-tools constraint solver.
from ortools.constraint_solver import pywrapcp
import random
import numpy as np


def main(machines, processing_times):
    # Create the solver.
    solver = pywrapcp.Solver('jobshop')

    machines_count = len(machines)
    jobs_count = len(machines[0])

    all_machines = range(0, machines_count)
    all_jobs = range(0, jobs_count)
    # Define data.
    # print(processing_times)
    # Computes horizon.
    horizon = 0
    for i in all_jobs:
        horizon += sum(processing_times[i])
    # Creates jobs.
    all_tasks = {}
    for i in all_jobs:
        for j in range(0, len(machines[i])):
            all_tasks[(i, j)] = solver.FixedDurationIntervalVar(
                0,  horizon, processing_times[i][j], False, 'Job_%i_%i' % (i, j))

    # Creates sequence variables and add disjunctive constraints.
    all_sequences = []
    all_machines_jobs = []
    for i in all_machines:

        machines_jobs = []
        for j in all_jobs:
            for k in range(0, len(machines[j])):
                if machines[j][k] == i:
                    machines_jobs.append(all_tasks[(j, k)])
        disj = solver.DisjunctiveConstraint(machines_jobs, 'machine %i' % i)
        all_sequences.append(disj.SequenceVar())
        solver.Add(disj)

    # Add conjunctive contraints.
    for i in all_jobs:
        for j in range(0, len(machines[i]) - 1):
            solver.Add(all_tasks[(i, j + 1)].StartsAfterEnd(all_tasks[(i, j)]))

    # Set the objective.
    obj_var = solver.Max([all_tasks[(i, len(machines[i])-1)].EndExpr()
                          for i in all_jobs])
    objective_monitor = solver.Minimize(obj_var, 1)
    # Create search phases.
    sequence_phase = solver.Phase([all_sequences[i] for i in all_machines],
                                  solver.SEQUENCE_DEFAULT)
    vars_phase = solver.Phase([obj_var],
                              solver.CHOOSE_FIRST_UNBOUND,
                              solver.ASSIGN_MIN_VALUE)
    main_phase = solver.Compose([sequence_phase, vars_phase])
    # Create the solution collector.
    collector = solver.LastSolutionCollector()

    # Add the interesting variables to the SolutionCollector.
    collector.Add(all_sequences)
    collector.AddObjective(obj_var)

    for i in all_machines:
        sequence = all_sequences[i]
        sequence_count = sequence.Size()
        for j in range(0, sequence_count):
            t = sequence.Interval(j)
            collector.Add(t.StartExpr().Var())
            collector.Add(t.EndExpr().Var())
    # Solve the problem.
    disp_col_width = 10
    if solver.Solve(main_phase, [objective_monitor, collector]):
        # print("\nOptimal Schedule Length:", collector.ObjectiveValue(0), "\n")
        sol_line = ""
        sol_line_tasks = ""
        # print("Optimal Schedule", "\n")

        for i in all_machines:
            seq = all_sequences[i]
            sol_line += "Machine " + str(i) + ": "
            sol_line_tasks += "Machine " + str(i) + ": "
            sequence = collector.ForwardSequence(0, seq)
            seq_size = len(sequence)

            for j in range(0, seq_size):
                t = seq.Interval(sequence[j])
                # Add spaces to output to align columns.
                sol_line_tasks += t.Name() + " " * (disp_col_width - len(t.Name()))

            for j in range(0, seq_size):
                t = seq.Interval(sequence[j])
                sol_tmp = "[" + \
                    str(collector.Value(0, t.StartExpr().Var())) + ","
                sol_tmp += str(collector.Value(0, t.EndExpr().Var())) + "] "
                # Add spaces to output to align columns.
                sol_line += sol_tmp + " " * (disp_col_width - len(sol_tmp))

            sol_line += "\n"
            sol_line_tasks += "\n"

        # print(sol_line_tasks)
        # print("Time Intervals for Tasks\n")
        # print(sol_line)
        return machines, processing_times, sol_line_tasks, sol_line


def getdatas(machines, processing_times, sol_line_tasks, sol_line):
    numberofjob = len(machines)
    numberofmashine = len(machines[0])

    # get the sum processing time for every macshine
    sumtimeforeachmachine = np.zeros((numberofmashine))
    for i in range(numberofjob):
        for j in range(numberofmashine):
            for m in range(numberofmashine):

                if machines[i][j] == m:
                    sumtimeforeachmachine[m] += processing_times[i][j]

    sumtimeforeachjob = np.array(processing_times).sum(1)

    sumprosessingtime = [np.array(processing_times).sum().sum()]
    avetime_machine = sumtimeforeachmachine.sum()/numberofmashine
    avetime_job = sumtimeforeachjob.sum()/numberofjob

    datas = []
    machinereco = []
    id = 0
    for i in range(numberofjob):
        for j in range(numberofmashine):
            data1 = [
                id,                                    # id
                1/numberofjob,
                1/numberofmashine,
                j/numberofmashine,
                sumtimeforeachmachine[j]/avetime_machine,
                sumtimeforeachjob[i]/avetime_job,
                processing_times[i][j]/sumtimeforeachmachine[j],
                processing_times[i][j]/sumtimeforeachjob[i],
                machines[i][j]/numberofmashine,
            ]
            data2 = []
            for ii in range(numberofjob):
                for jj in range(numberofmashine):
                    data2.append(
                        processing_times[ii][jj] / sumprosessingtime[0])

            data3 = []
            for ii in range(numberofjob):
                for jj in range(numberofmashine):
                    data3.append(machines[ii][jj] / numberofmashine)
            a = sol_line_tasks.find(str(i)+'_'+str(j))
            b = a % (11 + numberofjob * 10 + 1)
            # b = a % 82
            c = b - 11
            data4 = [c // 10]+[machines[i][j]]
            machinereco.append(machines[i][j])
            # print(data4)

            data = data1+data2+data3+data4
            datas.append(data)
            id += 1
    return datas, machinereco


def gettraindata(name, m, n, time_low, time_high, numofloop):
    # machines is the workpiece processing sequence
        # line i is job i
        # row j is mashine j
        # means that job i's processing sequence
    # processing_time is the processing time of job i processing j
    fzzl = open('./data/log_' + name + '.txt', 'a')
    datatosave = []
    machinecdsave = []
    a = list(range(time_low, time_high))

    processing_times = []
    for k in range(n):
        processing_times.append(random.sample(a, m))

    pssave = np.array(processing_times)
    np.savetxt('./data/pssave_' + name + '.csv',
               pssave, fmt='%d', delimiter=',')

    for i in range(numofloop):
        print(i, file=fzzl)
        a = list(range(m))
        machines = []
        for k in range(n):
            machines.append(random.sample(a, m))

        machines, processing_times, sol_line_tasks, sol_line = main(
            machines, processing_times)
        print(sol_line_tasks, file=fzzl)
        print(sol_line, file=fzzl)
        datas, machinesco = getdatas(machines, processing_times, sol_line_tasks, sol_line)
        datatosave = datatosave + datas
        machinecdsave = machinecdsave + machinesco
        if i % 1 == 0:
            print(i)

    datanp = np.array(datatosave)
    machinecdsave = np.array(machinecdsave)
    np.savetxt('./data/featureandlable_' + name +
               '.csv', datanp, fmt='%.3f', delimiter=',')
    np.savetxt('./data/machineco_' + name +
               '.csv', machinecdsave, fmt='%.3f', delimiter=',')
               
    out_log = './data/log_' + name + '.txt'
    out_pssave = './data/pssave_' + name + '.csv'
    out_featureandlable = './data/featureandlable_' + name + '.txt'
    # datatosave = []
    return out_log, out_pssave, out_featureandlable


if __name__ == '__main__':
    m = 8
    n = 8
    time_low = 6
    time_high = 30
    numofloop = 1000

    gettraindata('traindata_'+'m='+str(m)+'_n='+str(n) +
                 '_timelow='+str(time_low)+'_timehight='+str(time_high)
                 + '_numofloop='+str(numofloop), m=m, n=n, time_low=time_low,
                 time_high=time_high, numofloop=numofloop)
