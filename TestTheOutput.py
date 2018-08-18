import trainnetwork_ann
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def makespan(T, Xstr, plot_if):
    # this function use the scheduling result to calculate the loss of the schedule,
    # and draw the scheduled Gantt map
    # 'T' is the time consumpution of current problem
    # 'Xstr' is a list of scheduling result 
    # plot_if is a flag show wheather need to draw a figure 
    # Xstr = [工件编号,加工机器编号,工序编号,开始时间,结束时间]
    # Xstr = [number of jobs, number of machine, number of process, start time and end time]
    
    # init he makespan with 0, and some other paramater  
    makespan = 0
    number_of_machine, number_of_job = T.shape


    N = len(Xstr)
    machine_busy = np.zeros((number_of_machine), dtype= int)
    job_busy = np.zeros((number_of_job), dtype=int)

    Time_start = np.zeros((number_of_machine, number_of_job), dtype=int)
    Time_end = np.zeros((number_of_machine, number_of_job), dtype=int)

    machine_next_end_time = np.zeros((number_of_machine), dtype=int)
    job_next_end_time = np.zeros((number_of_job), dtype=int)

    job_count_in_machine = np.zeros((number_of_machine), dtype=int)
    NO = np.zeros((number_of_machine, number_of_job), dtype=int)
    proc = np.zeros((number_of_machine, number_of_job), dtype=int)
    for i in range(N):
        # % current_job = Xstr(i)
        # % 提取当前的处理的工作指向

        current_job = Xstr[i][0]
        # % 当前的工件
        current_machine = Xstr[i][1]
        # % 当前的机器
        current_proccess = Xstr[i][2]

        current_job_count_in_machine = job_count_in_machine[
            current_machine]
        # % 提取当前的机器的工作的次数。

        Time_start[current_machine, int(current_job_count_in_machine)] = max(
            machine_next_end_time[current_machine], job_next_end_time[current_job])
        # % 提取开始时间
        Time_end[current_machine, current_job_count_in_machine] = Time_start[current_machine,
                                                                             current_job_count_in_machine] + Xstr[i][4]-Xstr[i][3]
        # % 计算结束时间

        machine_next_end_time[current_machine] = Time_end[current_machine,
                                                          current_job_count_in_machine]
        # % 赋值新的结束时间
        job_next_end_time[current_job] = Time_end[current_machine,
                                                  current_job_count_in_machine]
        job_count_in_machine[current_machine] = job_count_in_machine[current_machine] + 1
        # % Time_start[current_mashine, current_job_count_in_mashine] =
        NO[current_machine,  current_job_count_in_machine] = current_job
        proc[current_machine,  current_job_count_in_machine] = current_proccess

    Y1p = Time_start  # % 开始点的集合
    Y2p = Time_end  # % 结束点的集合
    Y3p = NO  # % 工件编号的集合

    Fit = np.max(Y2p)
    if plot_if == 1:
        zzl = plt.figure(figsize=(12,4))
        for i in range(number_of_machine):
            # number_of_mashine:
            for j in range(number_of_job):
                # number_of_job:

                # % 数据读写
                mPoint1 = Y1p[i, j]  # % 读取开始的点
                mPoint2 = Y2p[i, j]  # % 读取结束的点
                mText = number_of_machine-i  # % 读取机器编号
                PlotRec(mPoint1, mPoint2, mText)  # % 画图函数，（开始点，结束点，高度）
                Word = str(Y3p[i, j]+1)+'_'+str(proc[i, j])  # % 读取工件编号
                # hold on

                # % 填充
                x1 = mPoint1
                y1 = mText-1
                x2 = mPoint2
                y2 = mText-1
                x3 = mPoint2
                y3 = mText
                x4 = mPoint1
                y4 = mText
                colorbox = ['yellow', 'whitesmoke', 'lightyellow',
                            'khaki', 'silver', 'pink', 'lightgreen', 'orange']

                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4],
                         [1, 0.5, 1], color=colorbox[Y3p[i, j]])

                # % 书写文字
                plt.text(0.5*mPoint1+0.5*mPoint2-3, mText-0.5, Word)

    return Fit, Y1p, Y2p, Y3p


def PlotRec(mPoint1, mPoint2, mText):

    vPoint = np.zeros((4, 2))
    vPoint[0, :] = [mPoint1, mText-1]
    vPoint[1, :] = [mPoint2, mText-1]
    vPoint[2, :] = [mPoint1, mText]
    vPoint[3, :] = [mPoint2, mText]
    plt.plot([vPoint[0, 0], vPoint[1, 0]], [vPoint[0, 1], vPoint[1, 1]], 'k')
    # hold on
    plt.plot([vPoint[0, 0], vPoint[2, 0]], [vPoint[0, 1], vPoint[2, 1]], 'k')
    plt.plot([vPoint[1, 0], vPoint[3, 0]], [vPoint[1, 1], vPoint[3, 1]], 'k')
    plt.plot([vPoint[2, 0], vPoint[3, 0]], [vPoint[2, 1], vPoint[3, 1]], 'k')


def FindMost(mat, m):
    # print(mat)
    mat = np.array(mat)
    mc = mat[:, :m]
    max1 = np.max(mc)
    hang, lei = np.where(mc == max1)
    hang = hang[0]
    lei = lei[0]
    index = int(mat[hang, -1])
    return int(hang), int(lei), index


def Record(hang, lei, index, macshine, out):

    out[macshine, lei] = index
    return out


def Eraser(mat, hang, lei, m):
    mat = np.array(mat)
    for i in range(m):
        mat[i, lei] = -1
    for i in range(m):
        mat[hang, i] = -1
    return mat


def SortMachine(macshine, m0, out, m):
    m_current = m0.copy()
    for i in range(m):
        hang, lei, index = FindMost(m_current, m)
        out = Record(hang, lei, index, macshine, out)
        m_current = Eraser(m_current, hang, lei, m)
    # print(out)
    return out


def LineUpTheSolution(model_output, path_of_machine, m, n, T, plot_if):
    # This function lineup the solution of one problem,
    # 'model_output' is the output of current model
    # 'path_of_machine' is the path of the machine information which created by another
    # nn model.
    # 'T' is the time constraint current problem
    # 'm' and 'n' is the number of machine and the jobs.

    # load the machine information with 'path of machine'
    Machinearrangement = np.loadtxt(path_of_machine, delimiter=',')

    # init the matrix M,  M is used to combine the workpieces from the same machine
    M = []
    for i in range(m):
        M.append([])
    for i in range(m*n):
        M[int(Machinearrangement[i])].append(list(model_output[i])+[i])

    # init the lined solution by m,and n
    solution_lined = np.zeros((m, n), dtype=int)-1
    for i in range(m): # line up the solution with sortmacshine method, one by one machine 
        solution_lined = SortMachine(i, M[i], solution_lined, m)

    # calculate the time consumption and the 
    Tc = np.zeros((m, n))-1
    Xtr = []
    for i in range(m):
        for j in range(n):
            num_of_job = solution_lined[j, i] // m
            num_of_process = 0
            num_of_time_s = 0
            num_of_time_e = T[solution_lined[j, i] // m][solution_lined[j, i] % m]
            num_of_machine = j
            Xtr.append([num_of_job, num_of_machine,
                        num_of_process, num_of_time_s, num_of_time_e])

    # calculate the makespan and draw the picture
    makespan(Tc, Xtr, plot_if)


def lineuptheoptimal(optimallogfile, m, n, plot_if):
    # this function draw the picture of optimal solution
    # optimallogfile is the optimal output
    # 'm' and 'n' is the number of machine and the jobs

    # calculate the time consumption and the 
    Xtr = []
    Tc = np.zeros((m, n))-1

    for i in range(m):
        Maclin = optimallogfile[1 + i]
        Timlin = optimallogfile[2 + m + i]
        item_in_mac = Maclin.split('Job')
        item_in_tim = Timlin.split('[')

        for j in range(n):

            current_jp = item_in_mac[1+j].split('_')
            num_of_job = int(current_jp[1])
            num_of_process = int(current_jp[2])
            current_ti = item_in_tim[1+j].split(',')
            num_of_time_s = int(current_ti[0])
            num_of_time_e = int(current_ti[1].split(']')[0])
            num_of_machine = i

            Xtr.append([num_of_job, num_of_machine,
                        num_of_process, num_of_time_s, num_of_time_e])

    for i in range(m*n):
        for j in range(m*n):
            if Xtr[i][3] < Xtr[j][3]:
                Xtr[i], Xtr[j] = Xtr[j], Xtr[i]

    # calculate the makespan and draw the picture
    # print(Xtr)
    makespan(Tc, Xtr, plot_if)


def TestOneSchedule(test_feature, path_of_model, path_of_machine,
                    T, m, n, optimallogfile):
    # This function test the schedule ability with one example, and draw the picture
    # 'test_feature' is the input feature, this function put it into the trained model
    # and test the output of the model.
    # 'test_label' is the optimal label, used to Compared with the output model.
    # 'path_of_model' is the path of the model to be test.
    # 'path_of_machine' is the path of the machine information which created by another
    # nn model.
    # 'T' is the time constraint current problem
    # 'm' and 'n' is the number of machine and the jobs
    # optimallogfile is the optimal output

    # load the model
    model = load_model(path_of_model)
    # calculate the output of the model
    model_output = model.predict(test_feature)
    # test and show the output of the model
    LineUpTheSolution(model_output, path_of_machine, m, n, T, 1)
    # show the optimal solution
    lineuptheoptimal(optimallogfile, m, n, 1)

    plt.show()

def main():
    m = 8
    featuredata = 'featureandlable_traindata_m=8_n=8_timelow=6_' + \
        'timehight=30_numofloop=1000.csv'
    train_feature, train_label, test_feature, test_label, \
        inputnum, nb_classes = trainnetwork_ann.importdata(featuredata)
    modelname = './model/ann_schedual_2018_06_29::19_20_11' + \
        'ann_layer15_featureandlable_traindata_m=8_n=8_' + \
        'timelow=6_timehight=30_numofloop=1000.csv.h5'
    path_of_machine = './data/machineco_traindata_m=8_n=8_' + \
        'timelow=6_timehight=30_numofloop=1000.csv'
    T = np.loadtxt('./data/pssave_traindata_m=8_n=8_' +
                   'timelow=6_timehight=30_numofloop=1000.csv', delimiter=',')
    logdata = []
    flogdata = open('./data/log_traindata_m=8_n=8_timelow' +
                    '=6_timehight=30_numofloop=1000.txt', 'r')

    # load the logfile of current problem, it restore the best solution
    for i in range(601*(3+2*m)):
        # print(i)
        if i >= 600*(3+2*m):
            logdata.append(flogdata.readline())
        else:
            flogdata.readline()

    print(logdata)

    TestOneSchedule(test_feature, modelname,
                    path_of_machine, T, 8, 8, logdata)


if __name__ == '__main__':
    main()
    