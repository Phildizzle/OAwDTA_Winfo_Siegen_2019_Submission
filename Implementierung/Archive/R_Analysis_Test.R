# To be specified by the user:
# Inputfile, sim, x_figure, x_sc, Plot names

#clear current environment variables
rm(list=ls())

#read data
myData = read.csv(file.choose(), sep=";", header = FALSE)
str(myData)

myNames = c("Customers", "Resources", "Max_Tickets", "Capacity", "Epsilon", "Simulations", "Permutations", "OPT", "One", "Dyn", "Greedy", "Interval", "One_Relaxed", "Dyn_Relaxed", "WTP", "Amazon", "Simulation_Time", "Permutation_Time", "Input_Time", "OPT_f_Time", "OPT_i_Time", "Permute_Time", "One_Time", "Dyn_Time", "Greedy_Time", "Interval_Time", "One_Relaxed_Time", "Dyn_Relaxed_Time", "WTP_Time", "Amazon_Time")
names(myData) <- myNames
attach(myData)

#calculate averages
One_Avg = vector()
Dyn_Avg = vector()
Greedy_Avg = vector()
Interval_Avg = vector()
One_Relaxed_Avg = vector()
Dyn_Relaxed_Avg = vector()
WTP_Avg = vector()
Amazon_Avg = vector()

Sim_Time_Avg = vector()
Perm_Time_Avg = vector()
Input_Time_Avg = vector()
OPT_f_Time_Avg = vector()
OPT_i_Time_Avg = vector()
Permute_Time_Avg = vector()
One_Time_Avg = vector()
Dyn_Time_Avg = vector()
Greedy_Time_Avg = vector()
Interval_Time_Avg = vector()
One_Relaxed_Time_Avg = vector()
Dyn_Relaxed_Time_Avg = vector()
WTP_Time_Avg = vector()
Amazon_Time_Avg = vector()

Cust = vector()
Res = vector()
Max_Tick = vector()
Cap = vector()
Eps = vector()
Perm = vector()

sim = 10 #Number of simulations for taking average: Change here if necessary
for (i in 1:(length(One)/sim)) {
  One_Avg[i] <- mean(One[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Dyn_Avg[i] <- mean(Dyn[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Greedy_Avg[i] <- mean(Greedy[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Interval_Avg[i] <- mean(Interval[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  One_Relaxed_Avg[i] <- mean(One_Relaxed[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Dyn_Relaxed_Avg[i] <- mean(Dyn_Relaxed[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  WTP_Avg[i] <- mean(WTP[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Amazon_Avg[i] <- mean(Amazon[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  
  Sim_Time_Avg[i] <- mean(Simulation_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Perm_Time_Avg[i] <- mean(Permutation_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Input_Time_Avg[i] <- mean(Input_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  OPT_f_Time_Avg[i] <- mean(OPT_f_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  OPT_i_Time_Avg[i] <- mean(OPT_i_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Permute_Time_Avg[i] <- mean(Permute_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  One_Time_Avg[i] <- mean(One_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Dyn_Time_Avg[i] <- mean(Dyn_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Greedy_Time_Avg[i] <- mean(Greedy_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Interval_Time_Avg[i] <- mean(Interval_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  One_Relaxed_Time_Avg[i] <- mean(One_Relaxed_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Dyn_Relaxed_Time_Avg[i] <- mean(Dyn_Relaxed_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  WTP_Time_Avg[i] <- mean(WTP_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  Amazon_Time_Avg[i] <- mean(Amazon_Time[(sim*(i-1)+1):(sim*i)]/OPT[(sim*(i-1)+1):(sim*i)])
  
  #Potential x-figures
  Cust[i] <- mean(Customers[(sim*(i-1)+1):(sim*i)])
  Res[i] <- mean(Resources[(sim*(i-1)+1):(sim*i)])
  Max_Tick[i] <- mean(Max_Tickets[(sim*(i-1)+1):(sim*i)])
  Cap[i] <- mean(Capacity[(sim*(i-1)+1):(sim*i)])
  Eps[i] <- mean(Epsilon[(sim*(i-1)+1):(sim*i)])
  Perm[i] <- mean(Permutations[(sim*(i-1)+1):(sim*i)])
}

x_figure = NULL #Change here


png( filename = "NULL_Performance.png" #Change name of illustration here
     , units = "px"
     , height = 1600
     , width = 1600
     , res = 300
)
#Performance plot
x_sc = c(NULL,NULL) #Change here
plot(x_figure, Dyn_Avg, xlim = x_sc, ylim = c(0,1), col = "darkgreen", type = "o", lwd=2, xlab = "x-Figure", ylab = "Average Performance in % of OPT")
par(new = TRUE)
plot(x_figure, One_Avg, xlim = x_sc, ylim = c(0,1), col = "blue", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Greedy_Avg, xlim = x_sc, ylim = c(0,1), col = "red", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Interval_Avg, xlim = x_sc, ylim = c(0,1), col = "darkorange", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, One_Relaxed_Avg, xlim = x_sc, ylim = c(0,1), col = "mediumpurple", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Dyn_Relaxed_Avg, xlim = x_sc, ylim = c(0,1), col = "gold", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, WTP_Avg, xlim = x_sc, ylim = c(0,1), col = "hotpink", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Amazon_Avg, xlim = x_sc, ylim = c(0,1), col = "saddlebrown", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(xpd=TRUE)
legend(0, 0.5, legend=c("Dynamic", "One-Time", "Greedy", "Interval", "One-Time Relaxed", "Dynamic Relaxed", "WTP Learner", "Amazon Learner"),
       col=c("darkgreen", "blue", "red", "darkorange", "mediumpurple", "gold", "hotpink", "saddlebrown"), lty=1, cex=0.8)
par(xpd=FALSE)
dev.off()


#Runtime plot
png( filename = "NULL_Max_Runtime.png" #Change name of illustration here
     , units = "px"
     , height = 1600
     , width = 1600
     , res = 300
)
lb = 0.8*min(OPT_f_Time_Avg, OPT_i_Time_Avg, Dyn_Time_Avg, One_Time_Avg, Greedy_Time_Avg, Interval_Time_Avg, One_Relaxed_Time_Avg, Dyn_Relaxed_Time_Avg, WTP_Time_Avg, Amazon_Time_Avg)
ub = 1.05*max(OPT_f_Time_Avg, OPT_i_Time_Avg, Dyn_Time_Avg, One_Time_Avg, Greedy_Time_Avg, Interval_Time_Avg, One_Relaxed_Time_Avg, Dyn_Relaxed_Time_Avg, WTP_Time_Avg, Amazon_Time_Avg)
x_sc = c(NULL,NULL) #Change here
plot(x_figure, OPT_f_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "gray35", type = "o", lwd=2, xlab = "x-Figure", ylab = "Average Runtime")
par(new = TRUE)
plot(x_figure, OPT_i_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "darkmagenta", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Dyn_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "darkgreen", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, One_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "blue", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Greedy_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "red", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Interval_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "darkorange", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, One_Relaxed_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "mediumpurple", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Dyn_Relaxed_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "gold", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, WTP_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "hotpink", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Amazon_Time_Avg, xlim = x_sc, ylim = c(lb, ub), col = "saddlebrown", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(xpd=TRUE)
legend(3, (lb+(ub-lb)*0.9), legend=c("OPT Fractional", "OPT Integral", "Dynamic", "One-Time", "Greedy", "Interval", "One-Time Relaxed", "Dynamic Relaxed", "WTP Learner", "Amazon Learner"),
       col=c("gray35", "darkmagenta", "darkgreen", "blue", "red", "darkorange", "mediumpurple", "gold", "hotpink", "saddlebrown"), lty=1, cex=0.8, bg="transparent")
par(xpd=FALSE)
dev.off()

#Simulation_Time plot
plot(x_figure, Sim_Time_Avg, col = "navy", type = "o", lwd=2, xlab = "x-Figure", ylab = "Average Simulation Time")

#Permutation_Time, Input_Time, Permute_Time plot
lb = 0.05*min(Perm_Time_Avg, Input_Time_Avg, Permute_Time_Avg)
ub = 1.05*max(Perm_Time_Avg, Input_Time_Avg, Permute_Time_Avg)
plot(x_figure, Perm_Time_Avg, ylim = c(lb, ub), col = "navy", type = "o", lwd=2, xlab = "x-Figure", ylab = "Average Runtime")
par(new = TRUE)
plot(x_figure, Input_Time_Avg, ylim = c(lb, ub), col = "firebrick1", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(new = TRUE)
plot(x_figure, Permute_Time_Avg, ylim = c(lb, ub), col = "limegreen", type = "o", lwd=2, axes = FALSE, xlab = "", ylab = "")
par(xpd=TRUE)
legend(-0.08, (lb+(ub-lb)*(-0.1)), legend=c("Average Permutation Runtime", "Average Input Simulation Runtime", "Average Input Permutation Runtime"),
       col=c("navy", "firebrick1", "limegreen"), lty=1, cex=0.8)
par(xpd=FALSE)
