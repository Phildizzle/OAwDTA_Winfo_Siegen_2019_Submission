%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Agrawal et al. Implementation
%Version 1.0 
%July 18, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To Do: Implementierung des "Omega", Über alle Permutationen laufen lassen,
%Prototyp mit eigener Eingabe eines jeden Kunden, Check wieso dynamic <
%one-time


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization
clear all; close all; clc;
disp('Willkommen zur Simulation des Ticket Sales Case');
disp('Bitte initialisieren Sie im Folgenden die Probleminstanz');
n = input('Total number of customers n: ');
m = input('Total number of resources m: ');
for i = 1:m
    str1 = 'Total capacity of resource %d: ';
    b(i) = input(sprintf(str1, i));
end
eps = input('Fraction epsilon: ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulate incoming customers
for j = 1:n
   p(j) = poissrnd(60); %Objective function coefficient
   for i = 1:m
      a(i,j) = poissrnd(0.75)+1; %Capacity consumption
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Choose random permutation
perm = randperm(n);
for j = 1:n
    p1(j) = p(perm(j));
    for i = 1:m
       a1(i,j) = a(i,perm(j));
    end
end
p = p1;
a = a1;
%Later: Retrieve all permutations of p via perms(p)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate ex-post optimum
lob = zeros(n,1);
upb = zeros(n,1)+1;
intcon = 1:n;
[x,obj] = linprog(-p,a,b,[],[],lob,upb); %Use -p since linprog solves a minimization problem
[x_bin,obj_bin] = intlinprog(-p,intcon,a,b,[],[],lob,upb); %Use -p since linprog solves a minimization problem
obj_ex = -obj; %Store ex-post fractional optimum
obj_bin_ex = -obj_bin; %Store ex-post integral optimum
disp(sprintf('Ex-post fractional optimum: %0.5f',obj_ex));
input(sprintf('Ex-post integral optimum: %0.5f',obj_bin_ex));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%One-time Learning Algorithm
rev_one = 0; %Total revenue
b_con = b;

if min(b_con) >= (6*m*log(n/eps))/eps^3
    input('Right-hand-side condition for one-time learning: TRUE');
else
    input('Right-hand-side condition for one-time learning: FALSE');
end

%Step (i)
s = ceil(n*eps);
for j = 1:s
   x(j) = 0; 
end
lob = zeros(s,1);
upb = zeros(s,1)+1;
[x_cal,obj,exitflag,output,lambda] = linprog(-p(1:s),a(:,1:s),b*(1-eps)*s/n,[],[],lob,upb); %Use -p since linprog solves a minimization problem
%Note: no shadow prices / lambda can be obtained for intlinprog
sp_one = lambda.ineqlin %Store shadow prices

for j = (s+1):n
    
    %Equation 12
    if p(j) <= sp_one'*a(:,j)
        x_hat = 0;
    else x_hat = 1;
    end
    
    %Step (ii)
    help1 = a(:,1:j-1)*x(1:j-1); %Resource consumption by previous customers
    help2 = 1; %Indicator
    for i = 1:m
        if a(i,j)*x_hat > b_con(i)-help1(i)
           help2 = 0; 
        end
    end
    if help2 == 1
       x(j) = x_hat;
       rev_one = rev_one + p(j)*x(j);
       for i = 1:m
          b(i) = b(i) - a(i,j)*x(j);
       end
    else x(j) = 0;
    end
end

obj_one = rev_one;
disp(sprintf('Ex-post fractional optimum: %0.5f',obj_ex));
disp(sprintf('Ex-post integral optimum: %0.5f',obj_bin_ex));
input(sprintf('One-time learning result: %0.5f',obj_one));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dynamic Pricing Algorithm
rev_dyn = 0; %Total revenue
b = b_con; %Reset capacities
l = 0;

if min(b_con) >= (10*m*log(n/eps))/eps^2
    input('Right-hand-side condition for dynamic learning: TRUE');
else
    input('Right-hand-side condition for dynamic learning: FALSE');
end

%Step (i)
t0 = ceil(n*eps);
for t = 1:t0
   x(t) = 0; 
end

%Step (ii)
for t = (t0+1):n
    l_old = l;
    r = 0;
    while ceil(n*eps*2^r) < t
        if ceil(n*eps*2^(r+1)) >= t
           break; 
        end
        r = r+1;
    end
    l = ceil(n*eps*2^r);
    if l ~= l_old && l ~= 0
        lob = zeros(l,1);
        upb = zeros(l,1)+1;
        [x_cal,obj,exitflag,output,lambda] = linprog(-p(1:l),a(:,1:l),b*(1-eps*((n/l)^0.5))*l/n,[],[],lob,upb); %Use -p since linprog solves a minimization problem
    end
    sp_dyn = lambda.ineqlin; %Store shadow prices
    
    %(a)
    if p(t) <= sp_dyn'*a(:,t)
        x_hat = 0;
    else x_hat = 1;
    end
    
    %(b)
    help1 = a(:,1:t-1)*x(1:t-1); %Resource consumption by previous customers
    help2 = 1; %Indicator
    for i = 1:m
        if a(i,t)*x_hat > b_con(i)-help1(i)
           help2 = 0; 
        end
    end
    if help2 == 1
       x(t) = x_hat;
       rev_dyn = rev_dyn + p(t)*x(t);
       for i = 1:m
         b(i) = b(i) - a(i,t)*x(t);
       end
    else x(t) = 0;
    end
    
end

obj_dyn = rev_dyn;
disp(sprintf('Ex-post fractional optimum: %0.5f',obj_ex));
disp(sprintf('Ex-post integral optimum: %0.5f',obj_bin_ex));
disp(sprintf('One-time learning result: %0.5f',obj_one));
input(sprintf('Dynamic learning result: %0.5f',obj_dyn));

disp('Done!');