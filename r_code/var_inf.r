tbl <- read.csv('ratings.csv');
y <- tbl$avg_rating;
permutation = sample.int(length(y));
train_prop = 0.8;
y_train = y[permutation[1:as.integer(length(y)*train_prop)]];
y_test = y[permutation[(as.integer(length(y)*train_prop)+1):length(y)]];
y = y_train;

nu0 = 1;
sigmasq0 = 1;
k0 = 0.1;
mu01 = 2.5;
mu02 = 2.5;
num_components = 2;
ber_p1 <- array(data=0.5, dim=length(y));
ber_p2 <- array(data=1-ber_p1[1],dim=length(y));
sig1alpha <- (length(y)*ber_p1[1]+nu0)/2;
sig1beta <- 0.5*nu0*sigmasq0
sig2alpha <- (length(y)*ber_p2[1]+nu0)/2;
sig2beta <- 0.5*nu0*sigmasq0;
lambda1 <- 5;
tau1 <- sigmasq0/k0;
lambda2 <- 1;
tau2 <- sigmasq0/k0;
