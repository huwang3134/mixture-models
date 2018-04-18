computeI <- function(mu, sigmasq, alpha, y) {
    component_scores <- matrix(nrow=length(y), ncol=length(mu));
    for (j in 1:length(mu)) {
        component_scores[1:length(y),j] <- alpha[j]*dnorm(y, mean=mu[j], sd=sqrt(sigmasq[j]));
    }
    sum_components <- rowSums(component_scores);
    component_scores <- component_scores/sum_components;
    # print('component scores');
    # print(component_scores);
    I1 <- rbinom(length(y), 1, component_scores[1:length(y),1]);
    I = matrix(nrow=length(y),ncol=2);
    I[1:length(y),1] = I1;
    I[1:length(y),2] = 1-I1;
    return(I);
};

chain_length <- 200201;
num_components = 2;
mu <- matrix(nrow=chain_length, ncol=num_components);
mu[1,1:2] <- c(5, 1);
sigmasq <- matrix(nrow=chain_length, ncol=num_components);
sigmasq[1,1:2] <- c(1, 1);
alpha <- matrix(nrow=chain_length, ncol=num_components);
alpha[1,1:2] <- c(0.5, 0.5);
i <- 2;
tbl <- read.csv('ratings.csv', header=TRUE);
y <- tbl$avg_rating;
permutation = sample.int(length(y));
train_prop = 0.8;
y_train = y[permutation[1:as.integer(length(y)*train_prop)]];
y_test = y[permutation[as.integer(length(y)*train_prop)+1:length(y)]];
y = y_train;

nu0 = c(1, 1);
sigmasq0 = c(1, 1);
k0 = c(0.1, 0.1);
mu0 = c(2.5, 2.5);
while (i <= chain_length) {
    print(i);
    I <- computeI(mu[i-1,1:2], sigmasq[i-1,1:2], alpha[i-1,1:2], y);
    y_lens = array(dim=2);
    for (j in 1:2) {
        mu_stacked <- matrix(data=mu[i-1,j],nrow=length(y),ncol=1,byrow=TRUE);
        s_squared <- nu0[j]*sigmasq0[j]+I[1:length(y),j]%*%((mu_stacked-y)^2);
        # print('I');
        # print(sum(I[1:length(y),j]));
        # print('s_2');
        # print(s_squared);
        num_j <- sum(I[1:length(y),j]);
        sigmasq_j <- 1.0/rgamma(n=1, shape=0.5*(num_j+nu0[j]), rate=0.5*s_squared);
        # print('sigmasq');
        # print(sigmasq_j);
        mean_mu_j <- (k0[j]*mu0[j]+sum(I[1:length(y),j]%*%y))/(k0[j]+num_j);
        stdev_mu_j = sqrt(sigmasq_j/(k0[j]+num_j));
        mu_j = rnorm(n=1, mean=mean_mu_j, sd=stdev_mu_j);
        # print('mu');
        # print(mu_j);
        y_lens[j] = num_j;
        mu[i,j] = mu_j;
        sigmasq[i,j] = sigmasq_j;
    }
    alpha[i,1] = rbeta(n=1,shape1=y_lens[1]+1,shape2=y_lens[2]+1);
    alpha[i,2] = 1-alpha[i,1];
    # print('alpha');
    # print(alpha[i,1]);
    i = i+1;
}

plot(1:chain_length,mu[1:chain_length,1]);
dev.new();
plot(1:chain_length,mu[1:chain_length,2]);
dev.new();
acf(mu[1:chain_length,1], lag.max=10000);
dev.new();
acf(alpha[1:chain_length,1], lag.max=10000);

burn_in = 201;
thin_interval = 2000;
sample_size = (chain_length-burn_in)/thin_interval;
y_pred = array(dim=sample_size);
num_bins = sample_size/2+1;
bins = (0:(num_bins-1))*(5/(num_bins-1));
bins = c(0, bins);
pred_bins = array(data=0, dim=length(bins));
for (i in 1:sample_size) {
    mu_draw = mu[burn_in+(i-1)*thin_interval,1:num_components];
    sigmasq_draw = sigmasq[burn_in+(i-1)*thin_interval,1:num_components];
    alpha_draw = alpha[burn_in+(i-1)*thin_interval,1:num_components];
    y = -1;
    if (runif(n=1) < alpha_draw[1]) {
        y rnorm(n=1, mean=mu_draw[1], sd=sqrt(sigmasq_draw[1]));
    }
    else {
        y = rnorm(n=1, mean=mu_draw[2], sd=sqrt(sigmasq_draw[2]));
    }
    y_pred[i] = y;
    if (y < 0) {
        pred_bins[1] = pred_bins[1]+1;
    }
    else {
        index = which.min(max(0, y-bins[2:length(bins)]));
        pred_bins[index+1] = pred_bins[index+1]+1;
    }
}
test_bins = array(data=0, dim=length(bins));
for (i in 1:length(y_test)) {
    if (y_test[i] < 0) {
        pred_bins[1] = pred_bins[1]+1;
    }
    else {
        index = which.min(max(0, y_test[i]-bins[2:length(bins)]));
        test_bins[index+1] = test_bins[index+1]+1;
    }
}
test_bins = test_bins[pred_bins > 0];
pred_bins = pred_bins[pred_bins > 0];
print(sum(((pred_bins-test_bins)^2)/pred_bins));
