tbl <- read.csv('train_ratings.csv');
y_train <- tbl$avg_rating;
tbl2 <- read.csv('test_ratings.csv');
y_test <- tbl2$avg_rating;
y = y_train;

nu0 = 1;
sigmasq0 = 1;
k0 = 0.1;
mu01 = 2.5;
mu02 = 2.5;
num_components = 2;
ber_p1 <- array(data=0.5, dim=length(y));
ber_p2 <- array(data=1-ber_p1[1],dim=length(y));
sig1alpha <- 1;
sig1beta <- 0.5*nu0*sigmasq0
sig2alpha <- 1;
sig2beta <- 0.5*nu0*sigmasq0;
lambda1 <- 2.6;
tau1 <- sigmasq0/k0;
lambda2 <- 2.4;
tau2 <- sigmasq0/k0;
alpha_alpha <- 1;
alpha_beta <- 1;

itermax = 5000;
lambda1_vals = array(dim=itermax);
lambda2_vals = array(dim=itermax);
tau1_vals = array(dim=itermax);
tau2_vals = array(dim=itermax);
sig1alpha_vals = array(dim=itermax);
sig1beta_vals = array(dim=itermax);
alpha_alpha_vals = array(dim=itermax);
alpha_beta_vals = array(dim=itermax);

iters = 1;
while (iters <= itermax) {
    neg_log_sigsq1 = digamma(sig1alpha)-log(sig1beta);
    neg_log_sigsq2 = digamma(sig2alpha)-log(sig2beta);
    reciprocal_sigsq1 = sig1alpha/sig1beta;
    reciprocal_sigsq2 = sig2alpha/sig2beta;
    mu1_sq = tau1+lambda1^2;
    mu2_sq = tau2+lambda2^2;
    log_alpha = digamma(alpha_alpha)-digamma(alpha_alpha+alpha_beta);
    log_1minus_alpha = digamma(alpha_beta)-digamma(alpha_alpha+alpha_beta);
    w = 0.5*neg_log_sigsq1-0.5*reciprocal_sigsq1*(y^2-2*lambda1*y+mu1_sq)-0.5*neg_log_sigsq2+0.5*reciprocal_sigsq2*(y^2-2*lambda2*y+mu2_sq)+log_alpha-log_1minus_alpha;
    ber_p1 = exp(w)/(1+exp(w));
    ber_p2 = 1-ber_p1;
    sig1alpha = 0.5*(sum(ber_p1)+nu0);
    sig1beta = 0.5*(nu0*sigmasq0+sum(ber_p1*(y^2-2*y*lambda1+mu1_sq))+k0*(mu01^2-2*mu01*lambda1+mu1_sq));
    sig2alpha = 0.5*(sum(ber_p2)+nu0);
    sig2beta = 0.5*(nu0*sigmasq0+sum(ber_p2*(y^2-2*y*lambda2+mu2_sq))+k0*(mu02^2-2*mu02*lambda2+mu2_sq));
    reciprocal_sigsq1 = sig1alpha/sig1beta;
    reciprocal_sigsq2 = sig2alpha/sig2beta;
    lambda1 = (sum(ber_p1*y)+k0*mu01)/(sum(ber_p1)+k0);
    lambda1_vals[iters] = lambda1;
    tau1 = 1/(reciprocal_sigsq1*(sum(ber_p1)+k0));
    lambda2 = (sum(ber_p2*y)+k0*mu02)/(sum(ber_p2)+k0);
    lambda2_vals[iters] = lambda2;
    tau2 = 1/(reciprocal_sigsq2*(sum(ber_p2)+k0));
    tau1_vals[iters] = tau1;
    tau2_vals[iters] = tau2;
    sig1alpha_vals[iters] = sig1alpha;
    sig1beta_vals[iters] = sig1beta;
    alpha_alpha = sum(ber_p1)+1;
    alpha_beta = sum(ber_p2)+1;
    alpha_alpha_vals[iters] = alpha_alpha;
    alpha_beta_vals[iters] = alpha_beta;
    iters = iters+1;
}

plot(1:itermax, lambda1_vals, main='lambda1');
dev.new();
plot(1:itermax, lambda2_vals, main='lambda2');
dev.new();
plot(1:itermax, tau1_vals, main='tau1');
dev.new();
plot(1:itermax, tau2_vals, main='tau2');
dev.new();
plot(1:itermax, sig1alpha_vals, main='sig1alpha');
dev.new();
plot(1:itermax, sig1beta_vals, main='sig1beta');
dev.new();
plot(1:itermax, alpha_alpha_vals, main='alpha_alpha');
dev.new();
plot(1:itermax, alpha_beta_vals, main='alpha_beta');
print('lambda1');
print(lambda1);
print('lambda2');
print(lambda2);
print('tau1');
print(tau1);
print('tau2');
print(tau2);
print('alpha_alpha');
print(alpha_alpha);
print('alpha_beta');
print(alpha_beta);

sample_size = 100;
y_pred = array(dim=sample_size);
num_bins = 6;
bins = (0:(num_bins-1))*(5/(num_bins-1));
bins = c(0, bins);
pred_bins = array(data=0, dim=length(bins));
for (i in 1:sample_size) {
    mu_draw = c(rnorm(mean=lambda1, sd=sqrt(tau1), n=1), rnorm(mean=lambda2, sd=sqrt(tau2), n=1));
    sigmasq_draw = c(1/rgamma(shape=sig1alpha, rate=sig1beta, n=1), 1/rgamma(shape=sig2alpha, rate=sig2beta, n=1));
    alpha_draw1 = rbeta(shape1=alpha_alpha, shape2=alpha_beta, n=1);
    alpha_draw = c(alpha_draw1, 1-alpha_draw1);
    y = -1;
    if (runif(n=1) < alpha_draw[1]) {
        y = rnorm(n=1, mean=mu_draw[1], sd=sqrt(sigmasq_draw[1]));
    }
    else {
        y = rnorm(n=1, mean=mu_draw[2], sd=sqrt(sigmasq_draw[2]));
    }
    y_pred[i] = y;
    if (y < 0) {
        pred_bins[1] = pred_bins[1]+1;
    }
    else {
        index = which.min(abs(y-bins[2:length(bins)]));
        if (bins[index+1] > y) {
            pred_bins[index] = pred_bins[index]+1;
        }
        else {
            pred_bins[index+1] = pred_bins[index+1]+1;
        }
    }
}
test_bins = array(data=0, dim=length(bins));
for (i in 1:length(y_test)) {
    if (y_test[i] < 0) {
        test_bins[1] = test_bins[1]+1;
    }
    else {
        index = which.min(abs(y_test[i]-bins[2:length(bins)]));
        if (bins[index+1] > y_test[i]) {
            test_bins[index] = test_bins[index]+1;
        }
        else {
            test_bins[index+1] = test_bins[index+1]+1;
        }
    }
}
print((pred_bins > 0));
test_bins = test_bins[pred_bins > 0];
pred_bins = pred_bins[pred_bins > 0];
pred_bins = pred_bins*length(y_test)/sample_size;
print(sum(((pred_bins-test_bins)^2)/pred_bins));
print(pred_bins);
print(test_bins);
print(bins);
dev.new();
hist(y_pred, plot=TRUE, main='Histogram of predicted values from Variational Inference');
dev.new();
hist(y_test, plot=TRUE, main='Histogram of holdout data');
