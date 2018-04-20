tbl <- read.csv('train_ratings.csv');
tbl2 <- read.csv('test_ratings.csv');
y_train <- tbl$avg_rating;
y_test <- tbl2$avg_rating;

sample_size = 100;
y_pred = array(dim=sample_size)
mu_mean = mean(y_train);
n = length(y_train);
s_squared = sd(y_train)**2;
num_bins = 6;
bins = (0:(num_bins-1))*(5/(num_bins-1));
bins = c(0, bins);
pred_bins = array(data=0, dim=length(bins));
for (i in 1:sample_size) {
    sigmasq_draw = 1/rgamma(n=1, shape=0.5*(n-1), rate=(n-1)*s_squared/2);
    mu_draw = rnorm(n=1, mean=mu_mean, sd=sqrt(sigmasq_draw/n));
    y = rnorm(n=1, mean=mu_draw, sd=sqrt(sigmasq_draw));
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
print(bins);
dev.new();
hist(y_pred, plot=TRUE, main='Histogram of Normal Values');
dev.new();
hist(y_test, plot=TRUE, main='Histogram of Hold-out Data');
