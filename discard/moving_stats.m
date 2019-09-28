clear; close all
beta = 10e-4;

%% generate y
y = randi(1023, 1000, 5);
y_weired =65535 * ones(1, 5); % extreme large target, occurs per 1k samples

y_in_all = [y; y_weired];
y_in_all = reshape(y_in_all, 5005, 1);
y_in_all = y_in_all(1:end-1);

% initial with the first element
mu = y_in_all(1);
nu = y_in_all(1)^2;

%% moving statics
mu_ = [mu];
nu_ = [nu];
normalized_ = [];
for i = 2 : length(y_in_all)
    mu = (1-beta) * mu + beta * y_in_all(i);
    nu = (1-beta) * nu + beta * y_in_all(i)^2;
    sigma = sqrt(nu - mu^2);
    normalized_y = ( y_in_all(i) - mu ) / sigma;
    mu_ = [mu_; mu]; nu_ = [nu_; nu]; 
    normalized_ = [normalized_; normalized_y];
end

%% plot
figure
subplot(311)
plot(mu_); xlabel("samples"); ylabel('mu'); xlim([0, 5000])
subplot(312)
plot(nu_); xlabel("samples"); ylabel('nu'); xlim([0, 5000])
subplot(313)
plot(normalized_); xlabel("samples"); ylabel('Normalized y'); xlim([0, 5000])
