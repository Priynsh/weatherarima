using CSV
using DataFrames, Dates
using Plots
using StatsBase
using HypothesisTests
using LinearAlgebra
function exponential_smoothing(series::Vector{Float64}, alpha::Float64)
    smoothed_series = Float64[]
    push!(smoothed_series, series[1])

    for i in 2:length(series)
        smoothed_value = alpha * series[i] + (1 - alpha) * smoothed_series[end]
        push!(smoothed_series, smoothed_value)
    end

    return smoothed_series
end
function rolling_stats(series::Vector{Float64}, window::Int)
    rolling_mean = [mean(series[max(1, i-window+1):i]) for i in 1:length(series)]
    rolling_std = [std(series[max(1, i-window+1):i]) for i in 1:length(series)]
    return rolling_mean, rolling_std
end


function autocorrelation(series, k)
    n = length(series)
    mean_x = mean(series)

    numerator = sum((series[k+1:n] .- mean_x) .* (series[1:n-k] .- mean_x))
    denominator = sum((series .- mean_x) .^ 2)

    return numerator / denominator
end
function estimate_ar_coefficients(series, p)
    n = length(series)
    autocorr_values = [autocorrelation(series, k) for k in 0:p]
    R = [autocorr_values[abs(i-j)+1] for i in 1:p, j in 1:p]
    r = autocorr_values[2:p+1]
    phi = R \ r
    return phi
end
function compute_aic_bic(y, p)
    n = length(y)
    phi = estimate_ar_coefficients(y, p)
    y_pred = zeros(n-p)
    for t in 1:(n-p)
        y_pred[t] = sum(phi .* reverse(y[t:t+p-1]))
    end
    residuals = y[p+1:end] .- y_pred
    σ² = mean(residuals .^ 2)
    log_likelihood = -0.5 * n * log(2π * σ²) - sum(residuals .^ 2) / (2 * σ²)
    k = p + 1
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * log(n)

    return aic, bic
end
function compute_aic_bic_ma(residuals, q)
    n = length(residuals)
    theta = estimate_ma_coefficients(residuals, q)
    residuals_pred = zeros(n-q)
    
    for t in 1:(n-q)
        residuals_pred[t] = sum(theta .* reverse(residuals[t:t+q-1]))
    end
    
    new_residuals = residuals[q+1:end] .- residuals_pred
    σ² = mean(new_residuals .^ 2)
    log_likelihood = -0.5 * n * log(2π * σ²) - sum(new_residuals .^ 2) / (2 * σ²)
    
    k = q + 1
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * log(n)
    
    return aic, bic
end

function compute_residuals(series, phi)
    n = length(series)
    p = length(phi)
    residuals = zeros(n)

    for i in (p+1):n
        predicted = sum(phi[j] * series[i - j] for j in 1:p)
        residuals[i] = series[i] - predicted
    end

    return residuals[p+1:end]  # Remove initial p values
end
function estimate_ma_coefficients(residuals, q)
    acf_values = [autocorrelation(residuals, k) for k in 0:q]
    R = [acf_values[abs(i-j)+1] for i in 1:q, j in 1:q]
    r = acf_values[2:q+1]
    theta = R \ r
    return theta
end

function predict_arima(series, phi, theta, steps)
    n = length(series)
    p = length(phi)
    q = length(theta)

    predictions = copy(series)
    residuals = zeros(n)

    for i in (p+1):n
        predicted = sum(phi[j] * series[i - j] for j in 1:p if (i - j) > 0)
        residuals[i] = series[i] - predicted
    end

    for i in 1:steps
        ar_part = sum(phi[j] * predictions[n + i - j] for j in 1:p if (n + i - j) <= length(predictions) && (n + i - j) > 0; init=0)
        ma_part = sum(theta[j] * residuals[n + i - j] for j in 1:q if (n + i - j) <= length(residuals) && (n + i - j) > 0; init=0)
        pred = ar_part+ma_part
        push!(predictions, pred)
    end

    return predictions[(n+1):end]
end


df = CSV.read("daily-minimum-temperatures-in-me.csv", DataFrame)
df.Date = Date.(df.Date, dateformat"m/d/yyyy")
println(first(df, 5))

p=plot(df.Date, df[:,2], title="Daily Minimum Temperatures", label="Temperature", lw=2, xlabel="Date", ylabel="Temperature (°C)")
savefig(p, "temperature_plot.png")

rolling_mean, rolling_std = rolling_stats(df[:,2], 30)
p1=plot(df.Date, df[:,2], label="Original Data", lw=2, color=:blue)
plot!(df.Date, rolling_mean, label="Rolling Mean", lw=2, color=:red)
plot!(df.Date, rolling_std, label="Rolling Std Dev", lw=2, color=:green, title="Rolling Mean & Std Dev")
savefig(p1, "rolling_mean_std_plot.png")

rolling_mean_50, rolling_std_50 = rolling_stats(df[:,2][1:50], 30)
p_50 = plot(df.Date[1:50], df[:,2][1:50], label="Original Data", lw=2, color=:blue)
plot!(df.Date[1:50], rolling_mean_50, label="Rolling Mean", lw=2, color=:red)
plot!(df.Date[1:50], rolling_std_50, label="Rolling Std Dev", lw=2, color=:green, title="Rolling Mean & Std Dev (First 50 Points)")
savefig(p_50, "rolling_mean_std_50_plot.png")

df.Differenced = [missing; diff(df[:,2])]
rolling_mean_diff, rolling_std_diff = rolling_stats(collect(skipmissing(df.Differenced)), 30)
p_diff = plot(df.Date[2:end], skipmissing(df.Differenced), label="Differenced Data", lw=2, color=:blue)
plot!(df.Date[2:end], rolling_mean_diff, label="Rolling Mean", lw=2, color=:red)
plot!(df.Date[2:end], rolling_std_diff, label="Rolling Std Dev", lw=2, color=:green, title="Differenced Data - Rolling Mean & Std Dev")
savefig(p_diff, "differenced_stationarity_plot.png")

adf_test = ADFTest(collect(skipmissing(df[:,2])), :none, 0)
println(adf_test)
adf_test = ADFTest(collect(skipmissing(df.Differenced)), :none, 0)
println(adf_test)

train_size = length(df[:,2]) - 30
train_series = df[:,2][1:train_size]
test_series = df[:,2][train_size+1:end]

p_values = 5:50
results = DataFrame(p=p_values, AIC=zeros(length(p_values)), BIC=zeros(length(p_values)))
for i in eachindex(p_values)
    results.AIC[i], results.BIC[i] = compute_aic_bic(df[:,2], p_values[i])
end
best_p_aic = results.p[argmin(results.AIC)]
best_p_bic = results.p[argmin(results.BIC)]
println("Best AR order based on AIC: p = ", best_p_aic)
println("Best AR order based on BIC: p = ", best_p_bic)
best_p_p = plot(p_values, results.AIC, label="AIC", xlabel="p (AR Order)", ylabel="Score", linewidth=2)
plot!(p_values, results.BIC, label="BIC", linewidth=2, linestyle=:dash)
savefig(best_p_p, "p_plot.png")

p = 26 
phi = estimate_ar_coefficients(train_series, p)

residuals = compute_residuals(train_series, phi)
q_values = 0:10
results_q = DataFrame(q=q_values, AIC=zeros(length(q_values)), BIC=zeros(length(q_values)))
for i in eachindex(q_values)
    results_q.AIC[i], results_q.BIC[i] = compute_aic_bic_ma(residuals, q_values[i])
end
best_q_aic = results_q.q[argmin(results_q.AIC)]
best_q_bic = results_q.q[argmin(results_q.BIC)]
println("Best MA order based on AIC: q = ", best_q_aic)
println("Best MA order based on BIC: q = ", best_q_bic)
q_plot = plot(q_values, results_q.AIC, label="AIC", xlabel="q (MA Order)", ylabel="Score", linewidth=2)
plot!(q_values, results_q.BIC, label="BIC", linewidth=2, linestyle=:dash)
savefig(q_plot, "q_plot.png")
theta = estimate_ma_coefficients(residuals, 0)

forecast = predict_arima(train_series, phi, theta, 30)
println("Forecast length: ", length(forecast))
test_dates = df.Date[train_size+1:end]
p_forecast = plot(test_dates,test_series, label="Actual Data")
plot!(test_dates,forecast, label="ARIMA(26,0,5) Forecast",color=:green)    
savefig(p_forecast, "forecast.png")