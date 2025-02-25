module TemperatureARIMA

using CSV, DataFrames, Dates, Plots, StatsBase, HypothesisTests, LinearAlgebra
export ARIMAModel, analyze_dataset, fit!, forecast, save_plots

struct ARIMAModel
    p::Int
    d::Int
    q::Int
    phi::Vector{Float64}
    theta::Vector{Float64}
    residuals::Vector{Float64}
    aic::Float64
    bic::Float64
end

module Utils
using ..TemperatureARIMA: ARIMAModel
using Dates, Statistics, LinearAlgebra
function exponential_smoothing(series::Vector{Float64}, α::Float64)
    @assert 0 ≤ α ≤ 1 "Smoothing factor α must be ∈ [0,1]"
    smoothed = similar(series)
    smoothed[1] = series[1]
    for i in 2:length(series)
        smoothed[i] = α * series[i] + (1 - α) * smoothed[i-1]
    end
    return smoothed
end
function rolling_statistics(series::Vector{Float64}, window::Int)
    @assert window ≥ 1 "Window size must be ≥ 1"
    n = length(series)
    means = [mean(@view series[max(1,i-window+1):i]) for i in 1:n]
    stds = [std(@view series[max(1,i-window+1):i]) for i in 1:n]
    return means, stds
end
end 


module Core
using ..TemperatureARIMA: ARIMAModel
using ..Utils
using Dates, Statistics, LinearAlgebra, HypothesisTests

function compute_residuals(series::Vector{Float64}, phi::Vector{Float64})
    p = length(phi)
    n = length(series)
    residuals = zeros(n - p)
    for i in (p+1):n
        predicted = sum(phi .* reverse(series[i-p:i-1]))
        residuals[i-p] = series[i] - predicted
    end
    return residuals
end
function autocorrelation(series::Vector{Float64}, k::Int)
    n = length(series)
    μ = mean(series)
    
    numerator = sum((series[k+1:end] .- μ) .* (series[1:end-k] .- μ))
    denominator = sum((series .- μ).^2)
    
    return numerator / denominator
end
function compute_aic_bic(series::Vector{Float64}, p::Int)
    n = length(series)
    phi = estimate_ar_coefficients(series, p)
    residuals = compute_residuals(series, phi)
    σ² = var(residuals)
    
    log_likelihood = -n/2 * log(2π * σ²) - sum(residuals.^2)/(2σ²)
    
    k = p + 1  
    aic = -2*log_likelihood + 2*k
    bic = -2*log_likelihood + k*log(n)
    
    return aic, bic
end
function compute_aic_bic_ma(residuals::Vector{Float64}, q::Int)
    n = length(residuals)
    theta = estimate_ma_coefficients(residuals, q)
    ma_residuals = compute_ma_residuals(residuals, theta)
    σ² = var(ma_residuals)
    
    
    log_likelihood = -n/2 * log(2π * σ²) - sum(ma_residuals.^2)/(2σ²)
    
    k = q + 1 
    aic = -2*log_likelihood + 2*k
    bic = -2*log_likelihood + k*log(n)
    
    return aic, bic
end
function compute_ma_residuals(residuals, theta)
    q = length(theta)
    n = length(residuals)
    ma_residuals = zeros(n - q)
    
    for t in 1:(n-q)
        ma_residuals[t] = residuals[t+q] - sum(theta .* residuals[t+q-1:-1:t])
    end
    
    return ma_residuals
end
function fit!(model::ARIMAModel, train_series::Vector{Float64})
    phi = estimate_ar_coefficients(train_series, model.p)
    residuals = compute_residuals(train_series, phi)
    theta = estimate_ma_coefficients(residuals, model.q)
    return ARIMAModel(model.p, model.d, model.q, phi, theta, 
                    residuals, model.aic, model.bic)
end

function estimate_ar_coefficients(series::Vector{Float64}, p::Int)
    n = length(series)
    @assert p ≥ 1 "AR order must be ≥ 1"
    @assert n > p "Insufficient data for AR order"
    
    autocorr = [autocorrelation(series, k) for k in 0:p]
    R = [autocorr[abs(i-j)+1] for i in 1:p, j in 1:p]
    r = autocorr[2:p+1]
    return R \ r
end

function estimate_ma_coefficients(residuals::Vector{Float64}, q::Int)
    acf = [autocorrelation(residuals, k) for k in 0:q]
    R = [acf[abs(i-j)+1] for i in 1:q, j in 1:q]
    r = acf[2:q+1]
    return R \ r
end

function forecast(model::ARIMAModel, train_series::Vector{Float64}, steps::Int)
    n = length(train_series)
    predictions = copy(train_series)
    resids = zeros(n)
    for i in (model.p+1):n
        ar_part = sum(model.phi[j] * predictions[i-j] for j in 1:model.p)
        resids[i] = predictions[i] - ar_part
    end
    for i in 1:steps
        ar_term = sum(model.phi[j] * predictions[end-j+1] for j in 1:model.p;init = 0)
        ma_term = sum(model.theta[j] * resids[end-j+1] for j in 1:model.q;init = 0)
        new_pred = ar_term + ma_term
        push!(predictions, new_pred)
        push!(resids, 0.0)
    end
    
    return predictions[n+1:end]
end

end
module Visualization
using ..TemperatureARIMA: ARIMAModel
using Plots, Dates
function plot_series(dates::Vector{Date}, series::Vector{Float64}, title::String)
    plot(dates, series, 
        title=title, 
        xlabel="Date", 
        ylabel="Temperature (°C)", 
        legend=false,
        linewidth=2,
        color=:blue)
end
function plot_forecast_30days(test_dates::Vector{Date}, 
    actual::Vector{Float64},
    forecast::Vector{Float64})
    start_date = test_dates[1]
    end_date = test_dates[end]
    plot(test_dates, actual,
    label="Actual", 
    color=:blue,
    linewidth=2,
    xlims=(start_date, end_date),
    xticks=start_date:Day(7):end_date,
    xlabel="Date",
    ylabel="Temperature (°C)",
    title="Last 30 Days: Actual vs Forecast",
    legend=:topright)
    plot!(test_dates, forecast,
    label="Forecast",
    color=:red,
    linestyle=:dash,
    linewidth=2)
end

function plot_forecast(train_dates::Vector{Date}, 
                      test_dates::Vector{Date}, 
                      actual::Vector{Float64}, 
                      forecast::Vector{Float64})
    plot(train_dates, actual[1:length(train_dates)],
        label="Training Data",
        color=:blue)
    plot!(test_dates, actual[length(train_dates)+1:end],
        label="Actual Values",
        color=:green)
    plot!(test_dates, forecast,
        label="ARIMA Forecast",
        color=:red,
        linestyle=:dash,
        linewidth=2,
        title="Temperature Forecast Comparison")
end

end
function analyze_dataset(filepath::String; 
                        p_range::StepRange{Int,Int}=5:50,
                        q_range::StepRange{Int,Int}=0:10)
    df = CSV.read(filepath, DataFrame)
    df.Date = Date.(df.Date, dateformat"m/d/yyyy")
    temperature = Vector{Float64}(df[:,2])
    adf_test = pvalue(ADFTest(collect(skipmissing(temperature)), :none, 0))
    if(adf_test) > 0.05
        temperature = diff(temperature)
        adf_test = pvalue(ADFTest(collect(skipmissing(temperature)), :none, 0))
        if(adf_test) > 0.05
            temperature = diff(temperature)
        end
    end
    temperature = Utils.exponential_smoothing(temperature,0.3)
    train_size = length(temperature) - 30
    train_series = temperature[1:train_size]
    test_series = temperature[train_size+1:end]
    best_p, best_q = select_model_orders(train_series, p_range, q_range)
    model = ARIMAModel(best_p, 0, best_q, [], [], [], 0.0, 0.0)
    fitted_model = Core.fit!(model, train_series)
    forecast = Core.forecast(fitted_model, train_series, 30)
    return (
        data = df,
        model = fitted_model,
        forecast = forecast,
        test_series = test_series
    )
end

function select_model_orders(series::Vector{Float64}, p_range, q_range)
    aic_ar = zeros(length(p_range))
    bic_ar = zeros(length(p_range))
    for (i,p) in enumerate(p_range)
        aic_ar[i], bic_ar[i] = Core.compute_aic_bic(series, p)
    end
    best_p = p_range[argmin(aic_ar)]
    phi = Core.estimate_ar_coefficients(series, best_p)
    residuals = Core.compute_residuals(series, phi)
    aic_ma = zeros(length(q_range))
    bic_ma = zeros(length(q_range))
    for (i,q) in enumerate(q_range)
        aic_ma[i], bic_ma[i] = Core.compute_aic_bic_ma(residuals, q)
    end
    best_q = q_range[argmin(aic_ma)]
    return best_p, best_q
end

function save_plots(results, output_dir::String)
    isdir(output_dir) || mkdir(output_dir)
    p1 = Visualization.plot_series(results.data.Date, results.data[:,2], 
                                 "Daily Minimum Temperatures")
    savefig(p1, joinpath(output_dir, "temperature_series.png"))
    train_dates = results.data.Date[1:end-30]
    test_dates = results.data.Date[end-29:end]
    actual_last30 = results.data[:,2][end-29:end]
    p2 = Visualization.plot_forecast_30days(test_dates, 
                                          actual_last30, 
                                          results.forecast)
    savefig(p2, joinpath(output_dir, "30day_forecast_comparison.png"))
    
    return true
end
end 
