using Pkg
Pkg.activate(".")

# Install required packages if needed
required_packages = ["CSV", "DataFrames", "Dates", "Plots", 
                    "StatsBase", "HypothesisTests", "LinearAlgebra"]

for pkg in required_packages
    try
        using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
    end
end

include("src/tempr.jl")
using .TemperatureARIMA
results = analyze_dataset("data/daily-minimum-temperatures-in-me.csv", p_range=5:1:50,q_range=0:1:10)
println("\nModel Summary:")
println("AR order (p): ", results.model.p)
println("MA order (q): ", results.model.q)
println("AIC: ", round(results.model.aic; digits=2))
println("BIC: ", round(results.model.bic; digits=2))
save_plots(results, "output_plots")
println("\nSaved plots to 'output_plots/' directory")
