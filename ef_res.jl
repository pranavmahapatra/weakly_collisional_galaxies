using LinearAlgebra
using SparseArrays
using Arpack
using SpecialFunctions
using Printf
using Plots
using Quadmath
using Base.Threads
using LaTeXStrings

function main()

# ==========================================
# Global Plot Style (Option 2)
# ==========================================
gr()

sea_green = "#00607e"
olive     = "#0f5655"

default(
    fontfamily = "Computer Modern",
    framestyle = :box,
    linewidth = 2,
    guidefontsize = 13,
    tickfontsize = 10,
    legendfontsize = 10,
    grid = true,
    gridalpha = 0.15,
    gridlinewidth = 0.5,
    dpi = 300
)

# ==========================================
# Parameters
# ==========================================

Precision = 34 #just informational (Float64 used)

alpha = 0.95
mu    = 5e-6
N     = 3
n     = round(Int, N/mu)

@printf("alpha = %.2f\tmu = %.2g\tn = %d\tPrecision = %d\n",
        alpha, mu, n, Precision)

# ==========================================
# Define Z and D
# ==========================================

Z(x) = sqrt(pi) * exp(x^2) * (1 - erf(x))
D(x) = 1 - alpha * (1 - x * Z(x))

# ==========================================
# Find G
# ==========================================

G0 = (1 - 1/alpha) / sqrt(pi)

function solve_G(G0)
    G = G0
    for k in 1:100
        f = D(G)
        if abs(f) < 1e-14
            break
        end
        h  = 1e-8
        df = (D(G+h)-D(G-h))/(2h)
        G -= f/df
    end
    return -G
end

G = solve_G(G0)
@printf("G_Landau = %.15g\n", G)

# ==========================================
# Matrix S
# ==========================================

rows = Int[]
cols = Int[]
vals = ComplexF64[]   # FIXED

for j in 1:n-1
    push!(rows,j); push!(cols,j+1); push!(vals,sqrt(j) + 0im)   # FIXED
    push!(rows,j+1); push!(cols,j); push!(vals,-sqrt(j) + 0im)  # FIXED
end

S = sparse(rows,cols,vals,n,n)
S[2,1] = -1 + alpha
S .= -S / sqrt(2.0)

for i in 1:n
    S[i,i] += mu*(i-1)
end

# ==========================================
# Eigenvalue
# ==========================================

#sigma = Complex(G, -0.05)   # FIXED
sigma = Complex(G, -0.01093 * mu)
λ, U = eigs(S; nev=1, sigma=sigma, tol=1e-12, maxiter=10^6)

D_eig = λ[1]
U = U[:,1]

@printf("D_ev_re  = %.15g\n", real(D_eig))
@printf("D_ev_im  = %.15g\n", imag(D_eig))

# ==========================================
# Eigenvector plot
# ==========================================

i = 1:n

p_eig = plot(i, abs.(real.(U)),   # FIXED
    xscale=:log10, yscale=:log10,
    color=sea_green, lw=2.5, label="Re V")

plot!(p_eig, i, imag.(U),
    color=sea_green, lw=2,
    alpha=0.6, linestyle=:dash, label="Im V")

plot!(p_eig, i, 0.98e-2 ./ i.^0.28,
    color=olive, linestyle=:dot, label="reference")

xlabel!(L"n+1")
ylabel!(L"A_n")

savefig(p_eig, "eigenvector.pdf")

# ==========================================
# Compute H
# ==========================================

delta = (mu/2)^(1/3)
Gamma = abs(G)/delta   # FIXED

q = range(0,2.5*sqrt(Gamma),length=200001)
dq = q[2]-q[1]

weights = ones(length(q))
weights[1] = weights[end] = 0.5
weights .*= dq

fe = exp.(-q.^3/3 .+ q.*Gamma)
eSq = fe .* weights .* Gamma

p_fe = plot(q, fe,
    yscale=:log10, color=sea_green, lw=2.5, label="")

xlabel!(L"q")
ylabel!(L"f_e(q)")

savefig(p_fe, "fe_vs_q.pdf")

# ==========================================
# H computation
# ==========================================

xx = 10 .^ range(-6,1,length=2001)
Uvals = xx/delta

H = zeros(ComplexF64,length(Uvals))

@threads for k in eachindex(Uvals)
    tmp = exp.(-1im .* q .* Uvals[k])
    H[k] = dot(tmp, eSq)
end

cf = -alpha/sqrt(pi) .* xx ./ (xx .+ 1im*G)

# ==========================================
# Plot H
# ==========================================

p_H = plot(xx, real.(H),
    xscale=:log10,
    color=sea_green, lw=2.5, label="Re H")

plot!(p_H, xx, imag.(H),
    color=sea_green, lw=2, alpha=0.6,
    linestyle=:dash, label="Im H")

plot!(p_H, xx, real.(cf),
    color=olive, linestyle=:dot, label="Re analytic")

plot!(p_H, xx, imag.(cf),
    color=olive, linestyle=:dashdot, label="Im analytic")

xlabel!(L"u")
savefig(p_H, "H_compare.pdf")

# ==========================================
# Reconstruct eigenfunction
# ==========================================

x = 10 .^ range(-6,1,length=10001)

h0 = exp.(-x.^2) ./ sqrt(sqrt(pi))
h1 = sqrt(2.0) .* x .* h0

her_prev_prev = h0
her_prev = h1

V = U ./ U[1] ./ pi^(0.25)

Her_x = zeros(ComplexF64,length(x))

for j in 1:length(V)
    if j == 1
        her = h0
    elseif j == 2
        her = h1
    else
        her = sqrt(2/(j-1)) .* x .* her_prev .-
              sqrt((j-2)/(j-1)) .* her_prev_prev
        her_prev_prev = her_prev
        her_prev = her
    end
    Her_x .+= her .* V[j] .* (1im)^(j-1)
end

cf_x = -alpha/sqrt(pi) .* x .* exp.(-x.^2) ./ (x .+ 1im*G)
EF = -Her_x

# ==========================================
# Final plot
# ==========================================

p_final = plot(x, abs.(real.(EF)),   # FIXED
    xscale=:log10, yscale=:log10,
    color=sea_green, lw=2.5, label="Re EF")

plot!(p_final, x, imag.(EF),
    color=sea_green, lw=2,
    alpha=0.6, linestyle=:dash, label="Im EF")

plot!(p_final, x, real.(cf_x),
    color=olive, linestyle=:dot, label="Re analytic")

plot!(p_final, x, imag.(cf_x),
    color=olive, linestyle=:dashdot, label="Im analytic")

xlabel!(L"u")
savefig(p_final, "Final_log.pdf")

# ==========================================
# Step 9: Symlog plot
# ==========================================

function symlog(x; linthresh=1e-3)
    sign.(x) .* log10.(1 .+ abs.(x)./linthresh)
end

u_plot = symlog(x)
ReEF = symlog(real.(EF))
ImEF = symlog(imag.(EF))
ReCF = symlog(real.(cf_x))
ImCF = symlog(imag.(cf_x))

num_real = "#683659"
num_imag = "#625185"

p_symlog = plot(u_plot, ReEF,
    color=num_real, alpha=0.7, lw=3.5, label="Re matrix")

plot!(p_symlog, u_plot, ImEF,
    color=num_imag, lw=2,
    linestyle=:dot, label="Im matrix")

plot!(p_symlog, u_plot, ReCF,
    color=sea_green, lw=2.5, label="Re analytic")

plot!(p_symlog, u_plot, ImCF,
    color=olive, lw=2,
    alpha=0.6, linestyle=:dot, label="Im analytic")

xlabel!(L"u\ (\mathrm{symlog})")
ylabel!(L"g(u)\ (\mathrm{symlog})")

savefig(p_symlog, "Figure3_style_mu_t$(mu).pdf")

end

main()
