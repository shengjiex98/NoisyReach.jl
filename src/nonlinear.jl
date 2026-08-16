
using ReachabilityAnalysis
using Plots

# Scenario 1: the reference (tracking) location is on the unit circle, starting from the positive x-axis (1, 0), and moves at a constant speed of 0.5 rad/s counterclockwise.
function f!(ds, s, param, t)
    # Current posture and velocity. p = [x, y, θ], q = [v, ω]
    p = s[1:3]
    # q = s[4:5]

    # Reference posture and velocity
    # pr = [cos(0.5 * t)
    #       sin(0.5 * t)
    #       0.5 * t]
    pr = p
    qr = [0.5, 0.5]

    # Maximum velocity/acceleration
    # vmax, ωmax = 0.4, 0.8
    # amax, αmax = 0.5, 5

    # Error posture
    pe = pr - p
    
    # Control law decides the target velocity
    Kx = 10
    Ky = 6.4e-3
    Kθ = 0.16

    qt = [qr[1] * cos(pe[3]) + Kx * pe[1]
          qr[2] + qr[1] * (Ky * pe[2] + Kθ * sin(pe[3]))]

    # Naive case: no velocity/acceleration constraints
    q = qt

    # System dynamics
    ds .= [cos(p[3]) * q[1]
           sin(p[3]) * q[1]
           q[2]]

end

S0 = Hyperrectangle([1.0, 0.0, 1/2*π], [0.0, 0.2, 0.0])

prob = @ivp(s' = f!(s), s(0) ∈ S0, dim=3)
sol = solve(prob, alg=TMJets21a(abstol=1e-6), T=10)

fig = plot(sol, vars=(1, 2), xlab='x', ylab='y')
