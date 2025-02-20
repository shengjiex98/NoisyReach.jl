using ControlSystemsBase
using LinearAlgebra

"""
delay_lqr(sys, h; Q=I, R=I)

LQR controller design of a delayed discrete state-space model. Q and R matrices
are default to be identity matrices.
"""
function delay_lqr(sys::AbstractStateSpace{<:Continuous}, h::Float64; Q=I, R=I)
    sysd_delay = c2d(sys, h) |> augment
    lqr(sysd_delay, Q, R)
end

"""
pole_place(sys, h; p=0.9)

Pole placement controller design of a delayed discrete state-space model. Note
this only works for single input single output (SISO) systems. I.e., y and u 
must have dimensionality of 1.
"""
function pole_place(sys::AbstractStateSpace{<:Continuous}, h::Float64; p=0.9)
    sysd_delay = c2d(sys, h) |> augment
    place(sysd_delay, vcat([0], fill(p, size(sys.A)[1])))
end

"""
augment(sysd)

Augment the discrete state-space model `sysd` with a one-period delay
"""
function augment(sysd::AbstractStateSpace{<:Discrete})
    p = size(sysd.A, 1)
    q = size(sysd.B, 2)
    r = size(sysd.C, 1)
    A = [sysd.A sysd.B; zeros(q, p+q)]
    B = [zeros(p, q); I]
    C = [sysd.C zeros(r, q)]
    D = sysd.D
    ss(A, B, C, D, sysd.Ts)
end
