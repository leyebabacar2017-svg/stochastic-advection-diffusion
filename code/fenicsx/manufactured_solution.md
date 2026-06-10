# Manufactured Solution Test Case

## Domain

Ω = (0,1)²

## Final Time

T = 1.0

## Velocity Field

b = (1,0)

## Diffusion Coefficient

kappa = 0.01

## Exact Solution

u(x,y,t) = exp(-t) sin(pi x) sin(pi y)

## Time Derivative

∂u/∂t = -exp(-t) sin(pi x) sin(pi y)

## Gradient

∇u =
[
 pi exp(-t) cos(pi x) sin(pi y),
 pi exp(-t) sin(pi x) cos(pi y)
]

## Advection Term

b·∇u =
pi exp(-t) cos(pi x) sin(pi y)

## Laplacian

Δu =
-2 pi² exp(-t) sin(pi x) sin(pi y)

## Source Term

f(x,y,t) =
-exp(-t) sin(pi x) sin(pi y)
+ pi exp(-t) cos(pi x) sin(pi y)
+ 2 kappa pi² exp(-t) sin(pi x) sin(pi y)

## Boundary Conditions

u = 0 on ∂Ω

## Initial Condition

u(x,y,0) =
sin(pi x) sin(pi y)

## Expected Convergence

P1 finite elements

||u - uh||_L2 = O(h²)

Expected EOC ≈ 2
