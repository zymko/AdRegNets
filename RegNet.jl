using Flux


"""
    input1 : û ∈ R^{nsensors x ts} (noisy)
    input2 : u ∈ R^{nsensors x ts} (simulation)
    output : scalar ∈ R
    loss : |model(input1) - model(input2)|
""" 

"""
xdata1 = Array{Float32,2}(undef,nsensors*ts,BS)
xdata2 = Array{Float32,2}(undef,nsensors*ts,BS)
"""
DL = 2*nsensors*ts
model = Chain(
    Dense(nsensors*ts => DL,relu),
    Dense(DL => DL,relu),
    Dense(DL => DL,relu),
    Dense(DL => DL,relu),
    Dense(DL => DL,relu),
    Dense(DL => DL,relu),
   Dense(DL => 1,relu)
)