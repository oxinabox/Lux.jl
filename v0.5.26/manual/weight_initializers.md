
<a id='Initializing-Weights'></a>

# Initializing Weights


`WeightInitializers.jl` provides common weight initialization schemes for deep learning models.


```julia
using WeightInitializers, Random

# Fixing rng
rng = Random.MersenneTwister(42)
```


```
Random.MersenneTwister(42)
```


```julia
# Explicit rng call
weights = kaiming_normal(rng, 2, 5)
```


```
2×5 Matrix{Float32}:
 -0.351662   0.0171745   1.12442   -0.296372   -1.67094
 -0.281053  -0.18941    -0.724099   0.0987538   0.634549
```


```julia
# Default rng call
weights = kaiming_normal(2, 5)
```


```
2×5 Matrix{Float32}:
 -0.227513  -0.265372   0.265788  1.29955  -0.192836
  0.687611   0.454679  -0.433656  0.20548   0.292002
```


```julia
# Passing kwargs (if needed) with explicit rng call
weights_cl = kaiming_normal(rng; gain=1.0)
weights = weights_cl(2, 5)
```


```
2×5 Matrix{Float32}:
 0.484056   0.231723   0.164379   0.306147   0.18365
 0.0836414  0.666965  -0.396323  -0.711329  -0.382971
```


```julia
# Passing kwargs (if needed) with default rng call
weights_cl = kaiming_normal(; gain=1.0)
weights = weights_cl(2, 5)
```


```
2×5 Matrix{Float32}:
 -0.160876  -0.187646   0.18794   0.918918  -0.136356
  0.486214   0.321506  -0.306641  0.145296   0.206476
```


To generate weights directly on GPU, pass in a `CUDA.RNG`. (Note that this is currently implemented only for NVIDIA GPUs)


```julia
using LuxCUDA

weights = kaiming_normal(CUDA.default_rng(), 2, 5)
```


```
2×5 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 -0.542337    0.472984  0.0543483   1.07533   0.822437
 -0.0287523  -0.100324  0.433427   -0.298834  0.35999
```


You can also generate Complex Numbers:


```julia
weights = kaiming_normal(CUDA.default_rng(), ComplexF32, 2, 5)
```


```
2×5 CuArray{ComplexF32, 2, CUDA.Mem.DeviceBuffer}:
 -0.0358299-0.0542666im  -0.263094-0.0869567im  …  -0.433261+0.465476im
 -0.0356132+0.114221im   -0.274497+0.417537im       0.371823-0.279039im
```


<a id='Quick-examples'></a>

## Quick examples


The package is meant to be working with deep learning libraries such as (F)Lux. All the methods take as input the chosen `rng` type and the dimension for the array.


```julia
weights = init(rng, dims...)
```


The `rng` is optional, if not specified a default one will be used.


```julia
weights = init(dims...)
```


If there is the need to use keyword arguments the methods can be called with just the `rng`  (optionally) and the keywords to get in return a function behaving like the two examples above.


```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)

# Or

weights_init = init(; kwargs...)
weights = weights_init(dims...)
```
