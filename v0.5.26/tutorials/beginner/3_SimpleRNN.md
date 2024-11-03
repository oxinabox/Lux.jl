


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


<a id='Dataset'></a>

## Dataset


We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq*len × batch*size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.


```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(
                                 d[1][:, (sequence_length + 1):end], :, sequence_length, 1)
                             for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


<a id='Creating-a-Classifier'></a>

## Creating a Classifier


We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.


We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.


To understand more about container layers, please look at [Container Layer](../../manual/interface#Container-Layer).


```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](../../api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](../../api/Lux/layers#Lux.Dense).


```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims), Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##225".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(
        x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(Lux._eachslice(x, Val(2)))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


<a id='Defining-Accuracy,-Loss-and-Optimiser'></a>

## Defining Accuracy, Loss and Optimiser


Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.


```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.Adam(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxCore/t4mG0/src/LuxCore.jl:13
Epoch [1]: Loss 0.5624044
Epoch [1]: Loss 0.5098107
Epoch [1]: Loss 0.4782407
Epoch [1]: Loss 0.45271456
Epoch [1]: Loss 0.42848754
Epoch [1]: Loss 0.39648807
Epoch [1]: Loss 0.38447148
Validation: Loss 0.37469032 Accuracy 1.0
Validation: Loss 0.37322304 Accuracy 1.0
Epoch [2]: Loss 0.37656733
Epoch [2]: Loss 0.34751013
Epoch [2]: Loss 0.32265812
Epoch [2]: Loss 0.31773078
Epoch [2]: Loss 0.30178046
Epoch [2]: Loss 0.29371923
Epoch [2]: Loss 0.26509047
Validation: Loss 0.26227957 Accuracy 1.0
Validation: Loss 0.2614018 Accuracy 1.0
Epoch [3]: Loss 0.26297706
Epoch [3]: Loss 0.24777621
Epoch [3]: Loss 0.23309101
Epoch [3]: Loss 0.21997863
Epoch [3]: Loss 0.20669936
Epoch [3]: Loss 0.19996515
Epoch [3]: Loss 0.1841955
Validation: Loss 0.18230529 Accuracy 1.0
Validation: Loss 0.18189308 Accuracy 1.0
Epoch [4]: Loss 0.18110725
Epoch [4]: Loss 0.16988924
Epoch [4]: Loss 0.16687642
Epoch [4]: Loss 0.15613934
Epoch [4]: Loss 0.14495401
Epoch [4]: Loss 0.13959348
Epoch [4]: Loss 0.14009625
Validation: Loss 0.1298093 Accuracy 1.0
Validation: Loss 0.12957372 Accuracy 1.0
Epoch [5]: Loss 0.129293
Epoch [5]: Loss 0.12119539
Epoch [5]: Loss 0.118239075
Epoch [5]: Loss 0.11132997
Epoch [5]: Loss 0.10825237
Epoch [5]: Loss 0.10101448
Epoch [5]: Loss 0.09488228
Validation: Loss 0.094474435 Accuracy 1.0
Validation: Loss 0.09428545 Accuracy 1.0
Epoch [6]: Loss 0.09447167
Epoch [6]: Loss 0.089034565
Epoch [6]: Loss 0.08551489
Epoch [6]: Loss 0.08025659
Epoch [6]: Loss 0.0783989
Epoch [6]: Loss 0.07393928
Epoch [6]: Loss 0.07684426
Validation: Loss 0.06999335 Accuracy 1.0
Validation: Loss 0.069791615 Accuracy 1.0
Epoch [7]: Loss 0.067652635
Epoch [7]: Loss 0.0667322
Epoch [7]: Loss 0.06356665
Epoch [7]: Loss 0.059426323
Epoch [7]: Loss 0.060501955
Epoch [7]: Loss 0.05494561
Epoch [7]: Loss 0.05213772
Validation: Loss 0.05245585 Accuracy 1.0
Validation: Loss 0.05225057 Accuracy 1.0
Epoch [8]: Loss 0.05155548
Epoch [8]: Loss 0.048422813
Epoch [8]: Loss 0.046409503
Epoch [8]: Loss 0.047176555
Epoch [8]: Loss 0.04284371
Epoch [8]: Loss 0.04291442
Epoch [8]: Loss 0.03753831
Validation: Loss 0.039685704 Accuracy 1.0
Validation: Loss 0.03947328 Accuracy 1.0
Epoch [9]: Loss 0.03811001
Epoch [9]: Loss 0.039114825
Epoch [9]: Loss 0.034543037
Epoch [9]: Loss 0.034987718
Epoch [9]: Loss 0.031134337
Epoch [9]: Loss 0.031973712
Epoch [9]: Loss 0.033469804
Validation: Loss 0.030509014 Accuracy 1.0
Validation: Loss 0.03029498 Accuracy 1.0
Epoch [10]: Loss 0.028404433
Epoch [10]: Loss 0.02849824
Epoch [10]: Loss 0.028236978
Epoch [10]: Loss 0.026689265
Epoch [10]: Loss 0.025433183
Epoch [10]: Loss 0.024510387
Epoch [10]: Loss 0.025881458
Validation: Loss 0.024084326 Accuracy 1.0
Validation: Loss 0.023881925 Accuracy 1.0
Epoch [11]: Loss 0.022782471
Epoch [11]: Loss 0.021218918
Epoch [11]: Loss 0.022367373
Epoch [11]: Loss 0.020566797
Epoch [11]: Loss 0.020719983
Epoch [11]: Loss 0.020877361
Epoch [11]: Loss 0.021121968
Validation: Loss 0.019639663 Accuracy 1.0
Validation: Loss 0.019455193 Accuracy 1.0
Epoch [12]: Loss 0.018748632
Epoch [12]: Loss 0.018756729
Epoch [12]: Loss 0.017406803
Epoch [12]: Loss 0.017840087
Epoch [12]: Loss 0.016908139
Epoch [12]: Loss 0.016252473
Epoch [12]: Loss 0.01731596
Validation: Loss 0.016500186 Accuracy 1.0
Validation: Loss 0.016335584 Accuracy 1.0
Epoch [13]: Loss 0.015840273
Epoch [13]: Loss 0.01476455
Epoch [13]: Loss 0.015247166
Epoch [13]: Loss 0.015028993
Epoch [13]: Loss 0.014733236
Epoch [13]: Loss 0.014241097
Epoch [13]: Loss 0.014452073
Validation: Loss 0.014242053 Accuracy 1.0
Validation: Loss 0.014094272 Accuracy 1.0
Epoch [14]: Loss 0.01385471
Epoch [14]: Loss 0.012879122
Epoch [14]: Loss 0.012776699
Epoch [14]: Loss 0.013185832
Epoch [14]: Loss 0.013100159
Epoch [14]: Loss 0.012804031
Epoch [14]: Loss 0.010919111
Validation: Loss 0.012564421 Accuracy 1.0
Validation: Loss 0.012430232 Accuracy 1.0
Epoch [15]: Loss 0.012404502
Epoch [15]: Loss 0.011974543
Epoch [15]: Loss 0.011171148
Epoch [15]: Loss 0.011695917
Epoch [15]: Loss 0.011068879
Epoch [15]: Loss 0.011169072
Epoch [15]: Loss 0.010676404
Validation: Loss 0.011245139 Accuracy 1.0
Validation: Loss 0.0111213755 Accuracy 1.0
Epoch [16]: Loss 0.0103202425
Epoch [16]: Loss 0.010864984
Epoch [16]: Loss 0.010947377
Epoch [16]: Loss 0.010321761
Epoch [16]: Loss 0.009596483
Epoch [16]: Loss 0.009919312
Epoch [16]: Loss 0.011470696
Validation: Loss 0.010187076 Accuracy 1.0
Validation: Loss 0.010072272 Accuracy 1.0
Epoch [17]: Loss 0.009743669
Epoch [17]: Loss 0.009909393
Epoch [17]: Loss 0.009385798
Epoch [17]: Loss 0.00909966
Epoch [17]: Loss 0.0098735355
Epoch [17]: Loss 0.008863858
Epoch [17]: Loss 0.008034686
Validation: Loss 0.009295378 Accuracy 1.0
Validation: Loss 0.009188581 Accuracy 1.0
Epoch [18]: Loss 0.009322809
Epoch [18]: Loss 0.009115167
Epoch [18]: Loss 0.00875639
Epoch [18]: Loss 0.008568138
Epoch [18]: Loss 0.008217154
Epoch [18]: Loss 0.008065594
Epoch [18]: Loss 0.0074024354
Validation: Loss 0.008547563 Accuracy 1.0
Validation: Loss 0.008447599 Accuracy 1.0
Epoch [19]: Loss 0.008119163
Epoch [19]: Loss 0.008150578
Epoch [19]: Loss 0.007892936
Epoch [19]: Loss 0.0079540685
Epoch [19]: Loss 0.007765579
Epoch [19]: Loss 0.0076217414
Epoch [19]: Loss 0.00842788
Validation: Loss 0.007906783 Accuracy 1.0
Validation: Loss 0.007812313 Accuracy 1.0
Epoch [20]: Loss 0.007519668
Epoch [20]: Loss 0.007204627
Epoch [20]: Loss 0.0076506916
Epoch [20]: Loss 0.007218949
Epoch [20]: Loss 0.007296878
Epoch [20]: Loss 0.0072716842
Epoch [20]: Loss 0.0071195043
Validation: Loss 0.0073452634 Accuracy 1.0
Validation: Loss 0.0072564203 Accuracy 1.0
Epoch [21]: Loss 0.007689229
Epoch [21]: Loss 0.0062611247
Epoch [21]: Loss 0.007043995
Epoch [21]: Loss 0.0068024453
Epoch [21]: Loss 0.00628228
Epoch [21]: Loss 0.0069219396
Epoch [21]: Loss 0.0069034426
Validation: Loss 0.006848287 Accuracy 1.0
Validation: Loss 0.0067645134 Accuracy 1.0
Epoch [22]: Loss 0.006792832
Epoch [22]: Loss 0.006671066
Epoch [22]: Loss 0.00632557
Epoch [22]: Loss 0.006467769
Epoch [22]: Loss 0.005804547
Epoch [22]: Loss 0.0063206526
Epoch [22]: Loss 0.0060163485
Validation: Loss 0.00640596 Accuracy 1.0
Validation: Loss 0.006326798 Accuracy 1.0
Epoch [23]: Loss 0.0055961874
Epoch [23]: Loss 0.005740798
Epoch [23]: Loss 0.006456722
Epoch [23]: Loss 0.005943247
Epoch [23]: Loss 0.005862677
Epoch [23]: Loss 0.0060421973
Epoch [23]: Loss 0.006754979
Validation: Loss 0.006016548 Accuracy 1.0
Validation: Loss 0.005941656 Accuracy 1.0
Epoch [24]: Loss 0.005570687
Epoch [24]: Loss 0.0062211556
Epoch [24]: Loss 0.005914542
Epoch [24]: Loss 0.005597069
Epoch [24]: Loss 0.0055237375
Epoch [24]: Loss 0.0049003996
Epoch [24]: Loss 0.005509998
Validation: Loss 0.005656875 Accuracy 1.0
Validation: Loss 0.0055859853 Accuracy 1.0
Epoch [25]: Loss 0.0056671468
Epoch [25]: Loss 0.0053874664
Epoch [25]: Loss 0.0047080903
Epoch [25]: Loss 0.0051407716
Epoch [25]: Loss 0.0053881737
Epoch [25]: Loss 0.005291883
Epoch [25]: Loss 0.0058048773
Validation: Loss 0.0053372784 Accuracy 1.0
Validation: Loss 0.0052700434 Accuracy 1.0

```


<a id='Saving-the-Model'></a>

## Saving the Model


We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model


```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model


```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
```


<a id='Appendix'></a>

## Appendix


```julia
using InteractiveUtils
InteractiveUtils.versioninfo()
if @isdefined(LuxCUDA) && CUDA.functional(); println(); CUDA.versioninfo(); end
if @isdefined(LuxAMDGPU) && LuxAMDGPU.functional(); println(); AMDGPU.versioninfo(); end
```


```
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 48 × AMD EPYC 7402 24-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)
Environment:
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-4/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.4
NVIDIA driver 550.54.14

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+550.54.14

Julia packages: 
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0

Toolchain:
- Julia: 1.10.2
- LLVM: 15.0.7

Environment:
- JULIA_CUDA_HARD_MEMORY_LIMIT: 25%

1 device:
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.359 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
