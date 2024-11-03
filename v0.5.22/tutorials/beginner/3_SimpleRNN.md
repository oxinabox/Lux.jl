


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
└ @ Lux /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/src/utils.jl:9
Epoch [1]: Loss 0.5616085
Epoch [1]: Loss 0.51188046
Epoch [1]: Loss 0.47491565
Epoch [1]: Loss 0.45123303
Epoch [1]: Loss 0.42882025
Epoch [1]: Loss 0.39880353
Epoch [1]: Loss 0.39746702
Validation: Loss 0.36402836 Accuracy 1.0
Validation: Loss 0.35980666 Accuracy 1.0
Epoch [2]: Loss 0.3709349
Epoch [2]: Loss 0.3508617
Epoch [2]: Loss 0.3290905
Epoch [2]: Loss 0.3213278
Epoch [2]: Loss 0.30338353
Epoch [2]: Loss 0.28443924
Epoch [2]: Loss 0.26675943
Validation: Loss 0.2552168 Accuracy 1.0
Validation: Loss 0.25296667 Accuracy 1.0
Epoch [3]: Loss 0.25410786
Epoch [3]: Loss 0.24643844
Epoch [3]: Loss 0.23045707
Epoch [3]: Loss 0.22147001
Epoch [3]: Loss 0.21344247
Epoch [3]: Loss 0.19665824
Epoch [3]: Loss 0.19125009
Validation: Loss 0.17826399 Accuracy 1.0
Validation: Loss 0.17720535 Accuracy 1.0
Epoch [4]: Loss 0.17948408
Epoch [4]: Loss 0.1691944
Epoch [4]: Loss 0.16233414
Epoch [4]: Loss 0.1555602
Epoch [4]: Loss 0.14873293
Epoch [4]: Loss 0.13983767
Epoch [4]: Loss 0.13529652
Validation: Loss 0.12699887 Accuracy 1.0
Validation: Loss 0.12636225 Accuracy 1.0
Epoch [5]: Loss 0.12750188
Epoch [5]: Loss 0.12236327
Epoch [5]: Loss 0.11529329
Epoch [5]: Loss 0.11171579
Epoch [5]: Loss 0.10509404
Epoch [5]: Loss 0.10314694
Epoch [5]: Loss 0.09797715
Validation: Loss 0.09197866 Accuracy 1.0
Validation: Loss 0.0913448 Accuracy 1.0
Epoch [6]: Loss 0.09216073
Epoch [6]: Loss 0.08970578
Epoch [6]: Loss 0.085136674
Epoch [6]: Loss 0.082330406
Epoch [6]: Loss 0.077736765
Epoch [6]: Loss 0.072385475
Epoch [6]: Loss 0.074550465
Validation: Loss 0.067530274 Accuracy 1.0
Validation: Loss 0.066841915 Accuracy 1.0
Epoch [7]: Loss 0.06806666
Epoch [7]: Loss 0.06566634
Epoch [7]: Loss 0.062743485
Epoch [7]: Loss 0.06133446
Epoch [7]: Loss 0.05708143
Epoch [7]: Loss 0.055899307
Epoch [7]: Loss 0.051570743
Validation: Loss 0.05011668 Accuracy 1.0
Validation: Loss 0.04940746 Accuracy 1.0
Epoch [8]: Loss 0.051136263
Epoch [8]: Loss 0.049436256
Epoch [8]: Loss 0.045960877
Epoch [8]: Loss 0.04622911
Epoch [8]: Loss 0.04370036
Epoch [8]: Loss 0.0411411
Epoch [8]: Loss 0.03735813
Validation: Loss 0.037494265 Accuracy 1.0
Validation: Loss 0.036779094 Accuracy 1.0
Epoch [9]: Loss 0.037471175
Epoch [9]: Loss 0.03629428
Epoch [9]: Loss 0.03668047
Epoch [9]: Loss 0.03522092
Epoch [9]: Loss 0.032479506
Epoch [9]: Loss 0.031569496
Epoch [9]: Loss 0.028921518
Validation: Loss 0.02851543 Accuracy 1.0
Validation: Loss 0.02780707 Accuracy 1.0
Epoch [10]: Loss 0.029460788
Epoch [10]: Loss 0.029934548
Epoch [10]: Loss 0.026222944
Epoch [10]: Loss 0.026292935
Epoch [10]: Loss 0.025317289
Epoch [10]: Loss 0.024227653
Epoch [10]: Loss 0.024810191
Validation: Loss 0.022380512 Accuracy 1.0
Validation: Loss 0.021726605 Accuracy 1.0
Epoch [11]: Loss 0.02254704
Epoch [11]: Loss 0.022422412
Epoch [11]: Loss 0.022176916
Epoch [11]: Loss 0.020948883
Epoch [11]: Loss 0.020587118
Epoch [11]: Loss 0.020619854
Epoch [11]: Loss 0.017800696
Validation: Loss 0.018210247 Accuracy 1.0
Validation: Loss 0.017629942 Accuracy 1.0
Epoch [12]: Loss 0.017999435
Epoch [12]: Loss 0.017956639
Epoch [12]: Loss 0.019060172
Epoch [12]: Loss 0.017782811
Epoch [12]: Loss 0.016320184
Epoch [12]: Loss 0.016671123
Epoch [12]: Loss 0.018172827
Validation: Loss 0.015304018 Accuracy 1.0
Validation: Loss 0.014795918 Accuracy 1.0
Epoch [13]: Loss 0.015731726
Epoch [13]: Loss 0.0156202
Epoch [13]: Loss 0.015160507
Epoch [13]: Loss 0.015198737
Epoch [13]: Loss 0.014902137
Epoch [13]: Loss 0.0141366385
Epoch [13]: Loss 0.012357281
Validation: Loss 0.013219781 Accuracy 1.0
Validation: Loss 0.012773553 Accuracy 1.0
Epoch [14]: Loss 0.014227949
Epoch [14]: Loss 0.012797753
Epoch [14]: Loss 0.013952888
Epoch [14]: Loss 0.012802935
Epoch [14]: Loss 0.012285331
Epoch [14]: Loss 0.012556584
Epoch [14]: Loss 0.012484008
Validation: Loss 0.011655683 Accuracy 1.0
Validation: Loss 0.011258184 Accuracy 1.0
Epoch [15]: Loss 0.011070745
Epoch [15]: Loss 0.012596181
Epoch [15]: Loss 0.01098256
Epoch [15]: Loss 0.011488103
Epoch [15]: Loss 0.011352739
Epoch [15]: Loss 0.011895499
Epoch [15]: Loss 0.012376968
Validation: Loss 0.010439353 Accuracy 1.0
Validation: Loss 0.010078218 Accuracy 1.0
Epoch [16]: Loss 0.010821207
Epoch [16]: Loss 0.010878562
Epoch [16]: Loss 0.009914115
Epoch [16]: Loss 0.010638686
Epoch [16]: Loss 0.010197597
Epoch [16]: Loss 0.010388163
Epoch [16]: Loss 0.009609264
Validation: Loss 0.009445695 Accuracy 1.0
Validation: Loss 0.00911682 Accuracy 1.0
Epoch [17]: Loss 0.009986697
Epoch [17]: Loss 0.009774229
Epoch [17]: Loss 0.009411738
Epoch [17]: Loss 0.00904651
Epoch [17]: Loss 0.009302911
Epoch [17]: Loss 0.009427908
Epoch [17]: Loss 0.009209704
Validation: Loss 0.008617703 Accuracy 1.0
Validation: Loss 0.008315929 Accuracy 1.0
Epoch [18]: Loss 0.009298562
Epoch [18]: Loss 0.009025093
Epoch [18]: Loss 0.008548114
Epoch [18]: Loss 0.008781625
Epoch [18]: Loss 0.0083300825
Epoch [18]: Loss 0.008510215
Epoch [18]: Loss 0.007108261
Validation: Loss 0.0079290345 Accuracy 1.0
Validation: Loss 0.007645623 Accuracy 1.0
Epoch [19]: Loss 0.008367594
Epoch [19]: Loss 0.0075551663
Epoch [19]: Loss 0.007793735
Epoch [19]: Loss 0.008383567
Epoch [19]: Loss 0.008038454
Epoch [19]: Loss 0.0078822635
Epoch [19]: Loss 0.007792745
Validation: Loss 0.0073176892 Accuracy 1.0
Validation: Loss 0.007054741 Accuracy 1.0
Epoch [20]: Loss 0.007963446
Epoch [20]: Loss 0.0075025326
Epoch [20]: Loss 0.0067413813
Epoch [20]: Loss 0.0075292406
Epoch [20]: Loss 0.007313558
Epoch [20]: Loss 0.0072101536
Epoch [20]: Loss 0.008100149
Validation: Loss 0.0067972783 Accuracy 1.0
Validation: Loss 0.0065486855 Accuracy 1.0
Epoch [21]: Loss 0.0070476537
Epoch [21]: Loss 0.007390582
Epoch [21]: Loss 0.006998795
Epoch [21]: Loss 0.0066453544
Epoch [21]: Loss 0.006526921
Epoch [21]: Loss 0.0065568946
Epoch [21]: Loss 0.0075205406
Validation: Loss 0.006330226 Accuracy 1.0
Validation: Loss 0.006097997 Accuracy 1.0
Epoch [22]: Loss 0.006612106
Epoch [22]: Loss 0.0063264486
Epoch [22]: Loss 0.006504682
Epoch [22]: Loss 0.0068865465
Epoch [22]: Loss 0.005823154
Epoch [22]: Loss 0.006511812
Epoch [22]: Loss 0.005964591
Validation: Loss 0.0059189876 Accuracy 1.0
Validation: Loss 0.0057008085 Accuracy 1.0
Epoch [23]: Loss 0.0056970725
Epoch [23]: Loss 0.0060758647
Epoch [23]: Loss 0.0059084436
Epoch [23]: Loss 0.006156966
Epoch [23]: Loss 0.005808864
Epoch [23]: Loss 0.0063720327
Epoch [23]: Loss 0.0062663043
Validation: Loss 0.005554602 Accuracy 1.0
Validation: Loss 0.0053483364 Accuracy 1.0
Epoch [24]: Loss 0.005743238
Epoch [24]: Loss 0.0057769986
Epoch [24]: Loss 0.0054934835
Epoch [24]: Loss 0.005660576
Epoch [24]: Loss 0.0055777933
Epoch [24]: Loss 0.0056111207
Epoch [24]: Loss 0.0058591706
Validation: Loss 0.0052244742 Accuracy 1.0
Validation: Loss 0.0050300434 Accuracy 1.0
Epoch [25]: Loss 0.0053289277
Epoch [25]: Loss 0.005583775
Epoch [25]: Loss 0.005324645
Epoch [25]: Loss 0.005377317
Epoch [25]: Loss 0.005134449
Epoch [25]: Loss 0.005279418
Epoch [25]: Loss 0.004982455
Validation: Loss 0.0049290666 Accuracy 1.0
Validation: Loss 0.0047445595 Accuracy 1.0

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
Julia Version 1.10.1
Commit 7790d6f0641 (2024-02-13 20:41 UTC)
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
  JULIA_PROJECT = /var/lib/buildkite-agent/builds/gpuci-12/julialang/lux-dot-jl/docs/Project.toml
  JULIA_AMDGPU_LOGGING_ENABLED = true
  JULIA_DEBUG = Literate
  JULIA_CPU_THREADS = 2
  JULIA_NUM_THREADS = 48
  JULIA_LOAD_PATH = @:@v#.#:@stdlib
  JULIA_CUDA_HARD_MEMORY_LIMIT = 25%

CUDA runtime 12.3, artifact installation
CUDA driver 12.3
NVIDIA driver 545.23.8

CUDA libraries: 
- CUBLAS: 12.3.4
- CURAND: 10.3.4
- CUFFT: 11.0.12
- CUSOLVER: 11.5.4
- CUSPARSE: 12.2.0
- CUPTI: 21.0.0
- NVML: 12.0.0+545.23.8

Julia packages: 
- CUDA: 5.2.0
- CUDA_Driver_jll: 0.7.0+1
- CUDA_Runtime_jll: 0.11.1+0

Toolchain:
- Julia: 1.10.1
- LLVM: 15.0.7

Environment:
- JULIA_CUDA_HARD_MEMORY_LIMIT: 25%

1 device:
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.130 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
