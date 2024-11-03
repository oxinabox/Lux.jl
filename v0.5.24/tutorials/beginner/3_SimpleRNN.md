


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
Epoch [1]: Loss 0.56333935
Epoch [1]: Loss 0.5136839
Epoch [1]: Loss 0.4778219
Epoch [1]: Loss 0.45830724
Epoch [1]: Loss 0.4309851
Epoch [1]: Loss 0.40499997
Epoch [1]: Loss 0.36390668
Validation: Loss 0.36606097 Accuracy 1.0
Validation: Loss 0.368439 Accuracy 1.0
Epoch [2]: Loss 0.37005138
Epoch [2]: Loss 0.3467431
Epoch [2]: Loss 0.33976758
Epoch [2]: Loss 0.31967992
Epoch [2]: Loss 0.30813438
Epoch [2]: Loss 0.29036492
Epoch [2]: Loss 0.2746907
Validation: Loss 0.2575819 Accuracy 1.0
Validation: Loss 0.25924146 Accuracy 1.0
Epoch [3]: Loss 0.25724977
Epoch [3]: Loss 0.2502552
Epoch [3]: Loss 0.23750934
Epoch [3]: Loss 0.22274128
Epoch [3]: Loss 0.21064095
Epoch [3]: Loss 0.20329139
Epoch [3]: Loss 0.19052799
Validation: Loss 0.18038641 Accuracy 1.0
Validation: Loss 0.18131073 Accuracy 1.0
Epoch [4]: Loss 0.18760301
Epoch [4]: Loss 0.1745981
Epoch [4]: Loss 0.16298734
Epoch [4]: Loss 0.15461794
Epoch [4]: Loss 0.15117684
Epoch [4]: Loss 0.14165995
Epoch [4]: Loss 0.13455904
Validation: Loss 0.1290251 Accuracy 1.0
Validation: Loss 0.1295974 Accuracy 1.0
Epoch [5]: Loss 0.12913242
Epoch [5]: Loss 0.12518889
Epoch [5]: Loss 0.121304765
Epoch [5]: Loss 0.113020055
Epoch [5]: Loss 0.10670753
Epoch [5]: Loss 0.10373017
Epoch [5]: Loss 0.09715074
Validation: Loss 0.09382136 Accuracy 1.0
Validation: Loss 0.09433276 Accuracy 1.0
Epoch [6]: Loss 0.09555197
Epoch [6]: Loss 0.090471946
Epoch [6]: Loss 0.086698346
Epoch [6]: Loss 0.08301762
Epoch [6]: Loss 0.08072689
Epoch [6]: Loss 0.0750684
Epoch [6]: Loss 0.072590016
Validation: Loss 0.069077104 Accuracy 1.0
Validation: Loss 0.06958441 Accuracy 1.0
Epoch [7]: Loss 0.06894362
Epoch [7]: Loss 0.06624253
Epoch [7]: Loss 0.06635922
Epoch [7]: Loss 0.062460832
Epoch [7]: Loss 0.058346078
Epoch [7]: Loss 0.056354444
Epoch [7]: Loss 0.05765489
Validation: Loss 0.051422313 Accuracy 1.0
Validation: Loss 0.051908273 Accuracy 1.0
Epoch [8]: Loss 0.050914146
Epoch [8]: Loss 0.050719973
Epoch [8]: Loss 0.048139848
Epoch [8]: Loss 0.04613207
Epoch [8]: Loss 0.044803705
Epoch [8]: Loss 0.043390658
Epoch [8]: Loss 0.041778546
Validation: Loss 0.03860087 Accuracy 1.0
Validation: Loss 0.039036997 Accuracy 1.0
Epoch [9]: Loss 0.041106284
Epoch [9]: Loss 0.03702393
Epoch [9]: Loss 0.03719309
Epoch [9]: Loss 0.034715157
Epoch [9]: Loss 0.033190012
Epoch [9]: Loss 0.03131763
Epoch [9]: Loss 0.03320401
Validation: Loss 0.02937739 Accuracy 1.0
Validation: Loss 0.029774431 Accuracy 1.0
Epoch [10]: Loss 0.0296462
Epoch [10]: Loss 0.02858942
Epoch [10]: Loss 0.027629316
Epoch [10]: Loss 0.027673712
Epoch [10]: Loss 0.026285566
Epoch [10]: Loss 0.02558849
Epoch [10]: Loss 0.024804357
Validation: Loss 0.023005672 Accuracy 1.0
Validation: Loss 0.023343652 Accuracy 1.0
Epoch [11]: Loss 0.023494221
Epoch [11]: Loss 0.02309254
Epoch [11]: Loss 0.022674322
Epoch [11]: Loss 0.021683626
Epoch [11]: Loss 0.020551182
Epoch [11]: Loss 0.020381412
Epoch [11]: Loss 0.018609371
Validation: Loss 0.018658526 Accuracy 1.0
Validation: Loss 0.018959533 Accuracy 1.0
Epoch [12]: Loss 0.019367732
Epoch [12]: Loss 0.019244622
Epoch [12]: Loss 0.017973874
Epoch [12]: Loss 0.01695874
Epoch [12]: Loss 0.017647557
Epoch [12]: Loss 0.01705606
Epoch [12]: Loss 0.016381774
Validation: Loss 0.015656147 Accuracy 1.0
Validation: Loss 0.015923424 Accuracy 1.0
Epoch [13]: Loss 0.016702525
Epoch [13]: Loss 0.015677698
Epoch [13]: Loss 0.016325656
Epoch [13]: Loss 0.014608845
Epoch [13]: Loss 0.013643465
Epoch [13]: Loss 0.01482619
Epoch [13]: Loss 0.01455888
Validation: Loss 0.013506062 Accuracy 1.0
Validation: Loss 0.01373807 Accuracy 1.0
Epoch [14]: Loss 0.0142694535
Epoch [14]: Loss 0.013880728
Epoch [14]: Loss 0.012789983
Epoch [14]: Loss 0.014317127
Epoch [14]: Loss 0.012393361
Epoch [14]: Loss 0.012549864
Epoch [14]: Loss 0.011383554
Validation: Loss 0.01189483 Accuracy 1.0
Validation: Loss 0.012105742 Accuracy 1.0
Epoch [15]: Loss 0.012276387
Epoch [15]: Loss 0.01195986
Epoch [15]: Loss 0.011929016
Epoch [15]: Loss 0.012296114
Epoch [15]: Loss 0.010780387
Epoch [15]: Loss 0.011504412
Epoch [15]: Loss 0.0114772795
Validation: Loss 0.010640824 Accuracy 1.0
Validation: Loss 0.0108352825 Accuracy 1.0
Epoch [16]: Loss 0.010545417
Epoch [16]: Loss 0.010220246
Epoch [16]: Loss 0.01069602
Epoch [16]: Loss 0.011555088
Epoch [16]: Loss 0.010165058
Epoch [16]: Loss 0.010476818
Epoch [16]: Loss 0.010005552
Validation: Loss 0.009628498 Accuracy 1.0
Validation: Loss 0.009802437 Accuracy 1.0
Epoch [17]: Loss 0.009971147
Epoch [17]: Loss 0.009754109
Epoch [17]: Loss 0.009972479
Epoch [17]: Loss 0.009865621
Epoch [17]: Loss 0.009466064
Epoch [17]: Loss 0.009146927
Epoch [17]: Loss 0.0077891047
Validation: Loss 0.00877974 Accuracy 1.0
Validation: Loss 0.008947973 Accuracy 1.0
Epoch [18]: Loss 0.008439337
Epoch [18]: Loss 0.009058863
Epoch [18]: Loss 0.008619916
Epoch [18]: Loss 0.0094860345
Epoch [18]: Loss 0.008655991
Epoch [18]: Loss 0.00876843
Epoch [18]: Loss 0.007869645
Validation: Loss 0.008072358 Accuracy 1.0
Validation: Loss 0.008220952 Accuracy 1.0
Epoch [19]: Loss 0.0086829765
Epoch [19]: Loss 0.008809444
Epoch [19]: Loss 0.008353992
Epoch [19]: Loss 0.007873991
Epoch [19]: Loss 0.0073811524
Epoch [19]: Loss 0.0076702246
Epoch [19]: Loss 0.0077477572
Validation: Loss 0.007452484 Accuracy 1.0
Validation: Loss 0.0075995005 Accuracy 1.0
Epoch [20]: Loss 0.007853085
Epoch [20]: Loss 0.007752986
Epoch [20]: Loss 0.00800846
Epoch [20]: Loss 0.0072520506
Epoch [20]: Loss 0.007305531
Epoch [20]: Loss 0.007235137
Epoch [20]: Loss 0.0061125886
Validation: Loss 0.006920118 Accuracy 1.0
Validation: Loss 0.0070508453 Accuracy 1.0
Epoch [21]: Loss 0.006982115
Epoch [21]: Loss 0.00677629
Epoch [21]: Loss 0.007486506
Epoch [21]: Loss 0.0073820045
Epoch [21]: Loss 0.0064743524
Epoch [21]: Loss 0.0070812544
Epoch [21]: Loss 0.005842489
Validation: Loss 0.006449352 Accuracy 1.0
Validation: Loss 0.0065785176 Accuracy 1.0
Epoch [22]: Loss 0.006389257
Epoch [22]: Loss 0.0065440927
Epoch [22]: Loss 0.006469347
Epoch [22]: Loss 0.006757633
Epoch [22]: Loss 0.0067930184
Epoch [22]: Loss 0.006383546
Epoch [22]: Loss 0.0056843366
Validation: Loss 0.006037912 Accuracy 1.0
Validation: Loss 0.0061543947 Accuracy 1.0
Epoch [23]: Loss 0.0064054085
Epoch [23]: Loss 0.006058749
Epoch [23]: Loss 0.0060576806
Epoch [23]: Loss 0.006118314
Epoch [23]: Loss 0.0058030332
Epoch [23]: Loss 0.006187264
Epoch [23]: Loss 0.00628213
Validation: Loss 0.0056646117 Accuracy 1.0
Validation: Loss 0.0057800775 Accuracy 1.0
Epoch [24]: Loss 0.0059870305
Epoch [24]: Loss 0.0058788885
Epoch [24]: Loss 0.0056843027
Epoch [24]: Loss 0.0059574284
Epoch [24]: Loss 0.005281595
Epoch [24]: Loss 0.005894967
Epoch [24]: Loss 0.004868773
Validation: Loss 0.0053309677 Accuracy 1.0
Validation: Loss 0.0054356977 Accuracy 1.0
Epoch [25]: Loss 0.0057878373
Epoch [25]: Loss 0.005832994
Epoch [25]: Loss 0.0050152745
Epoch [25]: Loss 0.005160064
Epoch [25]: Loss 0.005615782
Epoch [25]: Loss 0.005101732
Epoch [25]: Loss 0.005240244
Validation: Loss 0.00502773 Accuracy 1.0
Validation: Loss 0.0051298738 Accuracy 1.0

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
  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.297 GiB / 4.750 GiB available)
┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.
└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19

```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
