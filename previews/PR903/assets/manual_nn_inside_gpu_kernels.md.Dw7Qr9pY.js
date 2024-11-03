import{_ as a,c as i,a2 as n,o as p}from"./chunks/framework.DrUBLjQW.js";const E=JSON.parse('{"title":"Neural Networks Inside GPU Kernels","description":"","frontmatter":{},"headers":[],"relativePath":"manual/nn_inside_gpu_kernels.md","filePath":"manual/nn_inside_gpu_kernels.md","lastUpdated":null}'),l={name:"manual/nn_inside_gpu_kernels.md"};function e(t,s,h,k,r,d){return p(),i("div",null,s[0]||(s[0]=[n(`<h1 id="Neural-Networks-Inside-GPU-Kernels" tabindex="-1">Neural Networks Inside GPU Kernels <a class="header-anchor" href="#Neural-Networks-Inside-GPU-Kernels" aria-label="Permalink to &quot;Neural Networks Inside GPU Kernels {#Neural-Networks-Inside-GPU-Kernels}&quot;">​</a></h1><p>In this page, we will describe how to embed neural networks inside GPU kernels. We will use <a href="https://github.com/JuliaGPU/KernelAbstractions.jl" target="_blank" rel="noreferrer">KernelAbstractions.jl</a> to do this, making it compatible with multiple GPU backends.</p><div class="warning custom-block"><p class="custom-block-title">Experimental Feature</p><p>This is a relatively new and experimental feature. Expect edge cases and open issues on GitHub if you find any.</p></div><div class="tip custom-block"><p class="custom-block-title">Inference Only</p><p>Currently this works only for inference. We will eventually test automatic differentiation using Enzyme.jl</p></div><div class="danger custom-block"><p class="custom-block-title">Batching</p><p>In most usecases, this form of batching via embedding the neural network inside a GPU kernel is not recommended and will lead to suboptimal performance. Instead, batch the input data and let Lux handle the batching internally.</p></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, LuxCUDA, Random</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> KernelAbstractions, StaticArrays</span></span></code></pre></div><p>First thing to remember is that we can&#39;t use regular high-level operations inside the kernels, instead we will use Static Arrays. Leveraging Julia&#39;s multiple dispatch Lux will use specialized operations that are compatible with GPU kernels.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@kernel</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> nn_eval_single_batch!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output, model, input, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @index</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Global, Linear)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, input[i], ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    output[i] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>nn_eval_single_batch! (generic function with 4 methods)</span></span></code></pre></div><p>We define and initialize the neural network as usual, but we need to additionally convert the Arrays into SArrays.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, relu), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">123</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), nn)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">to_sarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SArray{Tuple{size(x)...}}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_static </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">recursive_map</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(to_sarray, ps)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_static </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">recursive_map</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(to_sarray, st)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(layer_1 = NamedTuple(), layer_2 = NamedTuple())</span></span></code></pre></div><p>First we will run it on CPU.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Currently due to a minor bug, we cannot call the Lux models with vector input. As a workaround we make them into Matrix with batch size 1.</p></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">input </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@SArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float64, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@SArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zeros</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float64, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Allocate the output</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1024-element Vector{StaticArraysCore.SMatrix{4, 1, Float64, 4}}:</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> ⋮</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span></code></pre></div><p>Now run the model using KernelAbstractions.jl</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">backend </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> KernelAbstractions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">cpu_kernel! </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nn_eval_single_batch!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cpu_kernel!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output, nn, input, ps_static, st_static; ndrange</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">KernelAbstractions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">synchronize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1024-element Vector{StaticArraysCore.SMatrix{4, 1, Float64, 4}}:</span></span>
<span class="line"><span> [2.0564903986057956; 1.1188200246206075; -1.2227837233928576; -0.8173783982243132;;]</span></span>
<span class="line"><span> [1.9721554734769875; 1.3940224213371761; -1.297959481822617; -0.7195462169922175;;]</span></span>
<span class="line"><span> [2.5680085614623662; 1.713567516238075; -1.7165512278088038; -1.009963844931984;;]</span></span>
<span class="line"><span> [1.800792614736468; 0.36222499022985155; -1.1204217935313214; -1.1836515766351254;;]</span></span>
<span class="line"><span> [1.486550215883336; 0.32839986131789933; -0.9019142280758281; -0.9452923791531558;;]</span></span>
<span class="line"><span> [2.716134755899883; 1.1617228180412864; -1.902982902377702; -1.5865265807660498;;]</span></span>
<span class="line"><span> [1.0228109822209213; 0.2525357728685884; -0.4376572711003852; -0.4500963619011972;;]</span></span>
<span class="line"><span> [2.2771862617010155; 0.5381101016248151; -1.4730743722547668; -1.488028235902512;;]</span></span>
<span class="line"><span> [3.2791573282471997; 1.3436353225087703; -2.4619778701480337; -2.1239749674027375;;]</span></span>
<span class="line"><span> [1.2290224145974982; 0.4158693023143286; -0.6370531107315014; -0.5779067839062536;;]</span></span>
<span class="line"><span> ⋮</span></span>
<span class="line"><span> [1.8674763752817416; 1.6423511984038721; -1.1477053709248992; -0.3834447782571344;;]</span></span>
<span class="line"><span> [2.091359335844565; 1.0621559246995447; -1.4763277207638008; -1.142470881033475;;]</span></span>
<span class="line"><span> [2.712979078066394; 0.42005835019799886; -1.717863343114228; -1.8601870861800127;;]</span></span>
<span class="line"><span> [0.7701346738750905; 0.2869913410456831; -0.1586047939092094; -0.10140238162746013;;]</span></span>
<span class="line"><span> [1.611584190904272; 1.2797048270773437; -0.923950547913545; -0.3558193508137715;;]</span></span>
<span class="line"><span> [2.0884834705765853; 0.862480861009647; -1.3942307655311696; -1.179584495291061;;]</span></span>
<span class="line"><span> [2.390200114697191; 0.5267549745189349; -1.657670184695808; -1.7089496198123055;;]</span></span>
<span class="line"><span> [2.1846486482317626; -0.031414255389526885; -1.3279041356366077; -1.6909446526419574;;]</span></span>
<span class="line"><span> [1.3650193059617517; 0.5210742834996898; -0.7689272356710357; -0.6642563709240284;;]</span></span></code></pre></div><p>Now we will run the same model on GPU.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">input_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> input </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@SArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zeros</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float64, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1024-element CuArray{StaticArraysCore.SMatrix{4, 1, Float64, 4}, 1, CUDA.DeviceMemory}:</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> ⋮</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span>
<span class="line"><span> [0.0; 0.0; 0.0; 0.0;;]</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">backend </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> KernelAbstractions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output_gpu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gpu_kernel! </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nn_eval_single_batch!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_kernel!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output_gpu, nn, input_gpu, ps_static, st_static; ndrange</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(output_gpu))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">KernelAbstractions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">synchronize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output_gpu</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1024-element CuArray{StaticArraysCore.SMatrix{4, 1, Float64, 4}, 1, CUDA.DeviceMemory}:</span></span>
<span class="line"><span> [2.0564903986057956; 1.1188200246206075; -1.2227837233928576; -0.8173783982243132;;]</span></span>
<span class="line"><span> [1.9721554734769875; 1.3940224213371761; -1.297959481822617; -0.7195462169922173;;]</span></span>
<span class="line"><span> [2.5680085614623662; 1.713567516238075; -1.7165512278088038; -1.009963844931984;;]</span></span>
<span class="line"><span> [1.800792614736468; 0.36222499022985155; -1.1204217935313214; -1.1836515766351254;;]</span></span>
<span class="line"><span> [1.486550215883336; 0.32839986131789933; -0.9019142280758281; -0.9452923791531558;;]</span></span>
<span class="line"><span> [2.716134755899883; 1.1617228180412864; -1.902982902377702; -1.5865265807660498;;]</span></span>
<span class="line"><span> [1.0228109822209213; 0.2525357728685884; -0.4376572711003852; -0.4500963619011972;;]</span></span>
<span class="line"><span> [2.2771862617010155; 0.5381101016248151; -1.4730743722547668; -1.488028235902512;;]</span></span>
<span class="line"><span> [3.2791573282471997; 1.3436353225087703; -2.4619778701480337; -2.1239749674027375;;]</span></span>
<span class="line"><span> [1.2290224145974982; 0.4158693023143286; -0.6370531107315014; -0.5779067839062536;;]</span></span>
<span class="line"><span> ⋮</span></span>
<span class="line"><span> [1.8674763752817414; 1.6423511984038721; -1.147705370924899; -0.3834447782571341;;]</span></span>
<span class="line"><span> [2.0913593358445652; 1.062155924699545; -1.4763277207638013; -1.142470881033475;;]</span></span>
<span class="line"><span> [2.712979078066394; 0.420058350197999; -1.717863343114228; -1.8601870861800127;;]</span></span>
<span class="line"><span> [0.7701346738750905; 0.2869913410456831; -0.1586047939092094; -0.10140238162746013;;]</span></span>
<span class="line"><span> [1.611584190904272; 1.2797048270773437; -0.923950547913545; -0.3558193508137715;;]</span></span>
<span class="line"><span> [2.0884834705765853; 0.862480861009647; -1.3942307655311696; -1.179584495291061;;]</span></span>
<span class="line"><span> [2.390200114697191; 0.5267549745189349; -1.657670184695808; -1.7089496198123055;;]</span></span>
<span class="line"><span> [2.1846486482317626; -0.031414255389526885; -1.3279041356366077; -1.6909446526419574;;]</span></span>
<span class="line"><span> [1.3650193059617517; 0.5210742834996898; -0.7689272356710357; -0.6642563709240284;;]</span></span></code></pre></div>`,24)]))}const g=a(l,[["render",e]]);export{E as __pageData,g as default};
