import{_ as i,c as a,a2 as n,o as t}from"./chunks/framework.CCjWn1F9.js";const g=JSON.parse('{"title":"Compiling Lux Models using Reactant.jl","description":"","frontmatter":{},"headers":[],"relativePath":"manual/compiling_lux_models.md","filePath":"manual/compiling_lux_models.md","lastUpdated":null}'),e={name:"manual/compiling_lux_models.md"};function l(p,s,h,k,d,r){return t(),a("div",null,s[0]||(s[0]=[n(`<h1 id="reactant-compilation" tabindex="-1">Compiling Lux Models using <code>Reactant.jl</code> <a class="header-anchor" href="#reactant-compilation" aria-label="Permalink to &quot;Compiling Lux Models using \`Reactant.jl\` {#reactant-compilation}&quot;">​</a></h1><p>Quoting the Reactant.jl Readme:</p><blockquote><p>Reactant takes Julia function and compile it into MLIR and run fancy optimizations on top of it, including using EnzymeMLIR for automatic differentiation, and create relevant executables for CPU/GPU/TPU via XLA. It presently operates as a tracing system. Compiled functions will assume the same control flow pattern as was original taken by objects used at compile time, and control flow (e.g. if, for) as well as any type instabilities will be removed. The benefits of this approach is immediately making all such code available for advanced optimization with little developer effort.</p></blockquote><div class="danger custom-block"><p class="custom-block-title">Experimental</p><p>Reactant compilation is a very new feature and is currently experimental. Certain models might not be compilable yet, but we are actively working on it. Open an issue if you encounter any problems.</p></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Reactant, Enzyme, Random, Zygote</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Functors, Optimisers, Printf</span></span></code></pre></div><div class="tip custom-block"><p class="custom-block-title">Using the <code>TrainState</code> API</p><p>If you are using the <a href="/dev/api/Lux/utilities#Lux.Training.TrainState"><code>Training.TrainState</code></a> API, skip to the <a href="/dev/manual/compiling_lux_models#compile_lux_model_trainstate">bottom of this page</a> to see how to train the model without any of this boilerplate.</p></div><p>We start by defining a simple MLP model:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>((layer_1 = (weight = Float32[-1.2228831 -0.87702435; 0.5031421 -0.15133555; … ; -0.31550723 -0.7672513; 0.111552626 0.6064619], bias = Float32[-0.63795453, 0.62450767, -0.014877922, 0.25385493, -0.20188306, 0.21950458, 0.109203495, 0.23021114, -0.26657984, 0.16187939  …  -0.6409691, 0.4391564, 0.14488737, 0.49998975, -0.04566476, -0.56069607, -0.33442986, -0.1549292, -0.42669478, 0.636308]), layer_2 = (weight = Float32[0.293211 0.19084926 … 0.2464001 0.2913357; -0.116796836 0.09926938 … -0.26311737 -0.15802455; … ; -0.2042089 -0.22406094 … 0.13504265 0.09289699; 0.25389904 0.28355134 … 0.28725442 0.13343152], bias = Float32[0.12992674, 0.14568081, -0.10754459, -0.15686738, -0.14118214, 0.088205874, -0.06301335, 0.06027697, 0.14445141, 0.08791955  …  0.053627778, -0.06618893, 0.1124609, 0.037500158, 0.12827216, -0.13913931, -0.17048413, -0.1032465, -0.15493166, -0.0069942693]), layer_3 = (weight = Float32[-0.031503614 -0.23162955 … 0.097182155 -0.099906564; 0.05729505 0.28042415 … 0.1293236 -0.18089005], bias = Float32[-0.16409892, 0.042256515])), (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))</span></span></code></pre></div><p>We then create a random input and output data:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.^</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2×32 Matrix{Float32}:</span></span>
<span class="line"><span> 0.203036   0.362593  0.354464   0.0320963  …  0.0954186  0.713316  0.438519</span></span>
<span class="line"><span> 0.0155126  1.13864   0.0187668  0.142251      2.24169    4.16407   0.415858</span></span></code></pre></div><p>We will use <a href="/dev/api/Accelerator_Support/MLDataDevices#MLDataDevices.xla_device"><code>xla_device</code></a> similar to <a href="/dev/api/Accelerator_Support/MLDataDevices#MLDataDevices.gpu_device"><code>gpu_device</code></a> to move the arrays to <code>Reactant</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xla_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span></span></code></pre></div><p>First let&#39;s run the model as we would normally:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">pred_lux, _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(Float32[-0.20053944 -0.8147778 … -2.3903124 -0.15544322; 0.1585735 0.4981351 … 1.2586653 0.27545732], (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple()))</span></span></code></pre></div><p>To run it using <code>XLA</code> we need to compile the model. We can do this using the <code>Reactant.@compile</code> macro. Note that the inputs need to be moved to the device using <a href="/dev/api/Accelerator_Support/MLDataDevices#MLDataDevices.xla_device"><code>xla_device</code></a> first.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, ps_ra, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st_ra))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Reactant.Compiler.Thunk{Symbol(&quot;##Chain{@NamedTuple{layer_1::Dense{typeof(gelu), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(gelu), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}((layer_1 = Dense(2 =&gt; 32, gelu), layer_2 = Dense(32 =&gt; 32, gelu), layer_3 = Dense(32 =&gt; 2)), nothing)_reactant#1069&quot;)}()</span></span></code></pre></div><p>Now we can test the difference between the results:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">pred_compiled, _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, ps_ra, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st_ra))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">pred_lux </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(pred_compiled)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2×32 Matrix{Float32}:</span></span>
<span class="line"><span> 0.0  -1.19209f-7  -2.98023f-8  2.98023f-8  …  2.98023f-8  0.0  -1.49012f-8</span></span>
<span class="line"><span> 0.0   1.19209f-7   8.9407f-8   1.78814f-7     1.49012f-8  0.0   2.98023f-8</span></span></code></pre></div><p>The difference is very small as we would expect. Now, let&#39;s try to differentiate the output of the model. We need to use <code>Enzyme.jl</code> to do this.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pred, _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> MSELoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()(pred, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loss_function (generic function with 1 method)</span></span></code></pre></div><p>We will use <code>Zygote.jl</code> to compute the gradient of the loss function for the vanilla model.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, x, y)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">∂ps_zyg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> only</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Zygote</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, x, y), ps))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(layer_1 = (weight = Float32[-0.011611392 -0.12556516; -0.09724939 0.11515345; … ; 0.08667634 -0.2689521; -0.09643307 0.030881835], bias = Float32[0.048133414, -0.106884085, 0.097701035, 0.105524555, -0.039647065, -0.018338889, -0.019115759, -0.15107606, 0.013992601, -0.014150472  …  0.0041674753, 0.032615878, 0.031403527, 0.13760866, -0.04225484, 0.049417753, -0.00059220614, -0.03242131, 0.18807876, -0.07640441]), layer_2 = (weight = Float32[-0.004287243 0.028275706 … -0.0073489705 0.0028297475; 0.016479947 0.030926052 … -0.0036810301 0.019791333; … ; 0.010637202 -0.002057937 … 0.010218928 -0.047897488; 0.13518015 0.25378025 … 0.0903271 0.048811335], bias = Float32[0.018884761, 0.053747915, -0.17435724, -0.059518166, -0.10950818, 0.13725635, -0.048533253, -0.11365668, -0.3891182, 0.26477236  …  0.2236399, 0.1377298, -0.027226413, -0.09919551, -0.12902719, 0.0072498624, -0.012183794, 0.066751055, -0.017432783, 0.26700422]), layer_3 = (weight = Float32[-2.5994074 0.07425845 … 0.08953094 -0.9130077; -1.1187928 0.0062888456 … -0.032405674 -0.4112945], bias = Float32[-1.6541586, -0.61384505]))</span></span></code></pre></div><p>Now we will compile the gradient function using <code>Reactant.@compile</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> enzyme_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, x, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Enzyme</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Enzyme</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reverse, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(loss_function), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ps, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">enzyme_gradient_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> enzyme_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps_ra, st_ra, x_ra, y_ra)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">∂ps_enzyme </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> enzyme_gradient_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps_ra, st_ra, x_ra, y_ra)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(layer_1 = (weight = Float32[-0.011611394 -0.12556516; -0.09724942 0.11515346; … ; 0.08667634 -0.2689521; -0.09643307 0.030881835], bias = Float32[0.048133414, -0.10688411, 0.097701035, 0.10552457, -0.039647065, -0.018338893, -0.01911576, -0.15107603, 0.013992591, -0.01415047  …  0.0041674743, 0.03261588, 0.031403534, 0.13760868, -0.042254835, 0.049417756, -0.0005922087, -0.0324213, 0.18807876, -0.076404385]), layer_2 = (weight = Float32[-0.004287243 0.02827571 … -0.0073489705 0.0028297466; 0.016479947 0.030926049 … -0.0036810287 0.019791335; … ; 0.010637204 -0.0020579377 … 0.0102189295 -0.047897488; 0.13518013 0.25378025 … 0.0903271 0.048811335], bias = Float32[0.018884756, 0.053747915, -0.17435724, -0.059518166, -0.1095082, 0.13725637, -0.04853325, -0.11365668, -0.38911813, 0.26477236  …  0.22363997, 0.1377298, -0.027226416, -0.0991955, -0.12902719, 0.007249862, -0.012183795, 0.06675106, -0.017432783, 0.26700416]), layer_3 = (weight = Float32[-2.5994072 0.07425846 … 0.089530945 -0.9130077; -1.1187928 0.006288857 … -0.032405667 -0.4112944], bias = Float32[-1.6541584, -0.6138451]))</span></span></code></pre></div><p>Now we check the difference:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fmap</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Broadcast</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">BroadcastFunction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), ∂ps_zyg, ∂ps_enzyme)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(layer_1 = (weight = Float32[1.8626451f-9 0.0; 2.9802322f-8 -1.4901161f-8; … ; 0.0 0.0; 0.0 0.0], bias = Float32[0.0, 2.2351742f-8, 0.0, -1.4901161f-8, 0.0, 3.7252903f-9, 1.8626451f-9, -2.9802322f-8, 1.0244548f-8, -2.7939677f-9  …  9.313226f-10, -3.7252903f-9, -7.450581f-9, -1.4901161f-8, -3.7252903f-9, -3.7252903f-9, 2.561137f-9, -1.1175871f-8, 0.0, -2.2351742f-8]), layer_2 = (weight = Float32[0.0 -3.7252903f-9 … 0.0 9.313226f-10; 0.0 3.7252903f-9 … -1.3969839f-9 -1.8626451f-9; … ; -1.8626451f-9 6.9849193f-10 … -1.8626451f-9 0.0; 1.4901161f-8 0.0 … 0.0 0.0], bias = Float32[5.5879354f-9, 0.0, 0.0, 0.0, 2.2351742f-8, -1.4901161f-8, -3.7252903f-9, 0.0, -5.9604645f-8, 0.0  …  -5.9604645f-8, 0.0, 3.7252903f-9, -7.450581f-9, 0.0, 4.656613f-10, 9.313226f-10, -7.450581f-9, 0.0, 5.9604645f-8]), layer_3 = (weight = Float32[-2.3841858f-7 -1.4901161f-8 … -7.450581f-9 0.0; 0.0 -1.1641532f-8 … -7.450581f-9 -8.940697f-8], bias = Float32[-2.3841858f-7, 5.9604645f-8]))</span></span></code></pre></div><h2 id="compile_lux_model_trainstate" tabindex="-1">Using the <code>TrainState</code> API <a class="header-anchor" href="#compile_lux_model_trainstate" aria-label="Permalink to &quot;Using the \`TrainState\` API {#compile_lux_model_trainstate}&quot;">​</a></h2><p>Now that we saw the low-level API let&#39;s see how to train the model without any of this boilerplate. Simply follow the following steps:</p><ol><li><p>Create a device using <code>xla_device</code>. Remember to load <code>Reactant.jl</code> before doing this.</p></li><li><p>Similar to other device functions move the model, parameters, states and data to the device. Note that you might want to use <a href="/dev/api/Accelerator_Support/MLDataDevices#MLDataDevices.DeviceIterator"><code>DeviceIterator</code></a> to move the data loader to the device with an iterator.</p></li><li><p>Construct a <code>TrainState</code> using <a href="/dev/api/Lux/utilities#Lux.Training.TrainState"><code>Training.TrainState</code></a>.</p></li><li><p>And most importantly use <code>AutoEnzyme</code> while calling <a href="/dev/api/Lux/utilities#Lux.Training.single_train_step!"><code>Training.single_train_step!</code></a> or <a href="/dev/api/Lux/utilities#Lux.Training.single_train_step"><code>Training.single_train_step</code></a>.</p></li></ol><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Chain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [xᵢ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.^</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xᵢ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_ra]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_ra </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> DeviceIterator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(xdev, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_ra, y_ra))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, dataloader)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> iteration </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (xᵢ, yᵢ) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            grads, loss, stats, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MSELoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), (xᵢ, yᵢ), train_state)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> iteration </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ||</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> iteration </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # We need to do this since scalar outputs are currently expressed as a zero-dim</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # array</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(loss)[]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Iter: [%4d/%4d]</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss: %.8f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, iteration, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, loss)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_state</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">train_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps_ra, st_ra, dataloader)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Iter: [   1/1000]	Loss: 3.07964921</span></span>
<span class="line"><span>Iter: [ 100/1000]	Loss: 1.06519687</span></span>
<span class="line"><span>Iter: [ 200/1000]	Loss: 0.44807646</span></span>
<span class="line"><span>Iter: [ 300/1000]	Loss: 0.24150778</span></span>
<span class="line"><span>Iter: [ 400/1000]	Loss: 0.14340512</span></span>
<span class="line"><span>Iter: [ 500/1000]	Loss: 0.09299411</span></span>
<span class="line"><span>Iter: [ 600/1000]	Loss: 0.06612328</span></span>
<span class="line"><span>Iter: [ 700/1000]	Loss: 0.04551310</span></span>
<span class="line"><span>Iter: [ 800/1000]	Loss: 0.03070261</span></span>
<span class="line"><span>Iter: [ 900/1000]	Loss: 0.02143306</span></span>
<span class="line"><span>Iter: [1000/1000]	Loss: 0.01542492</span></span></code></pre></div>`,40)]))}const E=i(e,[["render",l]]);export{g as __pageData,E as default};
