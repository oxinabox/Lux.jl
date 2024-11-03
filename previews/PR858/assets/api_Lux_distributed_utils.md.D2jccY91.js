import{_ as i,c as e,o as s,a4 as t}from"./chunks/framework.D-WlR71w.js";const k=JSON.parse('{"title":"Distributed Utils","description":"","frontmatter":{},"headers":[],"relativePath":"api/Lux/distributed_utils.md","filePath":"api/Lux/distributed_utils.md","lastUpdated":null}'),a={name:"api/Lux/distributed_utils.md"},d=t(`<h1 id="Distributed-Utils" tabindex="-1">Distributed Utils <a class="header-anchor" href="#Distributed-Utils" aria-label="Permalink to &quot;Distributed Utils {#Distributed-Utils}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">Note</p><p>These functionalities are available via the <code>Lux.DistributedUtils</code> module.</p></div><h2 id="index" tabindex="-1">Index <a class="header-anchor" href="#index" aria-label="Permalink to &quot;Index&quot;">​</a></h2><ul><li><a href="#Lux.DistributedUtils.DistributedDataContainer"><code>Lux.DistributedUtils.DistributedDataContainer</code></a></li><li><a href="#Lux.DistributedUtils.DistributedOptimizer"><code>Lux.DistributedUtils.DistributedOptimizer</code></a></li><li><a href="#Lux.MPIBackend"><code>Lux.MPIBackend</code></a></li><li><a href="#Lux.NCCLBackend"><code>Lux.NCCLBackend</code></a></li><li><a href="#Lux.DistributedUtils.allreduce!"><code>Lux.DistributedUtils.allreduce!</code></a></li><li><a href="#Lux.DistributedUtils.bcast!"><code>Lux.DistributedUtils.bcast!</code></a></li><li><a href="#Lux.DistributedUtils.get_distributed_backend"><code>Lux.DistributedUtils.get_distributed_backend</code></a></li><li><a href="#Lux.DistributedUtils.initialize"><code>Lux.DistributedUtils.initialize</code></a></li><li><a href="#Lux.DistributedUtils.initialized"><code>Lux.DistributedUtils.initialized</code></a></li><li><a href="#Lux.DistributedUtils.local_rank"><code>Lux.DistributedUtils.local_rank</code></a></li><li><a href="#Lux.DistributedUtils.reduce!"><code>Lux.DistributedUtils.reduce!</code></a></li><li><a href="#Lux.DistributedUtils.synchronize!!"><code>Lux.DistributedUtils.synchronize!!</code></a></li><li><a href="#Lux.DistributedUtils.total_workers"><code>Lux.DistributedUtils.total_workers</code></a></li></ul><h2 id="communication-backends" tabindex="-1">Backends <a class="header-anchor" href="#communication-backends" aria-label="Permalink to &quot;Backends {#communication-backends}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.MPIBackend" href="#Lux.MPIBackend">#</a> <b><u>Lux.MPIBackend</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MPIBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(comm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create an MPI backend for distributed training. Users should not use this function directly. Instead use <a href="/previews/PR858/api/Lux/distributed_utils#Lux.DistributedUtils.get_distributed_backend"><code>DistributedUtils.get_distributed_backend(MPIBackend)</code></a>.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/backend.jl#L3-L8" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.NCCLBackend" href="#Lux.NCCLBackend">#</a> <b><u>Lux.NCCLBackend</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NCCLBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(comm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, mpi_backend </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create an NCCL backend for distributed training. Users should not use this function directly. Instead use <a href="/previews/PR858/api/Lux/distributed_utils#Lux.DistributedUtils.get_distributed_backend"><code>DistributedUtils.get_distributed_backend(NCCLBackend)</code></a>.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/backend.jl#L20-L25" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="initialization" tabindex="-1">Initialization <a class="header-anchor" href="#initialization" aria-label="Permalink to &quot;Initialization&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.initialize" href="#Lux.DistributedUtils.initialize">#</a> <b><u>Lux.DistributedUtils.initialize</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDistributedBackend}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Initialize the given backend. Users can supply <code>cuda_devices</code> and <code>amdgpu_devices</code> to initialize the backend with the given devices. These can be set to <code>missing</code> to prevent initialization of the given device type. If set to <code>nothing</code>, and the backend is functional we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to initialize the backend with the given devices.</p><p>Possible values for <code>backend</code> are:</p><ul><li><p><code>MPIBackend</code>: MPI backend for distributed training. Requires <code>MPI.jl</code> to be installed.</p></li><li><p><code>NCCLBackend</code>: NCCL backend for CUDA distributed training. Requires <code>CUDA.jl</code>, <code>MPI.jl</code>, and <code>NCCL.jl</code> to be installed. This also wraps <code>MPI</code> backend for non-CUDA communications.</p></li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L24-L39" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.initialized" href="#Lux.DistributedUtils.initialized">#</a> <b><u>Lux.DistributedUtils.initialized</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialized</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDistributedBackend}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Check if the given backend is initialized.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L16-L20" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.get_distributed_backend" href="#Lux.DistributedUtils.get_distributed_backend">#</a> <b><u>Lux.DistributedUtils.get_distributed_backend</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_distributed_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDistributedBackend}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Get the distributed backend for the given backend type. Possible values are:</p><ul><li><p><code>MPIBackend</code>: MPI backend for distributed training. Requires <code>MPI.jl</code> to be installed.</p></li><li><p><code>NCCLBackend</code>: NCCL backend for CUDA distributed training. Requires <code>CUDA.jl</code>, <code>MPI.jl</code>, and <code>NCCL.jl</code> to be installed. This also wraps <code>MPI</code> backend for non-CUDA communications.</p></li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p><code>initialize(backend; kwargs...)</code> must be called before calling this function.</p></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L48-L61" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.local_rank" href="#Lux.DistributedUtils.local_rank">#</a> <b><u>Lux.DistributedUtils.local_rank</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">local_rank</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Get the local rank for the given backend.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L72-L76" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.total_workers" href="#Lux.DistributedUtils.total_workers">#</a> <b><u>Lux.DistributedUtils.total_workers</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">total_workers</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Get the total number of workers for the given backend.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L81-L85" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Communication-Primitives" tabindex="-1">Communication Primitives <a class="header-anchor" href="#Communication-Primitives" aria-label="Permalink to &quot;Communication Primitives {#Communication-Primitives}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.allreduce!" href="#Lux.DistributedUtils.allreduce!">#</a> <b><u>Lux.DistributedUtils.allreduce!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allreduce!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendrecvbuf, op)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allreduce!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendbuf, recvbuf, op)</span></span></code></pre></div><p>Backend Agnostic API to perform an allreduce operation on the given buffer <code>sendrecvbuf</code> or <code>sendbuf</code> and store the result in <code>recvbuf</code>.</p><p><code>op</code> allows a special <code>DistributedUtils.avg</code> operation that averages the result across all workers.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L121-L130" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.bcast!" href="#Lux.DistributedUtils.bcast!">#</a> <b><u>Lux.DistributedUtils.bcast!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">bcast!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendrecvbuf; root</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">bcast!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendbuf, recvbuf; root</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Backend Agnostic API to broadcast the given buffer <code>sendrecvbuf</code> or <code>sendbuf</code> to all workers into <code>recvbuf</code>. The value at <code>root</code> will be broadcasted to all other workers.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L90-L96" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.reduce!" href="#Lux.DistributedUtils.reduce!">#</a> <b><u>Lux.DistributedUtils.reduce!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reduce!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendrecvbuf, op; root</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reduce!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sendbuf, recvbuf, op; root</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Backend Agnostic API to perform a reduce operation on the given buffer <code>sendrecvbuf</code> or <code>sendbuf</code> and store the result in <code>recvbuf</code>.</p><p><code>op</code> allows a special <code>DistributedUtils.avg</code> operation that averages the result across all workers.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L153-L162" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.synchronize!!" href="#Lux.DistributedUtils.synchronize!!">#</a> <b><u>Lux.DistributedUtils.synchronize!!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">synchronize!!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps; root</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Synchronize the given structure <code>ps</code> using the given backend. The value at <code>root</code> will be broadcasted to all other workers.</p><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L187-L192" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Optimizers.jl-Integration" tabindex="-1">Optimizers.jl Integration <a class="header-anchor" href="#Optimizers.jl-Integration" aria-label="Permalink to &quot;Optimizers.jl Integration {#Optimizers.jl-Integration}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.DistributedOptimizer" href="#Lux.DistributedUtils.DistributedOptimizer">#</a> <b><u>Lux.DistributedUtils.DistributedOptimizer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DistributedOptimizer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBacked</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, optimizer)</span></span></code></pre></div><p>Wrap the <code>optimizer</code> in a <code>DistributedOptimizer</code>. Before updating the parameters, this averages the gradients across the processes using Allreduce.</p><p><strong>Arguments</strong></p><ul><li><code>optimizer</code>: An Optimizer compatible with the Optimisers.jl package</li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L254-L263" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="MLUtils.jl-Integration" tabindex="-1">MLUtils.jl Integration <a class="header-anchor" href="#MLUtils.jl-Integration" aria-label="Permalink to &quot;MLUtils.jl Integration {#MLUtils.jl-Integration}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.DistributedUtils.DistributedDataContainer" href="#Lux.DistributedUtils.DistributedDataContainer">#</a> <b><u>Lux.DistributedUtils.DistributedDataContainer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DistributedDataContainer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDistributedBackend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, data)</span></span></code></pre></div><p><code>data</code> must be compatible with <code>MLUtils</code> interface. The returned container is compatible with <code>MLUtils</code> interface and is used to partition the dataset across the available processes.</p><div class="danger custom-block"><p class="custom-block-title">Load <code>MLUtils.jl</code></p><p><code>MLUtils.jl</code> must be installed and loaded before using this.</p></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/39bfe62fc36d0ee00eed3b7fdcf541e19355c9c0/src/distributed/public_api.jl#L223-L233" target="_blank" rel="noreferrer">source</a></p></div><br>`,36),l=[d];function r(n,o,c,h,p,u){return s(),e("div",null,l)}const g=i(a,[["render",r]]);export{k as __pageData,g as default};
