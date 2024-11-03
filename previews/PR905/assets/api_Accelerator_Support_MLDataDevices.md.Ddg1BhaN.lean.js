import{_ as s,c as e,a2 as a,o as t}from"./chunks/framework.CvbbdR01.js";const k=JSON.parse('{"title":"MLDataDevices","description":"","frontmatter":{},"headers":[],"relativePath":"api/Accelerator_Support/MLDataDevices.md","filePath":"api/Accelerator_Support/MLDataDevices.md","lastUpdated":null}'),n={name:"api/Accelerator_Support/MLDataDevices.md"};function l(p,i,d,h,r,c){return t(),e("div",null,i[0]||(i[0]=[a(`<h1 id="MLDataDevices-API" tabindex="-1">MLDataDevices <a class="header-anchor" href="#MLDataDevices-API" aria-label="Permalink to &quot;MLDataDevices {#MLDataDevices-API}&quot;">​</a></h1><p><code>MLDataDevices.jl</code> is a lightweight package defining rules for transferring data across devices. Most users should directly use Lux.jl instead.</p><div class="tip custom-block"><p class="custom-block-title">Transitioning from <code>LuxDeviceUtils.jl</code></p><p><code>LuxDeviceUtils.jl</code> was renamed to <code>MLDataDevices.jl</code> in v1.0 as a part of allowing these packages to have broader adoption outsize the Lux community. However, Lux currently still uses <code>LuxDeviceUtils.jl</code> internally. This is supposed to change with the transition of Lux to <code>v1.0</code>.</p></div><h2 id="index" tabindex="-1">Index <a class="header-anchor" href="#index" aria-label="Permalink to &quot;Index&quot;">​</a></h2><ul><li><a href="#MLDataDevices.DeviceIterator"><code>MLDataDevices.DeviceIterator</code></a></li><li><a href="#MLDataDevices.cpu_device"><code>MLDataDevices.cpu_device</code></a></li><li><a href="#MLDataDevices.default_device_rng"><code>MLDataDevices.default_device_rng</code></a></li><li><a href="#MLDataDevices.functional"><code>MLDataDevices.functional</code></a></li><li><a href="#MLDataDevices.get_device"><code>MLDataDevices.get_device</code></a></li><li><a href="#MLDataDevices.get_device_type"><code>MLDataDevices.get_device_type</code></a></li><li><a href="#MLDataDevices.gpu_backend!"><code>MLDataDevices.gpu_backend!</code></a></li><li><a href="#MLDataDevices.gpu_device"><code>MLDataDevices.gpu_device</code></a></li><li><a href="#MLDataDevices.loaded"><code>MLDataDevices.loaded</code></a></li><li><a href="#MLDataDevices.reset_gpu_device!"><code>MLDataDevices.reset_gpu_device!</code></a></li><li><a href="#MLDataDevices.set_device!"><code>MLDataDevices.set_device!</code></a></li><li><a href="#MLDataDevices.supported_gpu_backends"><code>MLDataDevices.supported_gpu_backends</code></a></li></ul><h2 id="preferences" tabindex="-1">Preferences <a class="header-anchor" href="#preferences" aria-label="Permalink to &quot;Preferences&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.gpu_backend!" href="#MLDataDevices.gpu_backend!">#</a> <b><u>MLDataDevices.gpu_backend!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractGPUDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Creates a <code>LocalPreferences.toml</code> file with the desired GPU backend.</p><p>If <code>backend == &quot;&quot;</code>, then the <code>gpu_backend</code> preference is deleted. Otherwise, <code>backend</code> is validated to be one of the possible backends and the preference is set to <code>backend</code>.</p><p>If a new backend is successfully set, then the Julia session must be restarted for the change to take effect.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L128-L141" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Data-Transfer" tabindex="-1">Data Transfer <a class="header-anchor" href="#Data-Transfer" aria-label="Permalink to &quot;Data Transfer {#Data-Transfer}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.cpu_device" href="#MLDataDevices.cpu_device">#</a> <b><u>MLDataDevices.cpu_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CPUDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Return a <code>CPUDevice</code> object which can be used to transfer data to CPU.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L170-L174" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.gpu_device" href="#MLDataDevices.gpu_device">#</a> <b><u>MLDataDevices.gpu_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(device_id</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, Integer}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    force_gpu_usage</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Selects GPU device based on the following criteria:</p><ol><li><p>If <code>gpu_backend</code> preference is set and the backend is functional on the system, then that device is selected.</p></li><li><p>Otherwise, an automatic selection algorithm is used. We go over possible device backends in the order specified by <code>supported_gpu_backends()</code> and select the first functional backend.</p></li><li><p>If no GPU device is functional and <code>force_gpu_usage</code> is <code>false</code>, then <code>cpu_device()</code> is invoked.</p></li><li><p>If nothing works, an error is thrown.</p></li></ol><p><strong>Arguments</strong></p><ul><li><code>device_id::Union{Nothing, Integer}</code>: The device id to select. If <code>nothing</code>, then we return the last selected device or if none was selected then we run the autoselection and choose the current device using <code>CUDA.device()</code> or <code>AMDGPU.device()</code> or similar. If <code>Integer</code>, then we select the device with the given id. Note that this is <code>1</code>-indexed, in contrast to the <code>0</code>-indexed <code>CUDA.jl</code>. For example, <code>id = 4</code> corresponds to <code>CUDA.device!(3)</code>.</li></ul><div class="warning custom-block"><p class="custom-block-title">Warning</p><p><code>device_id</code> is only applicable for <code>CUDA</code> and <code>AMDGPU</code> backends. For <code>Metal</code>, <code>oneAPI</code> and <code>CPU</code> backends, <code>device_id</code> is ignored and a warning is printed.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p><code>gpu_device</code> won&#39;t select a CUDA device unless both CUDA.jl and cuDNN.jl are loaded. This is to ensure that deep learning operations work correctly. Nonetheless, if cuDNN is not loaded you can still manually create a <code>CUDADevice</code> object and use it (e.g. <code>dev = CUDADevice()</code>).</p></div><p><strong>Keyword Arguments</strong></p><ul><li><code>force_gpu_usage::Bool</code>: If <code>true</code>, then an error is thrown if no functional GPU device is found.</li></ul><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L63-L103" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="miscellaneous" tabindex="-1">Miscellaneous <a class="header-anchor" href="#miscellaneous" aria-label="Permalink to &quot;Miscellaneous&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.reset_gpu_device!" href="#MLDataDevices.reset_gpu_device!">#</a> <b><u>MLDataDevices.reset_gpu_device!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reset_gpu_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Resets the selected GPU device. This is useful when automatic GPU selection needs to be run again.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L43-L48" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.supported_gpu_backends" href="#MLDataDevices.supported_gpu_backends">#</a> <b><u>MLDataDevices.supported_gpu_backends</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">supported_gpu_backends</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{String, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Return a tuple of supported GPU backends.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This is not the list of functional backends on the system, but rather backends which <code>MLDataDevices.jl</code> supports.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L51-L60" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.default_device_rng" href="#MLDataDevices.default_device_rng">#</a> <b><u>MLDataDevices.default_device_rng</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_device_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns the default RNG for the device. This can be used to directly generate parameters and states on the device using <a href="https://github.com/LuxDL/WeightInitializers.jl" target="_blank" rel="noreferrer">WeightInitializers.jl</a>.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L177-L183" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.get_device" href="#MLDataDevices.get_device">#</a> <b><u>MLDataDevices.get_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Exception </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Nothing</span></span></code></pre></div><p>If all arrays (on the leaves of the structure) are on the same device, we return that device. Otherwise, we throw an error. If the object is device agnostic, we return <code>nothing</code>.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>Trigger Packages must be loaded for this to return the correct device.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>RNG types currently don&#39;t participate in device determination. We will remove this restriction in the future.</p></div><p>See also <a href="/previews/PR905/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device_type"><code>get_device_type</code></a> for a faster alternative that can be used for dispatch based on device type.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L206-L216" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.get_device_type" href="#MLDataDevices.get_device_type">#</a> <b><u>MLDataDevices.get_device_type</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_device_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Type{</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Exception </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Type{Nothing}</span></span></code></pre></div><p>Similar to <a href="/previews/PR905/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device"><code>get_device</code></a> but returns the type of the device instead of the device itself. This value is often a compile time constant and is recommended to be used instead of <a href="/previews/PR905/api/Accelerator_Support/MLDataDevices#MLDataDevices.get_device"><code>get_device</code></a> where ever defining dispatches based on the device type.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>Trigger Packages must be loaded for this to return the correct device.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>RNG types currently don&#39;t participate in device determination. We will remove this restriction in the future.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L219-L227" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.loaded" href="#MLDataDevices.loaded">#</a> <b><u>MLDataDevices.loaded</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loaded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loaded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p>Checks if the trigger package for the device is loaded. Trigger packages are as follows:</p><ul><li><p><code>CUDA.jl</code> and <code>cuDNN.jl</code> (or just <code>LuxCUDA.jl</code>) for NVIDIA CUDA Support.</p></li><li><p><code>AMDGPU.jl</code> for AMD GPU ROCM Support.</p></li><li><p><code>Metal.jl</code> for Apple Metal GPU Support.</p></li><li><p><code>oneAPI.jl</code> for Intel oneAPI GPU Support.</p></li></ul><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L24-L34" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.functional" href="#MLDataDevices.functional">#</a> <b><u>MLDataDevices.functional</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p>Checks if the device is functional. This is used to determine if the device can be used for computation. Note that even if the backend is loaded (as checked via <a href="/previews/PR905/api/Accelerator_Support/MLDataDevices#MLDataDevices.loaded"><code>MLDataDevices.loaded</code></a>), the device may not be functional.</p><p>Note that while this function is not exported, it is considered part of the public API.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L11-L20" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Multi-GPU-Support" tabindex="-1">Multi-GPU Support <a class="header-anchor" href="#Multi-GPU-Support" aria-label="Permalink to &quot;Multi-GPU Support {#Multi-GPU-Support}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.set_device!" href="#MLDataDevices.set_device!">#</a> <b><u>MLDataDevices.set_device!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dev_or_id)</span></span></code></pre></div><p>Set the device for the given type. This is a no-op for <code>CPUDevice</code>. For <code>CUDADevice</code> and <code>AMDGPUDevice</code>, it prints a warning if the corresponding trigger package is not loaded.</p><p>Currently, <code>MetalDevice</code> and <code>oneAPIDevice</code> don&#39;t support setting the device.</p><p><strong>Arguments</strong></p><ul><li><p><code>T::Type{&lt;:AbstractDevice}</code>: The device type to set.</p></li><li><p><code>dev_or_id</code>: Can be the device from the corresponding package. For example for CUDA it can be a <code>CuDevice</code>. If it is an integer, it is the device id to set. This is <code>1</code>-indexed.</p></li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p>This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use <code>Lux.DistributedUtils</code> instead of using this function.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L247-L260" target="_blank" rel="noreferrer">source</a></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, rank</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Integer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Set the device for the given type. This is a no-op for <code>CPUDevice</code>. For <code>CUDADevice</code> and <code>AMDGPUDevice</code>, it prints a warning if the corresponding trigger package is not loaded.</p><p>Currently, <code>MetalDevice</code> and <code>oneAPIDevice</code> don&#39;t support setting the device.</p><p><strong>Arguments</strong></p><ul><li><p><code>T::Type{&lt;:AbstractDevice}</code>: The device type to set.</p></li><li><p><code>rank::Integer</code>: Local Rank of the process. This is applicable for distributed training and must be <code>0</code>-indexed.</p></li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p>This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use <code>Lux.DistributedUtils</code> instead of using this function.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/public.jl#L274-L286" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="iteration" tabindex="-1">Iteration <a class="header-anchor" href="#iteration" aria-label="Permalink to &quot;Iteration&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="MLDataDevices.DeviceIterator" href="#MLDataDevices.DeviceIterator">#</a> <b><u>MLDataDevices.DeviceIterator</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DeviceIterator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, iterator)</span></span></code></pre></div><p>Create a <code>DeviceIterator</code> that iterates through the provided <code>iterator</code> via <code>iterate</code>. Upon each iteration, the current batch is copied to the device <code>dev</code>, and the previous iteration is marked as freeable from GPU memory (via <code>unsafe_free!</code>) (no-op for a CPU device).</p><p>The conversion follows the same semantics as <code>dev(&lt;item from iterator&gt;)</code>.</p><div class="tip custom-block"><p class="custom-block-title">Similarity to <code>CUDA.CuIterator</code></p><p>The design inspiration was taken from <code>CUDA.CuIterator</code> and was generalized to work with other backends and more complex iterators (using <code>Functors</code>).</p></div><div class="tip custom-block"><p class="custom-block-title"><code>MLUtils.DataLoader</code></p><p>Calling <code>dev(::MLUtils.DataLoader)</code> will automatically convert the dataloader to use the same semantics as <code>DeviceIterator</code>. This is generally preferred over looping over the dataloader directly and transferring the data to the device.</p></div><p><strong>Examples</strong></p><p>The following was run on a computer with an NVIDIA GPU.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices, MLUtils</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float64, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">33</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(X; batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">13</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (i, x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> enumerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataloader)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">           @show</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">       end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×13 Matrix{Float64}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×13 Matrix{Float64}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×7 Matrix{Float64}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (i, x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> enumerate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CUDADevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()(dataloader))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">           @show</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">       end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×13 CuArray{Float32, 2, CUDA.DeviceMemory}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×13 CuArray{Float32, 2, CUDA.DeviceMemory}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">summary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3×7 CuArray{Float32, 2, CUDA.DeviceMemory}&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v1.1.1/src/iterator.jl#L1-L46" target="_blank" rel="noreferrer">source</a></p></div><br>`,34)]))}const g=s(n,[["render",l]]);export{k as __pageData,g as default};
