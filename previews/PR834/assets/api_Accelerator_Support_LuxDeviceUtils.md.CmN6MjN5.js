import{_ as e,c as i,o as s,a4 as t}from"./chunks/framework.GhNgc9YX.js";const g=JSON.parse('{"title":"LuxDeviceUtils","description":"","frontmatter":{},"headers":[],"relativePath":"api/Accelerator_Support/LuxDeviceUtils.md","filePath":"api/Accelerator_Support/LuxDeviceUtils.md","lastUpdated":null}'),a={name:"api/Accelerator_Support/LuxDeviceUtils.md"},l=t(`<h1 id="LuxDeviceUtils-API" tabindex="-1">LuxDeviceUtils <a class="header-anchor" href="#LuxDeviceUtils-API" aria-label="Permalink to &quot;LuxDeviceUtils {#LuxDeviceUtils-API}&quot;">​</a></h1><p><code>LuxDeviceUtils.jl</code> is a lightweight package defining rules for transferring data across devices. Most users should directly use Lux.jl instead.</p><div class="tip custom-block"><p class="custom-block-title">Transition to <code>MLDataDevices.jl</code></p><p>Currently this package is in maintenance mode and won&#39;t receive any new features, however, we will backport bug fixes till Lux <code>v1.0</code> is released. Post that this package should be considered deprecated and users should switch to <code>MLDataDevices.jl</code>.</p><p>For more information on <code>MLDataDevices.jl</code> checkout the <a href="/previews/PR834/api/Accelerator_Support/MLDataDevices#MLDataDevices-API">MLDataDevices.jl Documentation</a>.</p></div><h2 id="index" tabindex="-1">Index <a class="header-anchor" href="#index" aria-label="Permalink to &quot;Index&quot;">​</a></h2><ul><li><a href="#LuxDeviceUtils.cpu_device"><code>LuxDeviceUtils.cpu_device</code></a></li><li><a href="#LuxDeviceUtils.default_device_rng"><code>LuxDeviceUtils.default_device_rng</code></a></li><li><a href="#LuxDeviceUtils.functional"><code>LuxDeviceUtils.functional</code></a></li><li><a href="#LuxDeviceUtils.get_device"><code>LuxDeviceUtils.get_device</code></a></li><li><a href="#LuxDeviceUtils.get_device_type"><code>LuxDeviceUtils.get_device_type</code></a></li><li><a href="#LuxDeviceUtils.gpu_backend!"><code>LuxDeviceUtils.gpu_backend!</code></a></li><li><a href="#LuxDeviceUtils.gpu_device"><code>LuxDeviceUtils.gpu_device</code></a></li><li><a href="#LuxDeviceUtils.loaded"><code>LuxDeviceUtils.loaded</code></a></li><li><a href="#LuxDeviceUtils.reset_gpu_device!"><code>LuxDeviceUtils.reset_gpu_device!</code></a></li><li><a href="#LuxDeviceUtils.set_device!"><code>LuxDeviceUtils.set_device!</code></a></li><li><a href="#LuxDeviceUtils.supported_gpu_backends"><code>LuxDeviceUtils.supported_gpu_backends</code></a></li></ul><h2 id="preferences" tabindex="-1">Preferences <a class="header-anchor" href="#preferences" aria-label="Permalink to &quot;Preferences&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.gpu_backend!" href="#LuxDeviceUtils.gpu_backend!">#</a> <b><u>LuxDeviceUtils.gpu_backend!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxGPUDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_backend!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Creates a <code>LocalPreferences.toml</code> file with the desired GPU backend.</p><p>If <code>backend == &quot;&quot;</code>, then the <code>gpu_backend</code> preference is deleted. Otherwise, <code>backend</code> is validated to be one of the possible backends and the preference is set to <code>backend</code>.</p><p>If a new backend is successfully set, then the Julia session must be restarted for the change to take effect.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L245-L258" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Data-Transfer" tabindex="-1">Data Transfer <a class="header-anchor" href="#Data-Transfer" aria-label="Permalink to &quot;Data Transfer {#Data-Transfer}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.cpu_device" href="#LuxDeviceUtils.cpu_device">#</a> <b><u>LuxDeviceUtils.cpu_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LuxCPUDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Return a <code>LuxCPUDevice</code> object which can be used to transfer data to CPU.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L287-L291" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.gpu_device" href="#LuxDeviceUtils.gpu_device">#</a> <b><u>LuxDeviceUtils.gpu_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(device_id</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Nothing, Integer}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    force_gpu_usage</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Selects GPU device based on the following criteria:</p><ol><li><p>If <code>gpu_backend</code> preference is set and the backend is functional on the system, then that device is selected.</p></li><li><p>Otherwise, an automatic selection algorithm is used. We go over possible device backends in the order specified by <code>supported_gpu_backends()</code> and select the first functional backend.</p></li><li><p>If no GPU device is functional and <code>force_gpu_usage</code> is <code>false</code>, then <code>cpu_device()</code> is invoked.</p></li><li><p>If nothing works, an error is thrown.</p></li></ol><p><strong>Arguments</strong></p><ul><li><code>device_id::Union{Nothing, Integer}</code>: The device id to select. If <code>nothing</code>, then we return the last selected device or if none was selected then we run the autoselection and choose the current device using <code>CUDA.device()</code> or <code>AMDGPU.device()</code> or similar. If <code>Integer</code>, then we select the device with the given id. Note that this is <code>1</code>-indexed, in contrast to the <code>0</code>-indexed <code>CUDA.jl</code>. For example, <code>id = 4</code> corresponds to <code>CUDA.device!(3)</code>.</li></ul><div class="warning custom-block"><p class="custom-block-title">Warning</p><p><code>device_id</code> is only applicable for <code>CUDA</code> and <code>AMDGPU</code> backends. For <code>Metal</code>, <code>oneAPI</code> and <code>CPU</code> backends, <code>device_id</code> is ignored and a warning is printed.</p></div><p><strong>Keyword Arguments</strong></p><ul><li><code>force_gpu_usage::Bool</code>: If <code>true</code>, then an error is thrown if no functional GPU device is found.</li></ul><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L125-L158" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="miscellaneous" tabindex="-1">Miscellaneous <a class="header-anchor" href="#miscellaneous" aria-label="Permalink to &quot;Miscellaneous&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.reset_gpu_device!" href="#LuxDeviceUtils.reset_gpu_device!">#</a> <b><u>LuxDeviceUtils.reset_gpu_device!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reset_gpu_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Resets the selected GPU device. This is useful when automatic GPU selection needs to be run again.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L100-L105" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.supported_gpu_backends" href="#LuxDeviceUtils.supported_gpu_backends">#</a> <b><u>LuxDeviceUtils.supported_gpu_backends</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">supported_gpu_backends</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Tuple{String, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Return a tuple of supported GPU backends.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This is not the list of functional backends on the system, but rather backends which <code>Lux.jl</code> supports.</p></div><div class="danger custom-block"><p class="custom-block-title">Danger</p><p><code>Metal.jl</code> and <code>oneAPI.jl</code> support is <strong>extremely</strong> experimental and most things are not expected to work.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L108-L122" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.default_device_rng" href="#LuxDeviceUtils.default_device_rng">#</a> <b><u>LuxDeviceUtils.default_device_rng</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_device_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns the default RNG for the device. This can be used to directly generate parameters and states on the device using <a href="https://github.com/LuxDL/WeightInitializers.jl" target="_blank" rel="noreferrer">WeightInitializers.jl</a>.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L294-L300" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.get_device" href="#LuxDeviceUtils.get_device">#</a> <b><u>LuxDeviceUtils.get_device</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDevice</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Exception </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span></code></pre></div><p>If all arrays (on the leaves of the structure) are on the same device, we return that device. Otherwise, we throw an error. If the object is device agnostic, we return <code>nothing</code>.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>Trigger Packages must be loaded for this to return the correct device.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>RNG types currently don&#39;t participate in device determination. We will remove this restriction in the future.</p></div><p>See also <a href="/previews/PR834/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.get_device_type"><code>get_device_type</code></a> for a faster alternative that can be used for dispatch based on device type.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L351-L361" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.get_device_type" href="#LuxDeviceUtils.get_device_type">#</a> <b><u>LuxDeviceUtils.get_device_type</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_device_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Type{</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Exception </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Type{Nothing}</span></span></code></pre></div><p>Similar to <a href="/previews/PR834/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.get_device"><code>get_device</code></a> but returns the type of the device instead of the device itself. This value is often a compile time constant and is recommended to be used instead of <a href="/previews/PR834/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.get_device"><code>get_device</code></a> where ever defining dispatches based on the device type.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>Trigger Packages must be loaded for this to return the correct device.</p></div><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>RNG types currently don&#39;t participate in device determination. We will remove this restriction in the future.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L364-L372" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.loaded" href="#LuxDeviceUtils.loaded">#</a> <b><u>LuxDeviceUtils.loaded</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loaded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loaded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p>Checks if the trigger package for the device is loaded. Trigger packages are as follows:</p><ul><li><p><code>LuxCUDA.jl</code> for NVIDIA CUDA Support.</p></li><li><p><code>AMDGPU.jl</code> for AMD GPU ROCM Support.</p></li><li><p><code>Metal.jl</code> for Apple Metal GPU Support.</p></li><li><p><code>oneAPI.jl</code> for Intel oneAPI GPU Support.</p></li></ul><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L36-L46" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.functional" href="#LuxDeviceUtils.functional">#</a> <b><u>LuxDeviceUtils.functional</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLuxDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p>Checks if the device is functional. This is used to determine if the device can be used for computation. Note that even if the backend is loaded (as checked via <a href="/previews/PR834/api/Accelerator_Support/LuxDeviceUtils#LuxDeviceUtils.loaded"><code>LuxDeviceUtils.loaded</code></a>), the device may not be functional.</p><p>Note that while this function is not exported, it is considered part of the public API.</p><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L22-L31" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Multi-GPU-Support" tabindex="-1">Multi-GPU Support <a class="header-anchor" href="#Multi-GPU-Support" aria-label="Permalink to &quot;Multi-GPU Support {#Multi-GPU-Support}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxDeviceUtils.set_device!" href="#LuxDeviceUtils.set_device!">#</a> <b><u>LuxDeviceUtils.set_device!</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dev_or_id)</span></span></code></pre></div><p>Set the device for the given type. This is a no-op for <code>LuxCPUDevice</code>. For <code>LuxCUDADevice</code> and <code>LuxAMDGPUDevice</code>, it prints a warning if the corresponding trigger package is not loaded.</p><p>Currently, <code>LuxMetalDevice</code> and <code>LuxoneAPIDevice</code> doesn&#39;t support setting the device.</p><p><strong>Arguments</strong></p><ul><li><p><code>T::Type{&lt;:AbstractLuxDevice}</code>: The device type to set.</p></li><li><p><code>dev_or_id</code>: Can be the device from the corresponding package. For example for CUDA it can be a <code>CuDevice</code>. If it is an integer, it is the device id to set. This is <code>1</code>-indexed.</p></li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p>This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use <code>Lux.DistributedUtils</code> instead of using this function.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L442-L455" target="_blank" rel="noreferrer">source</a></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_device!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{&lt;:AbstractLuxDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, rank</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Integer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Set the device for the given type. This is a no-op for <code>LuxCPUDevice</code>. For <code>LuxCUDADevice</code> and <code>LuxAMDGPUDevice</code>, it prints a warning if the corresponding trigger package is not loaded.</p><p>Currently, <code>LuxMetalDevice</code> and <code>LuxoneAPIDevice</code> doesn&#39;t support setting the device.</p><p><strong>Arguments</strong></p><ul><li><p><code>T::Type{&lt;:AbstractLuxDevice}</code>: The device type to set.</p></li><li><p><code>rank::Integer</code>: Local Rank of the process. This is applicable for distributed training and must be <code>0</code>-indexed.</p></li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p>This specific function should be considered experimental at this point and is currently provided to support distributed training in Lux. As such please use <code>Lux.DistributedUtils</code> instead of using this function.</p></div><p><a href="https://github.com/LuxDL/MLDataDevices.jl/blob/v0.1.26/src/LuxDeviceUtils.jl#L470-L482" target="_blank" rel="noreferrer">source</a></p></div><br>`,31),d=[l];function n(c,r,p,o,h,u){return s(),i("div",null,d)}const v=e(a,[["render",n]]);export{g as __pageData,v as default};
