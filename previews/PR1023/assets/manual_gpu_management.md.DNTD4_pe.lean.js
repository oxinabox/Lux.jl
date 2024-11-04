import{_ as a,c as i,a2 as e,o as n}from"./chunks/framework.DFwXuivk.js";const r=JSON.parse('{"title":"GPU Management","description":"","frontmatter":{},"headers":[],"relativePath":"manual/gpu_management.md","filePath":"manual/gpu_management.md","lastUpdated":null}'),t={name:"manual/gpu_management.md"};function p(l,s,h,c,d,k){return n(),i("div",null,s[0]||(s[0]=[e(`<h1 id="GPU-Management" tabindex="-1">GPU Management <a class="header-anchor" href="#GPU-Management" aria-label="Permalink to &quot;GPU Management {#GPU-Management}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">Info</p><p>Starting from <code>v0.5</code>, Lux has transitioned to a new GPU management system. The old system using <code>cpu</code> and <code>gpu</code> functions is still in place but will be removed in <code>v1</code>. Using the old functions might lead to performance regressions if used inside performance critical code.</p></div><p><code>Lux.jl</code> can handle multiple GPU backends. Currently, the following backends are supported:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Important to load trigger packages</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, LuxCUDA </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">#, AMDGPU, Metal, oneAPI</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">supported_gpu_backends</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(&quot;CUDA&quot;, &quot;AMDGPU&quot;, &quot;Metal&quot;, &quot;oneAPI&quot;)</span></span></code></pre></div><div class="danger custom-block"><p class="custom-block-title">Metal Support</p><p>Support for Metal GPUs should be considered extremely experimental at this point.</p></div><h2 id="Automatic-Backend-Management-(Recommended-Approach)" tabindex="-1">Automatic Backend Management (Recommended Approach) <a class="header-anchor" href="#Automatic-Backend-Management-(Recommended-Approach)" aria-label="Permalink to &quot;Automatic Backend Management (Recommended Approach) {#Automatic-Backend-Management-(Recommended-Approach)}&quot;">​</a></h2><p>Automatic Backend Management is done by two simple functions: <code>cpu_device</code> and <code>gpu_device</code>.</p><ul><li><p><a href="/previews/PR1023/api/Accelerator_Support/MLDataDevices#MLDataDevices.cpu_device"><code>cpu_device</code></a>: This is a simple function and just returns a <code>CPUDevice</code> object. <code>@example gpu_management cdev = cpu_device()</code><code>@example gpu_management x_cpu = randn(Float32, 3, 2)</code></p></li><li><p><a href="/previews/PR1023/api/Accelerator_Support/MLDataDevices#MLDataDevices.gpu_device"><code>gpu_device</code></a>: This function performs automatic GPU device selection and returns an object.</p><ol><li><p>If no GPU is available, it returns a <code>CPUDevice</code> object.</p></li><li><p>If a LocalPreferences file is present, then the backend specified in the file is used. To set a backend, use <code>Lux.gpu_backend!(&lt;backend_name&gt;)</code>. (a) If the trigger package corresponding to the device is not loaded, then a warning is displayed. (b) If no LocalPreferences file is present, then the first working GPU with loaded trigger package is used.</p></li></ol><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>x_gpu = x_cpu |&amp;gt; gdev  \`\`\`</span></span>
<span class="line"><span>\`@example gpu_management  (x_gpu |&gt; cdev) ≈ x_cpu\`</span></span></code></pre></div></li></ul><h2 id="Manual-Backend-Management" tabindex="-1">Manual Backend Management <a class="header-anchor" href="#Manual-Backend-Management" aria-label="Permalink to &quot;Manual Backend Management {#Manual-Backend-Management}&quot;">​</a></h2><p>Automatic Device Selection can be circumvented by directly using <code>CPUDevice</code> and <code>AbstractGPUDevice</code> objects.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x_cpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CUDADevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_cpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">elseif</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AMDGPUDevice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_cpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gdev</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">else</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @info</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;No GPU is available. Using CPU.&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_cpu</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_gpu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">≈</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_cpu</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>true</span></span></code></pre></div>`,13)]))}const g=a(t,[["render",p]]);export{r as __pageData,g as default};
