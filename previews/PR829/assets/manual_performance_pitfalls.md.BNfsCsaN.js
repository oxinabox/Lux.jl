import{_ as s,c as i,o as a,a4 as e}from"./chunks/framework.GYfaOXHm.js";const u=JSON.parse('{"title":"Performance Pitfalls &amp; How to Catch Them","description":"","frontmatter":{},"headers":[],"relativePath":"manual/performance_pitfalls.md","filePath":"manual/performance_pitfalls.md","lastUpdated":null}'),t={name:"manual/performance_pitfalls.md"},n=e(`<h1 id="Performance-Pitfalls-and-How-to-Catch-Them" tabindex="-1">Performance Pitfalls &amp; How to Catch Them <a class="header-anchor" href="#Performance-Pitfalls-and-How-to-Catch-Them" aria-label="Permalink to &quot;Performance Pitfalls &amp;amp; How to Catch Them {#Performance-Pitfalls-and-How-to-Catch-Them}&quot;">​</a></h1><p>Go through the following documentations for general performance tips:</p><ol><li><p><a href="https://docs.julialang.org/en/v1/manual/performance-tips/" target="_blank" rel="noreferrer">Official Julia Performance Tips</a>.</p></li><li><p><a href="/previews/PR829/manual/autodiff#autodiff-recommendations">Recommendations for selecting AD packages</a>.</p></li></ol><h2 id="Spurious-Type-Promotion" tabindex="-1">Spurious Type-Promotion <a class="header-anchor" href="#Spurious-Type-Promotion" aria-label="Permalink to &quot;Spurious Type-Promotion {#Spurious-Type-Promotion}&quot;">​</a></h2><p>Lux by-default uses Julia semantics for type-promotions, while this means that we do the &quot;correct&quot; numerical thing, this can often come as a surprise to users coming from a more deep learning background. For example, consider the following code:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Random</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, gelu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">recursive_eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((ps, st))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float32</span></span></code></pre></div><p>As we can see that <code>ps</code> and <code>st</code> are structures with the highest precision being <code>Float32</code>. Now let&#39;s run the model using some random data:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float64</span></span></code></pre></div><p>Oops our output became <code>Float64</code>. This will be bad on CPUs but an absolute performance disaster on GPUs. The reason this happened is that our input <code>x</code> was <code>Float64</code>. Instead, we should have used <code>Float32</code> input:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Float32</span></span></code></pre></div><p>This was easy to fix for a small model. But certain layers might incorrectly promote objects to a higher precision. This will cause a regression in performance. There are 2 recommendations to fix this or track them down:</p><ol><li><p>Use <a href="/previews/PR829/manual/debugging#debug-lux-layers"><code>Lux.Experimental.@debug_mode</code></a> to see which layer is causing the type-promotion.</p></li><li><p>Alternatively to control the global behavior of eltypes in Lux and allow it to auto-correct the precision use <a href="/previews/PR829/api/Lux/utilities#Lux.match_eltype"><code>match_eltype</code></a> and the <a href="/previews/PR829/manual/preferences#automatic-eltypes-preference"><code>eltype_mismatch_handling</code></a> preference.</p></li></ol><h2 id="Scalar-Indexing-on-GPU-Arrays" tabindex="-1">Scalar Indexing on GPU Arrays <a class="header-anchor" href="#Scalar-Indexing-on-GPU-Arrays" aria-label="Permalink to &quot;Scalar Indexing on GPU Arrays {#Scalar-Indexing-on-GPU-Arrays}&quot;">​</a></h2><p>When running code on GPUs, it is recommended to <a href="https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar" target="_blank" rel="noreferrer">disallow scalar indexing</a>. Note that this is disabled by default except in REPL. You can disable it even in REPL mode using:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> GPUArraysCore</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">GPUArraysCore</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">allowscalar</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><h2 id="Type-Instabilities" tabindex="-1">Type Instabilities <a class="header-anchor" href="#Type-Instabilities" aria-label="Permalink to &quot;Type Instabilities {#Type-Instabilities}&quot;">​</a></h2><p><code>Lux.jl</code> is integrated with <code>DispatchDoctor.jl</code> to catch type instabilities. You can easily enable it by setting the <code>instability_check</code> preference. This will help you catch type instabilities in your code. For more information on how to set preferences, check out <a href="/previews/PR829/api/Lux/utilities#Lux.set_dispatch_doctor_preferences!"><code>Lux.set_dispatch_doctor_preferences!</code></a>.</p>`,20),l=[n];function p(h,o,r,d,c,k){return a(),i("div",null,l)}const E=s(t,[["render",p]]);export{u as __pageData,E as default};
