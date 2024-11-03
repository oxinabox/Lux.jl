import{_ as s,c as a,a2 as t,o as e}from"./chunks/framework.Bg5zKH4v.js";const g=JSON.parse('{"title":"","description":"","frontmatter":{"layout":"home","hero":{"name":"LuxDL Docs","text":"Elegant & Performant Scientific Machine Learning in JuliaLang","tagline":"A Pure Julia Deep Learning Framework designed for Scientific Machine Learning","actions":[{"theme":"brand","text":"Tutorials","link":"/tutorials"},{"theme":"alt","text":"Ecosystem","link":"/ecosystem"},{"theme":"alt","text":"API Reference 📚","link":"/api/Lux/layers"},{"theme":"alt","text":"View on GitHub","link":"https://github.com/LuxDL/Lux.jl"}],"image":{"src":"/lux-logo.svg","alt":"Lux.jl"}},"features":[{"icon":"🚀","title":"Fast & Extendible","details":"Lux.jl is written in Julia itself, making it extremely extendible. CUDA and AMDGPU are supported first-class, with experimental support for Metal Hardware.","link":"/introduction"},{"icon":"🧑‍🔬","title":"SciML ❤️ Lux","details":"Lux is the default choice for many SciML packages, including DiffEqFlux.jl, NeuralPDE.jl, and more.","link":"https://sciml.ai/"},{"icon":"🧩","title":"Uniquely Composable","details":"Lux.jl natively supports Arbitrary Parameter Types, making it uniquely composable with other Julia packages (and even Non-Julia packages).","link":"/api/Lux/contrib#Training"},{"icon":"🧪","title":"Well Tested","details":"Lux.jl tests every supported Automatic Differentiation Framework with every supported hardware backend against Finite Differences to prevent sneaky 🐛 in your code.","link":"/api/Testing_Functionality/LuxTestUtils"}]},"headers":[],"relativePath":"index.md","filePath":"index.md","lastUpdated":null}'),l={name:"index.md"};function n(p,i,h,k,d,r){return e(),a("div",null,i[0]||(i[0]=[t(`<h2 id="How-to-Install-Lux.jl?" tabindex="-1">How to Install Lux.jl? <a class="header-anchor" href="#How-to-Install-Lux.jl?" aria-label="Permalink to &quot;How to Install Lux.jl? {#How-to-Install-Lux.jl?}&quot;">​</a></h2><p>Its easy to install Lux.jl. Since Lux.jl is registered in the Julia General registry, you can simply run the following command in the Julia REPL:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Lux&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>If you want to use the latest unreleased version of Lux.jl, you can run the following command: (in most cases the released version will be same as the version on github)</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;https://github.com/LuxDL/Lux.jl&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><h2 id="Want-GPU-Support?" tabindex="-1">Want GPU Support? <a class="header-anchor" href="#Want-GPU-Support?" aria-label="Permalink to &quot;Want GPU Support? {#Want-GPU-Support?}&quot;">​</a></h2><p>Install the following package(s):</p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-gmlln" id="tab-UlW0lzO" checked><label for="tab-UlW0lzO">NVIDIA GPUs</label><input type="radio" name="group-gmlln" id="tab-D-aUIEL"><label for="tab-D-aUIEL">AMD ROCm GPUs</label><input type="radio" name="group-gmlln" id="tab-jcwp3fJ"><label for="tab-jcwp3fJ">Metal M-Series GPUs</label><input type="radio" name="group-gmlln" id="tab-9ITEDPw"><label for="tab-9ITEDPw">Intel GPUs</label></div><div class="blocks"><div class="language-julia vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;LuxCUDA&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># or</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;CUDA&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cuDNN&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;AMDGPU&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Metal&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;oneAPI&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div></div></div>`,8)]))}const u=s(l,[["render",n]]);export{g as __pageData,u as default};
