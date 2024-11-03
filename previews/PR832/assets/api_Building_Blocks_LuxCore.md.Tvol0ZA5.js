import{_ as e,c as a,o as i,a4 as t}from"./chunks/framework.ClmyTGR2.js";const k=JSON.parse('{"title":"LuxCore","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/LuxCore.md","filePath":"api/Building_Blocks/LuxCore.md","lastUpdated":null}'),s={name:"api/Building_Blocks/LuxCore.md"},r=t(`<h1 id="luxcore" tabindex="-1">LuxCore <a class="header-anchor" href="#luxcore" aria-label="Permalink to &quot;LuxCore&quot;">​</a></h1><p><code>LuxCore.jl</code> defines the abstract layers for Lux. Allows users to be compatible with the entirely of <code>Lux.jl</code> without having such a heavy dependency. If you are depending on <code>Lux.jl</code> directly, you do not need to depend on <code>LuxCore.jl</code> (all the functionality is exported via <code>Lux.jl</code>).</p><h2 id="index" tabindex="-1">Index <a class="header-anchor" href="#index" aria-label="Permalink to &quot;Index&quot;">​</a></h2><ul><li><a href="#LuxCore.AbstractExplicitContainerLayer"><code>LuxCore.AbstractExplicitContainerLayer</code></a></li><li><a href="#LuxCore.AbstractExplicitLayer"><code>LuxCore.AbstractExplicitLayer</code></a></li><li><a href="#LuxCore.apply"><code>LuxCore.apply</code></a></li><li><a href="#LuxCore.check_fmap_condition"><code>LuxCore.check_fmap_condition</code></a></li><li><a href="#LuxCore.contains_lux_layer"><code>LuxCore.contains_lux_layer</code></a></li><li><a href="#LuxCore.display_name"><code>LuxCore.display_name</code></a></li><li><a href="#LuxCore.initialparameters"><code>LuxCore.initialparameters</code></a></li><li><a href="#LuxCore.initialstates"><code>LuxCore.initialstates</code></a></li><li><a href="#LuxCore.inputsize"><code>LuxCore.inputsize</code></a></li><li><a href="#LuxCore.outputsize"><code>LuxCore.outputsize</code></a></li><li><a href="#LuxCore.parameterlength"><code>LuxCore.parameterlength</code></a></li><li><a href="#LuxCore.replicate"><code>LuxCore.replicate</code></a></li><li><a href="#LuxCore.setup"><code>LuxCore.setup</code></a></li><li><a href="#LuxCore.statelength"><code>LuxCore.statelength</code></a></li><li><a href="#LuxCore.stateless_apply"><code>LuxCore.stateless_apply</code></a></li><li><a href="#LuxCore.testmode"><code>LuxCore.testmode</code></a></li><li><a href="#LuxCore.trainmode"><code>LuxCore.trainmode</code></a></li><li><a href="#LuxCore.update_state"><code>LuxCore.update_state</code></a></li></ul><h2 id="Abstract-Types" tabindex="-1">Abstract Types <a class="header-anchor" href="#Abstract-Types" aria-label="Permalink to &quot;Abstract Types {#Abstract-Types}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.AbstractExplicitLayer" href="#LuxCore.AbstractExplicitLayer">#</a> <b><u>LuxCore.AbstractExplicitLayer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">abstract type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AbstractExplicitLayer</span></span></code></pre></div><p>Abstract Type for all Lux Layers</p><p>Users implementing their custom layer, <strong>must</strong> implement</p><ul><li><p><code>initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)</code> – This returns a <code>NamedTuple</code> containing the trainable parameters for the layer.</p></li><li><p><code>initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)</code> – This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include <code>BatchNorm</code>, <code>LSTM</code>, <code>GRU</code>, etc.</p></li></ul><p>Optionally:</p><ul><li><p><code>parameterlength(layer::CustomAbstractExplicitLayer)</code> – These can be automatically calculated, but it is recommended that the user defines these.</p></li><li><p><code>statelength(layer::CustomAbstractExplicitLayer)</code> – These can be automatically calculated, but it is recommended that the user defines these.</p></li></ul><p>See also <a href="/previews/PR832/api/Building_Blocks/LuxCore#LuxCore.AbstractExplicitContainerLayer"><code>AbstractExplicitContainerLayer</code></a></p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L26-L48" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.AbstractExplicitContainerLayer" href="#LuxCore.AbstractExplicitContainerLayer">#</a> <b><u>LuxCore.AbstractExplicitContainerLayer</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">abstract type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AbstractExplicitContainerLayer{layers} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractExplicitLayer</span></span></code></pre></div><p>Abstract Container Type for certain Lux Layers. <code>layers</code> is a tuple containing fieldnames for the layer, and constructs the parameters and states using those.</p><p>Users implementing their custom layer can extend the same functions as in <a href="/previews/PR832/api/Building_Blocks/LuxCore#LuxCore.AbstractExplicitLayer"><code>AbstractExplicitLayer</code></a>.</p><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>Advanced structure manipulation of these layers post construction is possible via <code>Functors.fmap</code>. For a more flexible interface, we recommend using <code>Lux.Experimental.@layer_map</code>.</p></div><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L199-L213" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="general" tabindex="-1">General <a class="header-anchor" href="#general" aria-label="Permalink to &quot;General&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.apply" href="#LuxCore.apply">#</a> <b><u>LuxCore.apply</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, x, ps, st)</span></span></code></pre></div><p>In most cases this function simply calls <code>model(x, ps, st)</code>. However, it is still recommended to call <code>apply</code> instead of <code>model(x, ps, st)</code> directly. Some of the reasons for this include:</p><ol><li><p>For certain types of inputs <code>x</code>, we might want to perform preprocessing before calling <code>model</code>. For eg, if <code>x</code> is an Array of <code>ReverseDiff.TrackedReal</code>s this can cause significant regressions in <code>model(x, ps, st)</code> (since it won&#39;t hit any of the BLAS dispatches). In those cases, we would automatically convert <code>x</code> to a <code>ReverseDiff.TrackedArray</code>.</p></li><li><p>Certain user defined inputs need to be applied to specific layers but we want the datatype of propagate through all the layers (even unsupported ones). In these cases, we can unpack the input in <code>apply</code> and pass it to the appropriate layer and then repack it before returning. See the Lux manual on Custom Input Types for a motivating example.</p></li></ol><div class="tip custom-block"><p class="custom-block-title">Tip</p><p><code>apply</code> is integrated with <code>DispatchDoctor.jl</code> that allows automatic verification of type stability. By default this is &quot;disable&quot;d. For more information, see the <a href="https://github.com/MilesCranmer/DispatchDoctor.jl" target="_blank" rel="noreferrer">documentation</a>.</p></div><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L146-L169" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.stateless_apply" href="#LuxCore.stateless_apply">#</a> <b><u>LuxCore.stateless_apply</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">stateless_apply</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, x, ps)</span></span></code></pre></div><p>Calls <code>apply</code> and only returns the first argument. This function requires that <code>model</code> has an empty state of <code>NamedTuple()</code>. Behavior of other kinds of models are undefined and it is the responsibility of the user to ensure that the model has an empty state.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L174-L180" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.check_fmap_condition" href="#LuxCore.check_fmap_condition">#</a> <b><u>LuxCore.check_fmap_condition</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">check_fmap_condition</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(cond, tmatch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Union{Type, Nothing}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p><code>fmap</code>s into the structure <code>x</code> and see if <code>cond</code> is satisfied for any of the leaf elements.</p><p><strong>Arguments</strong></p><ul><li><p><code>cond</code> - A function that takes a single argument and returns a <code>Bool</code>.</p></li><li><p><code>tmatch</code> - A shortcut to check if <code>x</code> is of type <code>tmatch</code>. Can be disabled by passing <code>nothing</code>.</p></li><li><p><code>x</code> - The structure to check.</p></li></ul><p><strong>Returns</strong></p><p>A Boolean Value</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L299-L314" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.contains_lux_layer" href="#LuxCore.contains_lux_layer">#</a> <b><u>LuxCore.contains_lux_layer</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">contains_lux_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(l) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Bool</span></span></code></pre></div><p>Check if the structure <code>l</code> is a Lux AbstractExplicitLayer or a container of such a layer.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L289-L293" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.display_name" href="#LuxCore.display_name">#</a> <b><u>LuxCore.display_name</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">display_name</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractExplicitLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Printed Name of the <code>layer</code>. If the <code>layer</code> has a field <code>name</code> that is used, else the type name is used.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L185-L190" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.replicate" href="#LuxCore.replicate">#</a> <b><u>LuxCore.replicate</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">replicate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Creates a copy of the <code>rng</code> state depending on its type.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L10-L14" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.setup" href="#LuxCore.setup">#</a> <b><u>LuxCore.setup</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, layer)</span></span></code></pre></div><p>Shorthand for getting the parameters and states of the layer <code>l</code>. Is equivalent to <code>(initialparameters(rng, l), initialstates(rng, l))</code>.</p><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>This function is not pure, it mutates <code>rng</code>.</p></div><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L134-L143" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="parameters" tabindex="-1">Parameters <a class="header-anchor" href="#parameters" aria-label="Permalink to &quot;Parameters&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.initialparameters" href="#LuxCore.initialparameters">#</a> <b><u>LuxCore.initialparameters</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialparameters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, layer)</span></span></code></pre></div><p>Generate the initial parameters of the layer <code>l</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L51-L55" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.parameterlength" href="#LuxCore.parameterlength">#</a> <b><u>LuxCore.parameterlength</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer)</span></span></code></pre></div><p>Return the total number of parameters of the layer <code>l</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L82-L86" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="states" tabindex="-1">States <a class="header-anchor" href="#states" aria-label="Permalink to &quot;States&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.initialstates" href="#LuxCore.initialstates">#</a> <b><u>LuxCore.initialstates</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">initialstates</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractRNG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, layer)</span></span></code></pre></div><p>Generate the initial states of the layer <code>l</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L58-L62" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.statelength" href="#LuxCore.statelength">#</a> <b><u>LuxCore.statelength</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">statelength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer)</span></span></code></pre></div><p>Return the total number of states of the layer <code>l</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L95-L99" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.testmode" href="#LuxCore.testmode">#</a> <b><u>LuxCore.testmode</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Make all occurrences of <code>training</code> in state <code>st</code> – <code>Val(false)</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L255-L259" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.trainmode" href="#LuxCore.trainmode">#</a> <b><u>LuxCore.trainmode</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">trainmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Make all occurrences of <code>training</code> in state <code>st</code> – <code>Val(true)</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L262-L266" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.update_state" href="#LuxCore.update_state">#</a> <b><u>LuxCore.update_state</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">update_state</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, key</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, value;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_check</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_default_layer_check</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(key))</span></span></code></pre></div><p>Recursively update all occurrences of the <code>key</code> in the state <code>st</code> with the <code>value</code>.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L269-L274" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Layer-size" tabindex="-1">Layer size <a class="header-anchor" href="#Layer-size" aria-label="Permalink to &quot;Layer size {#Layer-size}&quot;">​</a></h2><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>These specifications have been added very recently and most layers currently do not implement them.</p></div><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.inputsize" href="#LuxCore.inputsize">#</a> <b><u>LuxCore.inputsize</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">inputsize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer)</span></span></code></pre></div><p>Return the input size of the layer.</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L105-L109" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="LuxCore.outputsize" href="#LuxCore.outputsize">#</a> <b><u>LuxCore.outputsize</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">outputsize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer, x, rng)</span></span></code></pre></div><p>Return the output size of the layer. If <code>outputsize(layer)</code> is defined, that method takes precedence, else we compute the layer output to determine the final size.</p><p>The fallback implementation of this function assumes the inputs were batched, i.e., if any of the outputs are Arrays, with <code>ndims(A) &gt; 1</code>, it will return <code>size(A)[1:(end - 1)]</code>. If this behavior is undesirable, provide a custom <code>outputsize(layer, x, rng)</code> implementation).</p><p><a href="https://github.com/LuxDL/LuxCore.jl/blob/v0.1.23/src/LuxCore.jl#L116-L126" target="_blank" rel="noreferrer">source</a></p></div><br>`,46),o=[r];function l(d,n,p,c,h,u){return i(),a("div",null,o)}const g=e(s,[["render",l]]);export{k as __pageData,g as default};
