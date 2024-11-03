import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CGE-QiV-.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.2/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.2/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.2/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.2/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62105</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59012</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57846</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53616</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51960</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50188</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47619</span></span>
<span class="line"><span>Validation: Loss 0.46270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46282</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46087</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44761</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42874</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41352</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39352</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36613</span></span>
<span class="line"><span>Validation: Loss 0.36464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38250</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35839</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34617</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32904</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33328</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29715</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29360</span></span>
<span class="line"><span>Validation: Loss 0.27939 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29342</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27501</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27018</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25193</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25523</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23191</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20409</span></span>
<span class="line"><span>Validation: Loss 0.20976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22603 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22495</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20204</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20478</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19399</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18426</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17860</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15872</span></span>
<span class="line"><span>Validation: Loss 0.15488 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16980 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16499</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15829</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13559</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15194</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14243</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12202</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14704</span></span>
<span class="line"><span>Validation: Loss 0.11319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12147</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12116</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10334</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09850</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10009</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09603</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09994</span></span>
<span class="line"><span>Validation: Loss 0.08091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08983 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08312</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09099</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07382</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07461</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06279</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06942</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06492</span></span>
<span class="line"><span>Validation: Loss 0.05633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06219 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06049</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05702</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05282</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05185</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05046</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04714</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04247</span></span>
<span class="line"><span>Validation: Loss 0.04164 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04569 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04766</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04016</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03854</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03958</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03905</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03783</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03740</span></span>
<span class="line"><span>Validation: Loss 0.03355 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03473</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03632</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03378</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03240</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03268</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02964</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03159</span></span>
<span class="line"><span>Validation: Loss 0.02836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03120 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03158</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03068</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02755</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02723</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02627</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02643</span></span>
<span class="line"><span>Validation: Loss 0.02461 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02790</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02456</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02478</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02477</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02167</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02249</span></span>
<span class="line"><span>Validation: Loss 0.02173 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02400 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02234</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02312</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02333</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01972</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02224</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02209</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02009</span></span>
<span class="line"><span>Validation: Loss 0.01943 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02052</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01926</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02073</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01973</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01741</span></span>
<span class="line"><span>Validation: Loss 0.01752 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01943 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01795</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01999</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01662</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02022</span></span>
<span class="line"><span>Validation: Loss 0.01590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01623</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01709</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01673</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01500</span></span>
<span class="line"><span>Validation: Loss 0.01452 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01617 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01593</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01321</span></span>
<span class="line"><span>Validation: Loss 0.01334 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01488 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01529</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01377 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01181</span></span>
<span class="line"><span>Validation: Loss 0.01143 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01356</span></span>
<span class="line"><span>Validation: Loss 0.00987 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01128</span></span>
<span class="line"><span>Validation: Loss 0.00906 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01012 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01079</span></span>
<span class="line"><span>Validation: Loss 0.00815 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00836</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62411</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60250</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55918</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53302</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52124</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49859</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48319</span></span>
<span class="line"><span>Validation: Loss 0.47542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46135 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46626</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45230</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43965</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42906</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40814</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39873</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37876</span></span>
<span class="line"><span>Validation: Loss 0.37879 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37277</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35597</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35464</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32969</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30974</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30793</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28532</span></span>
<span class="line"><span>Validation: Loss 0.29386 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28785</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28319</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26330</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23880</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22931</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24417</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22439</span></span>
<span class="line"><span>Validation: Loss 0.22371 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20525 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21265</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20872</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20606</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17968</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17798</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17276</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16496</span></span>
<span class="line"><span>Validation: Loss 0.16736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15021 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15522</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15093</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14203</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14152</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12858</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13211</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12994</span></span>
<span class="line"><span>Validation: Loss 0.12318 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11197</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11679</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10388</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11171</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08685</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08763</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09038</span></span>
<span class="line"><span>Validation: Loss 0.08800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08216</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08446</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07212</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07275</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06664</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06006</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05861</span></span>
<span class="line"><span>Validation: Loss 0.06108 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05411 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05673</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05319</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05019</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05022</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05087</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04814</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03889</span></span>
<span class="line"><span>Validation: Loss 0.04542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04053 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04298</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04091</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03959</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03840</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03912</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03804</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03007</span></span>
<span class="line"><span>Validation: Loss 0.03686 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03375</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03373</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03673</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03162</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03012</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03026</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03130</span></span>
<span class="line"><span>Validation: Loss 0.03139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03066</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02882</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03109</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02646</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02504</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02637</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02725</span></span>
<span class="line"><span>Validation: Loss 0.02736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02426 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02753</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02444</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02363</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02384</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02080</span></span>
<span class="line"><span>Validation: Loss 0.02423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02270</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02180</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02108</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01711</span></span>
<span class="line"><span>Validation: Loss 0.02173 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02023</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01981</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01799</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01803</span></span>
<span class="line"><span>Validation: Loss 0.01965 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01568</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01723</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01790</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01833</span></span>
<span class="line"><span>Validation: Loss 0.01790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01569 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01546</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01555</span></span>
<span class="line"><span>Validation: Loss 0.01638 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01265</span></span>
<span class="line"><span>Validation: Loss 0.01508 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01315 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01383</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01218</span></span>
<span class="line"><span>Validation: Loss 0.01394 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01215 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Validation: Loss 0.01292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01124 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01001</span></span>
<span class="line"><span>Validation: Loss 0.01192 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Validation: Loss 0.01086 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00887</span></span>
<span class="line"><span>Validation: Loss 0.00966 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00850</span></span>
<span class="line"><span>Validation: Loss 0.00857 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00754 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00692</span></span>
<span class="line"><span>Validation: Loss 0.00779 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00688 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",h]]);export{r as __pageData,d as default};