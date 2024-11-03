import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.Bg5zKH4v.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR940/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR940/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR940/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR940/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62739</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59729</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56977</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53168</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51589</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49878</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47171</span></span>
<span class="line"><span>Validation: Loss 0.46972 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46729</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46062</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44139</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41860</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41018</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39322</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37927</span></span>
<span class="line"><span>Validation: Loss 0.37257 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38266 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36621</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36459</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34688</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32902</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33379</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29399</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27692</span></span>
<span class="line"><span>Validation: Loss 0.28772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29723</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27840</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26495</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24299</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24307</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21976</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24960</span></span>
<span class="line"><span>Validation: Loss 0.21821 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22927 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20889</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22061</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20597</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19263</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16829</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16944</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17455</span></span>
<span class="line"><span>Validation: Loss 0.16281 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17273 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15622</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14744</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14325</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13941</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14234</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13429</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12463</span></span>
<span class="line"><span>Validation: Loss 0.11976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12782 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11612</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11841</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11298</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10589</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08684</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08856</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09550</span></span>
<span class="line"><span>Validation: Loss 0.08556 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08620</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07628</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07569</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07261</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06291</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06394</span></span>
<span class="line"><span>Validation: Loss 0.05936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06326 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05778</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05975</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05161</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05263</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04564</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04617</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04221</span></span>
<span class="line"><span>Validation: Loss 0.04391 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04353</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04029</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04167</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04201</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03730</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03563</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03126</span></span>
<span class="line"><span>Validation: Loss 0.03545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03770 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03459</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03538</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03258</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03346</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03080</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03037</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02944</span></span>
<span class="line"><span>Validation: Loss 0.03009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03205 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02858</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02741</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02804</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02996</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02553</span></span>
<span class="line"><span>Validation: Loss 0.02619 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02506</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02657</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02264</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02373</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02415</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02479</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02583</span></span>
<span class="line"><span>Validation: Loss 0.02316 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02475 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02376</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02519</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02061</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02130</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01983</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01708</span></span>
<span class="line"><span>Validation: Loss 0.02071 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02216 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02033</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02131</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01999</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02011</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01672</span></span>
<span class="line"><span>Validation: Loss 0.01870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01743</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01772</span></span>
<span class="line"><span>Validation: Loss 0.01701 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01826 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01632</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01658</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01922</span></span>
<span class="line"><span>Validation: Loss 0.01556 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01673 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01466</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01566</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01469</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01043</span></span>
<span class="line"><span>Validation: Loss 0.01432 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01274</span></span>
<span class="line"><span>Validation: Loss 0.01325 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01427 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01133</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01326 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01102</span></span>
<span class="line"><span>Validation: Loss 0.01147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00941</span></span>
<span class="line"><span>Validation: Loss 0.01068 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01146</span></span>
<span class="line"><span>Validation: Loss 0.00987 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01061 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00761</span></span>
<span class="line"><span>Validation: Loss 0.00894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00853 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61057</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59270</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58038</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54277</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51030</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49923</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49709</span></span>
<span class="line"><span>Validation: Loss 0.47211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46379</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45744</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43824</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43065</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41740</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38756</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40537</span></span>
<span class="line"><span>Validation: Loss 0.37620 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37093 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37874</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35684</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35337</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32713</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31911</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31187</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29409</span></span>
<span class="line"><span>Validation: Loss 0.29241 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29007</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26297</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25736</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25190</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25424</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25141</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25156</span></span>
<span class="line"><span>Validation: Loss 0.22329 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21779</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19688</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20829</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19688</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19273</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18137</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15038</span></span>
<span class="line"><span>Validation: Loss 0.16768 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16272</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15858</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14480</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14104</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13859</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13907</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12653</span></span>
<span class="line"><span>Validation: Loss 0.12420 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12975</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10471</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11694</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10207</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10454</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09105</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09484</span></span>
<span class="line"><span>Validation: Loss 0.08951 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08678</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08301</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07463</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07512</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07299</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06831</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06626</span></span>
<span class="line"><span>Validation: Loss 0.06224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05981 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06395</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05869</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05352</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05428</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04808</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04659</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04304</span></span>
<span class="line"><span>Validation: Loss 0.04558 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04386 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04379</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04264</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04118</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03955</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03710</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04040</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03925</span></span>
<span class="line"><span>Validation: Loss 0.03667 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03310</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03461</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03594</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03540</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03460</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02761</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03101</span></span>
<span class="line"><span>Validation: Loss 0.03102 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02980 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03052</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02938</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03050</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02916</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02639</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02637</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02582</span></span>
<span class="line"><span>Validation: Loss 0.02696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02493</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02715</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02464</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02687</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02294</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02466</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02023</span></span>
<span class="line"><span>Validation: Loss 0.02384 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02286 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02280</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02168</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02472</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02219</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02067</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02189</span></span>
<span class="line"><span>Validation: Loss 0.02135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02046 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02221</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01973</span></span>
<span class="line"><span>Validation: Loss 0.01928 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01912</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01853</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01774</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01794</span></span>
<span class="line"><span>Validation: Loss 0.01754 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01753</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01795</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01396</span></span>
<span class="line"><span>Validation: Loss 0.01604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01532 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01506</span></span>
<span class="line"><span>Validation: Loss 0.01475 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01408 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Validation: Loss 0.01363 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01300 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01338</span></span>
<span class="line"><span>Validation: Loss 0.01261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01203 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01237</span></span>
<span class="line"><span>Validation: Loss 0.01166 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01112 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01179</span></span>
<span class="line"><span>Validation: Loss 0.01068 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00984</span></span>
<span class="line"><span>Validation: Loss 0.00956 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00913 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00842</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00917</span></span>
<span class="line"><span>Validation: Loss 0.00762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
