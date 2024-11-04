import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DjZDIZsN.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<div class="danger custom-block"><p class="custom-block-title">Using older version of Lux.jl</p><p>This tutorial cannot be run on the latest Lux.jl release due to downstream packages not being updated yet.</p></div><h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR1000/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR1000/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR1000/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR1000/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61768</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59930</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56713</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54047</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51597</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50121</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47865</span></span>
<span class="line"><span>Validation: Loss 0.46795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46652 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46117</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45318</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44611</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42978</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40792</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39694</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38315</span></span>
<span class="line"><span>Validation: Loss 0.37014 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36843 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36004</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34925</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34130</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32959</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32650</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32634</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30148</span></span>
<span class="line"><span>Validation: Loss 0.28490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28284 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28177</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28421</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26347</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25438</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24091</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22673</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23449</span></span>
<span class="line"><span>Validation: Loss 0.21496 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21276 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20757</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20766</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19540</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19582</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17628</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18005</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17628</span></span>
<span class="line"><span>Validation: Loss 0.15941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15727 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15721</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15604</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15399</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13853</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13188</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12574</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11378</span></span>
<span class="line"><span>Validation: Loss 0.11670 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11486 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12207</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11456</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09355</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10569</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09949</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09237</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08481</span></span>
<span class="line"><span>Validation: Loss 0.08375 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08375</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08345</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07835</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06816</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06762</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06213</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06980</span></span>
<span class="line"><span>Validation: Loss 0.05841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05652</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05598</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05306</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04933</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05030</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04641</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04750</span></span>
<span class="line"><span>Validation: Loss 0.04302 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04240 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04441</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04124</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04077</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04081</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03709</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03546</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03116</span></span>
<span class="line"><span>Validation: Loss 0.03464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03415 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03606</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03372</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03322</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03393</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03189</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02646</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03294</span></span>
<span class="line"><span>Validation: Loss 0.02935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02888</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03046</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02827</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02635</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02588</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03031</span></span>
<span class="line"><span>Validation: Loss 0.02552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02514 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02553</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02398</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02293</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02138</span></span>
<span class="line"><span>Validation: Loss 0.02254 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02220 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02353</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02219</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02017</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02094</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01831</span></span>
<span class="line"><span>Validation: Loss 0.02016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01985 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01985</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01975</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01895</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01951</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01907</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01916</span></span>
<span class="line"><span>Validation: Loss 0.01821 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01867</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01574</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01371</span></span>
<span class="line"><span>Validation: Loss 0.01654 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01562</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01762</span></span>
<span class="line"><span>Validation: Loss 0.01514 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01489 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01419</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01443</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01570</span></span>
<span class="line"><span>Validation: Loss 0.01393 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01455</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01349</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01385</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01472</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Validation: Loss 0.01288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01265 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01301</span></span>
<span class="line"><span>Validation: Loss 0.01194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01294</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01087 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00997</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00754</span></span>
<span class="line"><span>Validation: Loss 0.01020 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01001 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Validation: Loss 0.00924 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Validation: Loss 0.00822 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62139</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59246</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57027</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53899</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50819</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49876</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47922</span></span>
<span class="line"><span>Validation: Loss 0.46861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.48832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47264</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45301</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43979</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41463</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41048</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39784</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36976</span></span>
<span class="line"><span>Validation: Loss 0.37132 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.39456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35881</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36296</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34028</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33553</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31731</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31076</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29243</span></span>
<span class="line"><span>Validation: Loss 0.28661 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.31255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27947</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27402</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26004</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25728</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23488</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24427</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20789</span></span>
<span class="line"><span>Validation: Loss 0.21665 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.24266 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20118</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21055</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20678</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18776</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17214</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18477</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14291</span></span>
<span class="line"><span>Validation: Loss 0.16093 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18504 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14809</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15519</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15090</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13281</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13164</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13361</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12742</span></span>
<span class="line"><span>Validation: Loss 0.11815 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12001</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11163</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09438</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09725</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09421</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09883</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10509</span></span>
<span class="line"><span>Validation: Loss 0.08456 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08722</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07809</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06755</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07408</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06964</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06397</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05753</span></span>
<span class="line"><span>Validation: Loss 0.05865 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06837 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05850</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05431</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05128</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04614</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05181</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04665</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04332</span></span>
<span class="line"><span>Validation: Loss 0.04329 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04497</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04112</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04005</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03662</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03646</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03716</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03474</span></span>
<span class="line"><span>Validation: Loss 0.03491 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03349</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03320</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03421</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02918</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03241</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03038</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03424</span></span>
<span class="line"><span>Validation: Loss 0.02958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03434 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02883</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02941</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02857</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02777</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02564</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02624</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02442</span></span>
<span class="line"><span>Validation: Loss 0.02570 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02416</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02456</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02412</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02207</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02306</span></span>
<span class="line"><span>Validation: Loss 0.02271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02653 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02232</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02179</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02015</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02059</span></span>
<span class="line"><span>Validation: Loss 0.02031 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02380 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02088</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01656</span></span>
<span class="line"><span>Validation: Loss 0.01832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01740</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01860</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01501</span></span>
<span class="line"><span>Validation: Loss 0.01665 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01966 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01472</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01598</span></span>
<span class="line"><span>Validation: Loss 0.01524 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01804 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01201</span></span>
<span class="line"><span>Validation: Loss 0.01403 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01453</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01261</span></span>
<span class="line"><span>Validation: Loss 0.01298 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01543 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01075</span></span>
<span class="line"><span>Validation: Loss 0.01205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01433 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01215</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00779</span></span>
<span class="line"><span>Validation: Loss 0.01119 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01332 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01004</span></span>
<span class="line"><span>Validation: Loss 0.01036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Validation: Loss 0.00944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01119 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00760</span></span>
<span class="line"><span>Validation: Loss 0.00843 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00765</span></span>
<span class="line"><span>Validation: Loss 0.00755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.6</span></span>
<span class="line"><span>Commit 67dffc4a8ae (2024-10-28 12:23 UTC)</span></span>
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
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.6</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};