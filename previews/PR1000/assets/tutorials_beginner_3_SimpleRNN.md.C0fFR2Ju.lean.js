import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DjZDIZsN.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63034</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60553</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56538</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54630</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50381</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50074</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47526</span></span>
<span class="line"><span>Validation: Loss 0.46276 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47391 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47023</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44691</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43783</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43551</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41303</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39960</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38515</span></span>
<span class="line"><span>Validation: Loss 0.36395 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37779 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37114</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34915</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36132</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33572</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31298</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31052</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30406</span></span>
<span class="line"><span>Validation: Loss 0.27789 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29328 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28022</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27531</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26726</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26451</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23373</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24550</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21351</span></span>
<span class="line"><span>Validation: Loss 0.20780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22336 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22156</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20017</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19203</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20301</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18432</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16845</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18846</span></span>
<span class="line"><span>Validation: Loss 0.15293 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16724 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16454</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15258</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14479</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13681</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13567</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12978</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14006</span></span>
<span class="line"><span>Validation: Loss 0.11109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12029</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10976</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11662</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09244</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10328</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08922</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08470</span></span>
<span class="line"><span>Validation: Loss 0.07890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08760 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07588</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07787</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07559</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07404</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07644</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06436</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05506</span></span>
<span class="line"><span>Validation: Loss 0.05515 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05574</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05409</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05341</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05125</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05115</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04846</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04370</span></span>
<span class="line"><span>Validation: Loss 0.04143 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04557 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04284</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04111</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03779</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03960</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04298</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03869</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03524</span></span>
<span class="line"><span>Validation: Loss 0.03367 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03339</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03459</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03342</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03407</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03388</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03202</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02971</span></span>
<span class="line"><span>Validation: Loss 0.02860 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03154 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02859</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02993</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02963</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02608</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03050</span></span>
<span class="line"><span>Validation: Loss 0.02489 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02750 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02694</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02671</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02622</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02449</span></span>
<span class="line"><span>Validation: Loss 0.02201 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02435 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02324</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02391</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02036</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02264</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02342</span></span>
<span class="line"><span>Validation: Loss 0.01970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02183 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02103</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02069</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02057</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01919</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01962</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02118</span></span>
<span class="line"><span>Validation: Loss 0.01777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01973 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01873</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01905</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01646</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01732</span></span>
<span class="line"><span>Validation: Loss 0.01614 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01795 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01722</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01571</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01460</span></span>
<span class="line"><span>Validation: Loss 0.01474 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01682</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01390</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01172</span></span>
<span class="line"><span>Validation: Loss 0.01352 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01510 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01519</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01383</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01076</span></span>
<span class="line"><span>Validation: Loss 0.01246 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01394 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01428</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01040</span></span>
<span class="line"><span>Validation: Loss 0.01150 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01288 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00776</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01114</span></span>
<span class="line"><span>Validation: Loss 0.00957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01072 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00997</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00712</span></span>
<span class="line"><span>Validation: Loss 0.00850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00849 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00833</span></span>
<span class="line"><span>Validation: Loss 0.00700 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62243</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61271</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55580</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53213</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52075</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49787</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47505</span></span>
<span class="line"><span>Validation: Loss 0.47777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46300 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46806</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44811</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44868</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42514</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41207</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38999</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40689</span></span>
<span class="line"><span>Validation: Loss 0.38313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36497 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37506</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35458</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34745</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32494</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31280</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32159</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31559</span></span>
<span class="line"><span>Validation: Loss 0.29940 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27966 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27458</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28234</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26623</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24585</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24220</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25250</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22705</span></span>
<span class="line"><span>Validation: Loss 0.22949 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20991 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20868</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20777</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21327</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18019</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18150</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17559</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20257</span></span>
<span class="line"><span>Validation: Loss 0.17268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15497 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15827</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15591</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14491</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13292</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14914</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12443</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13808</span></span>
<span class="line"><span>Validation: Loss 0.12726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11293 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12103</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11242</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10596</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09875</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09495</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09276</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10884</span></span>
<span class="line"><span>Validation: Loss 0.09070 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08028 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09406</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08342</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07543</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06123</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06837</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06419</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05732</span></span>
<span class="line"><span>Validation: Loss 0.06270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05821</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05601</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05212</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05552</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04673</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04686</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03952</span></span>
<span class="line"><span>Validation: Loss 0.04691 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04198 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04253</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04115</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04191</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03686</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03965</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03798</span></span>
<span class="line"><span>Validation: Loss 0.03820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03413 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03729</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03196</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03405</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03641</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03278</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02875</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03003</span></span>
<span class="line"><span>Validation: Loss 0.03254 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02899 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03042</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02825</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02816</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03091</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02586</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02826</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02898</span></span>
<span class="line"><span>Validation: Loss 0.02840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02510</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02558</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02468</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02516</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02303</span></span>
<span class="line"><span>Validation: Loss 0.02517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02233 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02156</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02310</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02294</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02135</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02244</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02294</span></span>
<span class="line"><span>Validation: Loss 0.02259 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01999 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01967</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02107</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01948</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02159</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02296</span></span>
<span class="line"><span>Validation: Loss 0.02044 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01805 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01883</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01874</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01866</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01752</span></span>
<span class="line"><span>Validation: Loss 0.01859 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01638 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01687</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01723</span></span>
<span class="line"><span>Validation: Loss 0.01700 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01495 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01479</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01408</span></span>
<span class="line"><span>Validation: Loss 0.01562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01348</span></span>
<span class="line"><span>Validation: Loss 0.01438 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01260 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01086</span></span>
<span class="line"><span>Validation: Loss 0.01319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01080</span></span>
<span class="line"><span>Validation: Loss 0.01194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01047 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00883</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00887</span></span>
<span class="line"><span>Validation: Loss 0.00937 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00775</span></span>
<span class="line"><span>Validation: Loss 0.00852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00755 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00701 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
