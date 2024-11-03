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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63922</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59006</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56046</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53851</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51878</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50837</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47988</span></span>
<span class="line"><span>Validation: Loss 0.46056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47123</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44911</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44093</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43190</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41116</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39899</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42438</span></span>
<span class="line"><span>Validation: Loss 0.36182 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37078 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37969</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36815</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34446</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34307</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31119</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30866</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29978</span></span>
<span class="line"><span>Validation: Loss 0.27641 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28617 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29006</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27774</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26425</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26136</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25087</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23835</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21375</span></span>
<span class="line"><span>Validation: Loss 0.20609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20882</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22010</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20382</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19422</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18539</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17881</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16316</span></span>
<span class="line"><span>Validation: Loss 0.15119 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16827</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15449</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15191</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13655</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13572</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13289</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13403</span></span>
<span class="line"><span>Validation: Loss 0.10993 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11702 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11842</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12253</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09597</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11399</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09454</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09874</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08202</span></span>
<span class="line"><span>Validation: Loss 0.07828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08202</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08031</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08125</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07767</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07224</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05991</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06217</span></span>
<span class="line"><span>Validation: Loss 0.05471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06353</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05217</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05646</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05128</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05138</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04732</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04093</span></span>
<span class="line"><span>Validation: Loss 0.04107 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04349 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04294</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04586</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04159</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04004</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03786</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03902</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03803</span></span>
<span class="line"><span>Validation: Loss 0.03334 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03532 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03516</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03884</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03437</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03213</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03224</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03263</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02999</span></span>
<span class="line"><span>Validation: Loss 0.02831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02805</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03113</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03118</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03009</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02828</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02706</span></span>
<span class="line"><span>Validation: Loss 0.02463 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02615 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02656</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02839</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02764</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02471</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02318</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02648</span></span>
<span class="line"><span>Validation: Loss 0.02177 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02313 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02412</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02261</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02290</span></span>
<span class="line"><span>Validation: Loss 0.01947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02070 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02160</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02149</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02041</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01964</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01997</span></span>
<span class="line"><span>Validation: Loss 0.01755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01773</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01748</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01898</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01945</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01908</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01547</span></span>
<span class="line"><span>Validation: Loss 0.01594 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01699 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01776</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01820</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01682</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01475</span></span>
<span class="line"><span>Validation: Loss 0.01456 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01553 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01450</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01295</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01358</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01320 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01105</span></span>
<span class="line"><span>Validation: Loss 0.01143 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01117</span></span>
<span class="line"><span>Validation: Loss 0.01057 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01130 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01094</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00936</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01034 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01148</span></span>
<span class="line"><span>Validation: Loss 0.00868 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00925 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Validation: Loss 0.00770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00697</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Validation: Loss 0.00699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00742 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60676</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57950</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54075</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52232</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49256</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48370</span></span>
<span class="line"><span>Validation: Loss 0.45461 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46845</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45738</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43860</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43470</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41848</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39610</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38415</span></span>
<span class="line"><span>Validation: Loss 0.35426 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35825</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35951</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36279</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32950</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31807</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32480</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30269</span></span>
<span class="line"><span>Validation: Loss 0.26718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27659</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28368</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25963</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26311</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25752</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23747</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21984</span></span>
<span class="line"><span>Validation: Loss 0.19750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21342 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21820</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20098</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20661</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19340</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18270</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18410</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18297</span></span>
<span class="line"><span>Validation: Loss 0.14377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16764</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14448</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14964</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13646</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15055</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13269</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13151</span></span>
<span class="line"><span>Validation: Loss 0.10387 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11588 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12214</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12336</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10728</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10316</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09760</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09230</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08866</span></span>
<span class="line"><span>Validation: Loss 0.07395 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08347</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08922</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07872</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07093</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06854</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06745</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05531</span></span>
<span class="line"><span>Validation: Loss 0.05181 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06019</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05877</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05510</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05308</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04923</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04534</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04535</span></span>
<span class="line"><span>Validation: Loss 0.03873 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04256</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04386</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04479</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03830</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03930</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03758</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03751</span></span>
<span class="line"><span>Validation: Loss 0.03130 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03479 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03591</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03614</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03522</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03385</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03280</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03028</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02854</span></span>
<span class="line"><span>Validation: Loss 0.02646 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02951 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03022</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03157</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02827</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02888</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02731</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02753</span></span>
<span class="line"><span>Validation: Loss 0.02297 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02568 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02879</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02518</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02496</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02617</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02511</span></span>
<span class="line"><span>Validation: Loss 0.02027 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02272 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02450</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02156</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02300</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01950</span></span>
<span class="line"><span>Validation: Loss 0.01812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02111</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01972</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02013</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01899</span></span>
<span class="line"><span>Validation: Loss 0.01635 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01711</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01998</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01763</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01537</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01674 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01888</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01708</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01637</span></span>
<span class="line"><span>Validation: Loss 0.01354 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01531 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01553</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01595</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01557</span></span>
<span class="line"><span>Validation: Loss 0.01239 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01405 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01480</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01330</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01456</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01329</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01284</span></span>
<span class="line"><span>Validation: Loss 0.00947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01075 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01027</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00955 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.00751 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00787</span></span>
<span class="line"><span>Validation: Loss 0.00682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Validation: Loss 0.00631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
