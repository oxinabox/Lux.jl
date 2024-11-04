import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DFwXuivk.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR1023/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR1023/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR1023/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR1023/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.64057</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59979</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57469</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52634</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51621</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49389</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45583</span></span>
<span class="line"><span>Validation: Loss 0.47458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47304</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44588</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43371</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42941</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40337</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40935</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41637</span></span>
<span class="line"><span>Validation: Loss 0.38048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37277 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37248</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35802</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33966</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34392</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32234</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31225</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30096</span></span>
<span class="line"><span>Validation: Loss 0.29702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28867 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29173</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27731</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27218</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24714</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25091</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23962</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21108</span></span>
<span class="line"><span>Validation: Loss 0.22730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22028</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20350</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19989</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19331</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19683</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17503</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16721</span></span>
<span class="line"><span>Validation: Loss 0.17114 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17147</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16302</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15212</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14303</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12738</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12871</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11721</span></span>
<span class="line"><span>Validation: Loss 0.12619 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12016 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11949</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12440</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10460</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10917</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10007</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08493</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09085</span></span>
<span class="line"><span>Validation: Loss 0.08996 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08013</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08090</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07609</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07364</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06906</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06931</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06541</span></span>
<span class="line"><span>Validation: Loss 0.06251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05100</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05607</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05368</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05426</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05340</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05104</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04335</span></span>
<span class="line"><span>Validation: Loss 0.04699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04493 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04503</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04124</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04448</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03911</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03938</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03829</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04026</span></span>
<span class="line"><span>Validation: Loss 0.03826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03658 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03930</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03604</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03612</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03189</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02936</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03284</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03300</span></span>
<span class="line"><span>Validation: Loss 0.03255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03256</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03052</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02825</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02876</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02636</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02841</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03380</span></span>
<span class="line"><span>Validation: Loss 0.02836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02709 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02411</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02593</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02645</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02552</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02444</span></span>
<span class="line"><span>Validation: Loss 0.02512 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02397 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02333</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02361</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02410</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02181</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02209</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02280</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02016</span></span>
<span class="line"><span>Validation: Loss 0.02251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02147 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02232</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01855</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02195</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01981</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01893</span></span>
<span class="line"><span>Validation: Loss 0.02036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01942 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01769</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01855</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01861</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01863</span></span>
<span class="line"><span>Validation: Loss 0.01855 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01695</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01848</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01627</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01834</span></span>
<span class="line"><span>Validation: Loss 0.01698 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01616 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01444</span></span>
<span class="line"><span>Validation: Loss 0.01560 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01442</span></span>
<span class="line"><span>Validation: Loss 0.01437 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01366 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01271</span></span>
<span class="line"><span>Validation: Loss 0.01324 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01258 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01164</span></span>
<span class="line"><span>Validation: Loss 0.01211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00924</span></span>
<span class="line"><span>Validation: Loss 0.01086 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01033 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00970</span></span>
<span class="line"><span>Validation: Loss 0.00960 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00741</span></span>
<span class="line"><span>Validation: Loss 0.00863 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00682</span></span>
<span class="line"><span>Validation: Loss 0.00793 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00757 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62690</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60657</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56281</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52881</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52633</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50039</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49069</span></span>
<span class="line"><span>Validation: Loss 0.45938 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46501 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47462</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45191</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43792</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43606</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41205</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39584</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38915</span></span>
<span class="line"><span>Validation: Loss 0.36038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36693 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37153</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36424</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34246</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33133</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33333</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30858</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29091</span></span>
<span class="line"><span>Validation: Loss 0.27377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29195</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28537</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26650</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26054</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23420</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22955</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23565</span></span>
<span class="line"><span>Validation: Loss 0.20360 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21095 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21585</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21214</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21145</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18220</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18090</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17336</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19317</span></span>
<span class="line"><span>Validation: Loss 0.14910 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15588 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15395</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15698</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15166</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13232</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14346</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13854</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11847</span></span>
<span class="line"><span>Validation: Loss 0.10827 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11389 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11237</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10993</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11363</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10598</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09643</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10049</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08523</span></span>
<span class="line"><span>Validation: Loss 0.07725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08140 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08910</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07531</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07807</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07407</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07255</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06424</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05728</span></span>
<span class="line"><span>Validation: Loss 0.05401 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05502</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05847</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05411</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05183</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05047</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04813</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04674</span></span>
<span class="line"><span>Validation: Loss 0.04039 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04477</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04242</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03914</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03948</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03582</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04193</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04221</span></span>
<span class="line"><span>Validation: Loss 0.03269 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03796</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03443</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03414</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03466</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02958</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03211</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02960</span></span>
<span class="line"><span>Validation: Loss 0.02769 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03248</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02984</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02850</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02643</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02822</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02842</span></span>
<span class="line"><span>Validation: Loss 0.02405 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02528 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02718</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02621</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02660</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02303</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02262</span></span>
<span class="line"><span>Validation: Loss 0.02125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02449</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02408</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02080</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02202</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02035</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02095</span></span>
<span class="line"><span>Validation: Loss 0.01900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01936</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02181</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02222</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01930</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01867</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02032</span></span>
<span class="line"><span>Validation: Loss 0.01713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01875</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01975</span></span>
<span class="line"><span>Validation: Loss 0.01555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01641 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01607</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01628</span></span>
<span class="line"><span>Validation: Loss 0.01418 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01498 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01404</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01550</span></span>
<span class="line"><span>Validation: Loss 0.01299 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01374 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01581</span></span>
<span class="line"><span>Validation: Loss 0.01195 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01264 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01123</span></span>
<span class="line"><span>Validation: Loss 0.01098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01162 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01021</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01126</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00846</span></span>
<span class="line"><span>Validation: Loss 0.00894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00943 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00911</span></span>
<span class="line"><span>Validation: Loss 0.00795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00837 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00797</span></span>
<span class="line"><span>Validation: Loss 0.00721 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00759 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Validation: Loss 0.00667 Accuracy 1.00000</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};