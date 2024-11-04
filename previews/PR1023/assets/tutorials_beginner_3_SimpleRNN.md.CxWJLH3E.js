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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.60926</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60205</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56447</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53935</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51961</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50630</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48399</span></span>
<span class="line"><span>Validation: Loss 0.46956 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47301</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45405</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43968</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43054</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40202</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39666</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40138</span></span>
<span class="line"><span>Validation: Loss 0.37273 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38210 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36731</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36875</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34892</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33812</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31629</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30792</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27809</span></span>
<span class="line"><span>Validation: Loss 0.28817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28662</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27989</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27278</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25235</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23497</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23847</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23192</span></span>
<span class="line"><span>Validation: Loss 0.21844 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21529</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21660</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21147</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18347</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18387</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16418</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18488</span></span>
<span class="line"><span>Validation: Loss 0.16251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15106</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15557</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15604</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12610</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14466</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13525</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13401</span></span>
<span class="line"><span>Validation: Loss 0.11923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11300</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11270</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11182</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10579</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10077</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09092</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08957</span></span>
<span class="line"><span>Validation: Loss 0.08530 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09085 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08321</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07613</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07561</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07250</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06895</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07155</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06246</span></span>
<span class="line"><span>Validation: Loss 0.05935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06304 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06135</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05983</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05429</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04415</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04965</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04801</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04264</span></span>
<span class="line"><span>Validation: Loss 0.04389 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04647 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04243</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04109</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04136</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04201</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03979</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03471</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03760</span></span>
<span class="line"><span>Validation: Loss 0.03546 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03545</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03571</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03202</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03209</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03134</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03114</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03593</span></span>
<span class="line"><span>Validation: Loss 0.03006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03210</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02768</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02955</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02631</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02720</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02667</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03031</span></span>
<span class="line"><span>Validation: Loss 0.02612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02589</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02454</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02579</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02323</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02301</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02099</span></span>
<span class="line"><span>Validation: Loss 0.02307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02452 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02105</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02234</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02259</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02282</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01795</span></span>
<span class="line"><span>Validation: Loss 0.02066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02199 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02140</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02017</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02011</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01963</span></span>
<span class="line"><span>Validation: Loss 0.01866 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01740</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01824</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01976</span></span>
<span class="line"><span>Validation: Loss 0.01696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01745</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01777</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01627</span></span>
<span class="line"><span>Validation: Loss 0.01549 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01656 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01496</span></span>
<span class="line"><span>Validation: Loss 0.01423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01523 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01639</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01405 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01292</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01227</span></span>
<span class="line"><span>Validation: Loss 0.01211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01082</span></span>
<span class="line"><span>Validation: Loss 0.01112 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00940</span></span>
<span class="line"><span>Validation: Loss 0.01002 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01071 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Validation: Loss 0.00888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00947 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00662</span></span>
<span class="line"><span>Validation: Loss 0.00795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62249</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58988</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57122</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54145</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51676</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49941</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48712</span></span>
<span class="line"><span>Validation: Loss 0.46707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46435</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45555</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45454</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42345</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41436</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38527</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37442</span></span>
<span class="line"><span>Validation: Loss 0.36940 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36752</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36360</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34430</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32734</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31783</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31825</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28565</span></span>
<span class="line"><span>Validation: Loss 0.28440 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28307</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27199</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26836</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26051</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24528</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23063</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22536</span></span>
<span class="line"><span>Validation: Loss 0.21475 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21368 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21305</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21531</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19616</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18414</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18294</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17875</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17815</span></span>
<span class="line"><span>Validation: Loss 0.15941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16464</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14669</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14234</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13936</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13121</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11054</span></span>
<span class="line"><span>Validation: Loss 0.11688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11621 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11895</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11755</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11153</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10806</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08931</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08989</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08885</span></span>
<span class="line"><span>Validation: Loss 0.08377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08332 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08392</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07975</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07711</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07462</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06929</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06475</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06222</span></span>
<span class="line"><span>Validation: Loss 0.05835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05808 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05835</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05645</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05303</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04974</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04989</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04836</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04374</span></span>
<span class="line"><span>Validation: Loss 0.04304 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04373</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03963</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04024</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03893</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04085</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03933</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02782</span></span>
<span class="line"><span>Validation: Loss 0.03470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03413</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03603</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03246</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03142</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03040</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03336</span></span>
<span class="line"><span>Validation: Loss 0.02942 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02924 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03113</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02712</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02845</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02904</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02709</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02722</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02449</span></span>
<span class="line"><span>Validation: Loss 0.02555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02730</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02638</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02358</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02337</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02417</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02397</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02159</span></span>
<span class="line"><span>Validation: Loss 0.02258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02377</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02260</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02070</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02060</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02141</span></span>
<span class="line"><span>Validation: Loss 0.02019 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02047</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01953</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01824</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02201</span></span>
<span class="line"><span>Validation: Loss 0.01821 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01872</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01763</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01802</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01691</span></span>
<span class="line"><span>Validation: Loss 0.01653 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01638</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01747</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01570</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01431</span></span>
<span class="line"><span>Validation: Loss 0.01511 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01501 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01520</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01366</span></span>
<span class="line"><span>Validation: Loss 0.01390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01381 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01404</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01277 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01259</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01124</span></span>
<span class="line"><span>Validation: Loss 0.01194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01186 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01311</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01098</span></span>
<span class="line"><span>Validation: Loss 0.01033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00747</span></span>
<span class="line"><span>Validation: Loss 0.00952 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00945 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00884</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00760</span></span>
<span class="line"><span>Validation: Loss 0.00766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00762 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>2 devices:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 15.227 GiB / 16.000 GiB available)</span></span>
<span class="line"><span>  1: Quadro RTX 5000 (sm_75, 15.549 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
