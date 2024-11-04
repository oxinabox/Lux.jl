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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61661</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58996</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57687</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55132</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50781</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49596</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48793</span></span>
<span class="line"><span>Validation: Loss 0.46813 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47473 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46617</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45535</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45338</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43396</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40364</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38895</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37596</span></span>
<span class="line"><span>Validation: Loss 0.37124 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36971</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35691</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36144</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33950</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32096</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30152</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27254</span></span>
<span class="line"><span>Validation: Loss 0.28692 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29577 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29549</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28718</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25735</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26152</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24068</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23295</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21167</span></span>
<span class="line"><span>Validation: Loss 0.21792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22688 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21899</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21726</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19443</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18801</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19127</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17910</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16584</span></span>
<span class="line"><span>Validation: Loss 0.16288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17115 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16903</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14695</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15209</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14904</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12526</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13829</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13958</span></span>
<span class="line"><span>Validation: Loss 0.12001 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12684 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11661</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11352</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11137</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10743</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10335</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09421</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09135</span></span>
<span class="line"><span>Validation: Loss 0.08609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08850</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08201</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08410</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07152</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07124</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06383</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05679</span></span>
<span class="line"><span>Validation: Loss 0.05977 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06310 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06126</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06181</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05644</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04954</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04736</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04554</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04745</span></span>
<span class="line"><span>Validation: Loss 0.04406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04638 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04549</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04044</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03979</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03686</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04192</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04034</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03681</span></span>
<span class="line"><span>Validation: Loss 0.03554 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03743 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03734</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03548</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03372</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02902</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03352</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03204</span></span>
<span class="line"><span>Validation: Loss 0.03008 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03171 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02879</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03197</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02859</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02961</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02737</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02500</span></span>
<span class="line"><span>Validation: Loss 0.02613 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02759 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02727</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02495</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02387</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02496</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02155</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02620</span></span>
<span class="line"><span>Validation: Loss 0.02309 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02145</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02394</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02173</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02218</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02192</span></span>
<span class="line"><span>Validation: Loss 0.02066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02186 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02052</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02152</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01965</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01883</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01978</span></span>
<span class="line"><span>Validation: Loss 0.01865 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01934</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01880</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01799</span></span>
<span class="line"><span>Validation: Loss 0.01694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01343</span></span>
<span class="line"><span>Validation: Loss 0.01548 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01644 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01390</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01428</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01567</span></span>
<span class="line"><span>Validation: Loss 0.01423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01396</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01342</span></span>
<span class="line"><span>Validation: Loss 0.01315 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01244</span></span>
<span class="line"><span>Validation: Loss 0.01219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01253</span></span>
<span class="line"><span>Validation: Loss 0.01133 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01206 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00988</span></span>
<span class="line"><span>Validation: Loss 0.01051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01119 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01142</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00720</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00823</span></span>
<span class="line"><span>Validation: Loss 0.00772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63004</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58991</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57065</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53632</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50174</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48842</span></span>
<span class="line"><span>Validation: Loss 0.47399 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46649</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45943</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43846</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43251</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41312</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38961</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38049</span></span>
<span class="line"><span>Validation: Loss 0.37707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37372</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35464</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35244</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33361</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30642</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31109</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31760</span></span>
<span class="line"><span>Validation: Loss 0.29229 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27149 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26959</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28019</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27420</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25994</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23996</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23108</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24025</span></span>
<span class="line"><span>Validation: Loss 0.22234 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20159 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21327</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19822</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20809</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18728</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17842</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17958</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18695</span></span>
<span class="line"><span>Validation: Loss 0.16605 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14716 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15775</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15910</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14891</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14839</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12861</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11971</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13139</span></span>
<span class="line"><span>Validation: Loss 0.12191 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10657 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11436</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11011</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11313</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09972</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09732</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09076</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09734</span></span>
<span class="line"><span>Validation: Loss 0.08729 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09067</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08513</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06780</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07034</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06810</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06339</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06129</span></span>
<span class="line"><span>Validation: Loss 0.06063 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05314 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05538</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05266</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05473</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05193</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04967</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04915</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03914</span></span>
<span class="line"><span>Validation: Loss 0.04495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04394</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04410</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03820</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03733</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03652</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03875</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04063</span></span>
<span class="line"><span>Validation: Loss 0.03641 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03198 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03444</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03662</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03331</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03066</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03077</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02832</span></span>
<span class="line"><span>Validation: Loss 0.03092 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02706 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02845</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03005</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02872</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02770</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02668</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02754</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02816</span></span>
<span class="line"><span>Validation: Loss 0.02695 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02350 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02565</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02713</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02640</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02365</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02200</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02160</span></span>
<span class="line"><span>Validation: Loss 0.02386 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02074 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02191</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02389</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02032</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02153</span></span>
<span class="line"><span>Validation: Loss 0.02139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01854 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01928</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02055</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01739</span></span>
<span class="line"><span>Validation: Loss 0.01934 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01649</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01785</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01898</span></span>
<span class="line"><span>Validation: Loss 0.01762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01516 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01621</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01678</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01687</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01649</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01464</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01682</span></span>
<span class="line"><span>Validation: Loss 0.01612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01383 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01378</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01270 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01369</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01064</span></span>
<span class="line"><span>Validation: Loss 0.01373 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Validation: Loss 0.01276 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01280</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01013 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01062</span></span>
<span class="line"><span>Validation: Loss 0.01100 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00939 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00784</span></span>
<span class="line"><span>Validation: Loss 0.01007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00814</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00752</span></span>
<span class="line"><span>Validation: Loss 0.00803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00695 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
