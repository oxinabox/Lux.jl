import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CCjWn1F9.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61512</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60524</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54823</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54691</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52434</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50498</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49311</span></span>
<span class="line"><span>Validation: Loss 0.46981 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47294</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45229</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44374</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42407</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41602</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40209</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39644</span></span>
<span class="line"><span>Validation: Loss 0.37335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37796</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36535</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34490</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33645</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32186</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31185</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29527</span></span>
<span class="line"><span>Validation: Loss 0.28839 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29135</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28370</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26049</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26873</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24797</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23672</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18915</span></span>
<span class="line"><span>Validation: Loss 0.21823 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20570 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21983</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22068</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20337</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18072</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18741</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18099</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15911</span></span>
<span class="line"><span>Validation: Loss 0.16261 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15102 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16786</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14418</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14681</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15733</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13193</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13234</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13811</span></span>
<span class="line"><span>Validation: Loss 0.11957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11001 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12181</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10795</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10904</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09474</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10197</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10666</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09703</span></span>
<span class="line"><span>Validation: Loss 0.08569 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09038</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08012</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08368</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07743</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06634</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06049</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06252</span></span>
<span class="line"><span>Validation: Loss 0.05944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05485 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05594</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05566</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05580</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05207</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05399</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04829</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03966</span></span>
<span class="line"><span>Validation: Loss 0.04414 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04086 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04453</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04691</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03821</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03967</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04043</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03839</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03051</span></span>
<span class="line"><span>Validation: Loss 0.03575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03306 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03488</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03574</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03342</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03160</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03473</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03210</span></span>
<span class="line"><span>Validation: Loss 0.03038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02803 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02933</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03052</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02711</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03082</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02924</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02724</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02788</span></span>
<span class="line"><span>Validation: Loss 0.02645 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02436 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02505</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02758</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02597</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02615</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02354</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02294</span></span>
<span class="line"><span>Validation: Loss 0.02339 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02151 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02550</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02284</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02288</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02034</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02200</span></span>
<span class="line"><span>Validation: Loss 0.02094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01923 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02073</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02081</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01883</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01974</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02074</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02274</span></span>
<span class="line"><span>Validation: Loss 0.01891 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01733 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01785</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01951</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01925</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01738</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01609</span></span>
<span class="line"><span>Validation: Loss 0.01717 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01727</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01405</span></span>
<span class="line"><span>Validation: Loss 0.01568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01433 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01485</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01515</span></span>
<span class="line"><span>Validation: Loss 0.01440 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01314 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01378</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01444</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01025</span></span>
<span class="line"><span>Validation: Loss 0.01325 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01110 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01264</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01118</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01178</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00896 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00815</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00653</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00729 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62544</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58391</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57736</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53167</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51671</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49884</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48786</span></span>
<span class="line"><span>Validation: Loss 0.47871 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46396 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47308</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45026</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43611</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42725</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40983</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39966</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37452</span></span>
<span class="line"><span>Validation: Loss 0.38291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36610 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37263</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36606</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34123</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33349</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32480</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29689</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28596</span></span>
<span class="line"><span>Validation: Loss 0.29870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29523</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28169</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25398</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25350</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24640</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23055</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.19666</span></span>
<span class="line"><span>Validation: Loss 0.22878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21061 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20867</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21100</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19381</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18173</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19679</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17571</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16454</span></span>
<span class="line"><span>Validation: Loss 0.17258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15561 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15738</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15615</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15424</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13024</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13682</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13057</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11993</span></span>
<span class="line"><span>Validation: Loss 0.12772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12709</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11010</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10527</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10227</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09745</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08796</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09235</span></span>
<span class="line"><span>Validation: Loss 0.09154 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08131 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08486</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07846</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08010</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06830</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07391</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05747</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07270</span></span>
<span class="line"><span>Validation: Loss 0.06335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05660 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05451</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05514</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05569</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05111</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04776</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04815</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04527</span></span>
<span class="line"><span>Validation: Loss 0.04674 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04214</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04455</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04306</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03731</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03861</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03507</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03270</span></span>
<span class="line"><span>Validation: Loss 0.03776 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03550</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03377</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03181</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03216</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03193</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03174</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03156</span></span>
<span class="line"><span>Validation: Loss 0.03209 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02861 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03212</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02960</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02662</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02777</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02676</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02374</span></span>
<span class="line"><span>Validation: Loss 0.02795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02595</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02439</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02377</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02294</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02093</span></span>
<span class="line"><span>Validation: Loss 0.02477 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02195 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02142</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02263</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02096</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02011</span></span>
<span class="line"><span>Validation: Loss 0.02221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01962 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01816</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01997</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02098</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01955</span></span>
<span class="line"><span>Validation: Loss 0.02010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01770 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01766</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01865</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01765</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01711</span></span>
<span class="line"><span>Validation: Loss 0.01829 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01607 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01744</span></span>
<span class="line"><span>Validation: Loss 0.01676 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01468 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01574</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01454</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01437</span></span>
<span class="line"><span>Validation: Loss 0.01542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01349 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01327</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01247 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01258</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01087</span></span>
<span class="line"><span>Validation: Loss 0.01326 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01157 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01236</span></span>
<span class="line"><span>Validation: Loss 0.01233 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01077 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Validation: Loss 0.01143 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00999 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Validation: Loss 0.01047 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00848</span></span>
<span class="line"><span>Validation: Loss 0.00936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00825 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Validation: Loss 0.00831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
