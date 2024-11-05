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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62882</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59277</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57225</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53251</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50229</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50188</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50507</span></span>
<span class="line"><span>Validation: Loss 0.47853 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46911</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44729</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44189</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41557</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42245</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40179</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37552</span></span>
<span class="line"><span>Validation: Loss 0.38442 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36719 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36709</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35896</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34375</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33866</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32693</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30125</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31062</span></span>
<span class="line"><span>Validation: Loss 0.30134 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28408</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28016</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25523</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25847</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24381</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23695</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24566</span></span>
<span class="line"><span>Validation: Loss 0.23166 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21024</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21564</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20701</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18890</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18090</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16907</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18680</span></span>
<span class="line"><span>Validation: Loss 0.17471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15648</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15988</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15562</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13284</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14282</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12854</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10278</span></span>
<span class="line"><span>Validation: Loss 0.12917 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11552 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12839</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10801</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11081</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10765</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09260</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09265</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06954</span></span>
<span class="line"><span>Validation: Loss 0.09268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08263 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08558</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07884</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07281</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07558</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06900</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06710</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05843</span></span>
<span class="line"><span>Validation: Loss 0.06452 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06327</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05694</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05676</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04561</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05024</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04497</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04777</span></span>
<span class="line"><span>Validation: Loss 0.04804 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04318 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04301</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04122</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04065</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03980</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04101</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03758</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04240</span></span>
<span class="line"><span>Validation: Loss 0.03906 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03507 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03654</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03592</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03475</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03660</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03023</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02871</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03144</span></span>
<span class="line"><span>Validation: Loss 0.03320 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03102</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03173</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02765</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02761</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02655</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02823</span></span>
<span class="line"><span>Validation: Loss 0.02894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02589 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02750</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02342</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02557</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02534</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02537</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02405</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02677</span></span>
<span class="line"><span>Validation: Loss 0.02564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02290 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02260</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02378</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02067</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02303</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02156</span></span>
<span class="line"><span>Validation: Loss 0.02299 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02081</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02207</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01970</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02063</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01943</span></span>
<span class="line"><span>Validation: Loss 0.02078 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01849 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01897</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01889</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01701</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01832</span></span>
<span class="line"><span>Validation: Loss 0.01892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01811</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01419</span></span>
<span class="line"><span>Validation: Loss 0.01732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01534 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01732</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01479</span></span>
<span class="line"><span>Validation: Loss 0.01594 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01444</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01310</span></span>
<span class="line"><span>Validation: Loss 0.01474 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01302 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01107</span></span>
<span class="line"><span>Validation: Loss 0.01364 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01204 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01119</span></span>
<span class="line"><span>Validation: Loss 0.01260 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01112 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.01148 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01014 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00841</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00716</span></span>
<span class="line"><span>Validation: Loss 0.00903 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Validation: Loss 0.00820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62838</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58418</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56192</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53274</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51631</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50452</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50991</span></span>
<span class="line"><span>Validation: Loss 0.47583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47713</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45155</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44397</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41821</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40725</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39733</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40230</span></span>
<span class="line"><span>Validation: Loss 0.38098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36048</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36073</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34257</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33288</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33311</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31300</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31612</span></span>
<span class="line"><span>Validation: Loss 0.29758 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28606 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29020</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29033</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25595</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25890</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25033</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22638</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22241</span></span>
<span class="line"><span>Validation: Loss 0.22775 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21623 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21796</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20916</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21166</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19219</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18036</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17794</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14330</span></span>
<span class="line"><span>Validation: Loss 0.17132 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16066 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16874</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14619</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14839</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13905</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14039</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11206</span></span>
<span class="line"><span>Validation: Loss 0.12713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11825 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10855</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11420</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11633</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10222</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09588</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10025</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10460</span></span>
<span class="line"><span>Validation: Loss 0.09154 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08492 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08522</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07215</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07973</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07433</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07438</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06809</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06329</span></span>
<span class="line"><span>Validation: Loss 0.06347 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06186</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05738</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04929</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04957</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05149</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05069</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04632</span></span>
<span class="line"><span>Validation: Loss 0.04707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04399 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04784</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04364</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04188</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03879</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03762</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03797</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03341</span></span>
<span class="line"><span>Validation: Loss 0.03814 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03564 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03561</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03473</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03599</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03489</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03118</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03047</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03367</span></span>
<span class="line"><span>Validation: Loss 0.03245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03134</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02805</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03062</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02663</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03094</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03031</span></span>
<span class="line"><span>Validation: Loss 0.02828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02636 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02707</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02657</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02657</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02492</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02655</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01971</span></span>
<span class="line"><span>Validation: Loss 0.02503 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02377</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02102</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02133</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02156</span></span>
<span class="line"><span>Validation: Loss 0.02244 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02087 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02066</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02072</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01916</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01963</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02012</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02022</span></span>
<span class="line"><span>Validation: Loss 0.02029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01881</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01892</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01799</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01827</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02045</span></span>
<span class="line"><span>Validation: Loss 0.01847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01712 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01704</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01723</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01701</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01617</span></span>
<span class="line"><span>Validation: Loss 0.01691 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01566 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01627</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01385</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01428</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01784</span></span>
<span class="line"><span>Validation: Loss 0.01555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01439 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01204</span></span>
<span class="line"><span>Validation: Loss 0.01435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01057</span></span>
<span class="line"><span>Validation: Loss 0.01327 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01227 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00928</span></span>
<span class="line"><span>Validation: Loss 0.01220 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01129 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00933</span></span>
<span class="line"><span>Validation: Loss 0.01102 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00868</span></span>
<span class="line"><span>Validation: Loss 0.00975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00904 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Validation: Loss 0.00869 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00808 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Validation: Loss 0.00795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
