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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62375</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60243</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56139</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54222</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51561</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49499</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48444</span></span>
<span class="line"><span>Validation: Loss 0.47445 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45347</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45552</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44177</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42541</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40849</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40816</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40472</span></span>
<span class="line"><span>Validation: Loss 0.37881 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36074 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38014</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35794</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34824</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31906</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32618</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30117</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31319</span></span>
<span class="line"><span>Validation: Loss 0.29458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27493 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27974</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28550</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26204</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24744</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24158</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24136</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23088</span></span>
<span class="line"><span>Validation: Loss 0.22467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20467 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22525</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21606</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18593</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18954</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18803</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17077</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14730</span></span>
<span class="line"><span>Validation: Loss 0.16831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14972 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16703</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15578</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14849</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13886</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13355</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12752</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10219</span></span>
<span class="line"><span>Validation: Loss 0.12425 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12176</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10564</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10274</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10203</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10603</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09253</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08853</span></span>
<span class="line"><span>Validation: Loss 0.08950 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07776 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08306</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08199</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07649</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07017</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06790</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05572</span></span>
<span class="line"><span>Validation: Loss 0.06228 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05618</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05743</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05183</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04835</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04556</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04078</span></span>
<span class="line"><span>Validation: Loss 0.04612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04061 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04457</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04024</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04004</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04045</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04056</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03817</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03148</span></span>
<span class="line"><span>Validation: Loss 0.03736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03159</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03612</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03557</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03442</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03144</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03180</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02871</span></span>
<span class="line"><span>Validation: Loss 0.03179 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02787 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03151</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02803</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02897</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03055</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02706</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02962</span></span>
<span class="line"><span>Validation: Loss 0.02771 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02423 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02489</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02559</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02446</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02557</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02470</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02556</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02281</span></span>
<span class="line"><span>Validation: Loss 0.02453 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02140 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02242</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02371</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02315</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02205</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02270</span></span>
<span class="line"><span>Validation: Loss 0.02197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01912 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02002</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01942</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02006</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02119</span></span>
<span class="line"><span>Validation: Loss 0.01985 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01754</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01913</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01716</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01869</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01666</span></span>
<span class="line"><span>Validation: Loss 0.01804 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01562 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01705</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01593</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Validation: Loss 0.01651 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01425 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01491</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01562</span></span>
<span class="line"><span>Validation: Loss 0.01518 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01018</span></span>
<span class="line"><span>Validation: Loss 0.01401 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01205 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01287</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01114 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01415</span></span>
<span class="line"><span>Validation: Loss 0.01196 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01028 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01038</span></span>
<span class="line"><span>Validation: Loss 0.01084 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01029</span></span>
<span class="line"><span>Validation: Loss 0.00962 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00833 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00817</span></span>
<span class="line"><span>Validation: Loss 0.00854 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00759</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00801</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00680 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61012</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60239</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56276</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54745</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52699</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50096</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48785</span></span>
<span class="line"><span>Validation: Loss 0.45828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46227</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44769</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44552</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43040</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42415</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40605</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37192</span></span>
<span class="line"><span>Validation: Loss 0.35975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36742 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37162</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36109</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35089</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33516</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31777</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31629</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31397</span></span>
<span class="line"><span>Validation: Loss 0.27395 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28241 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29087</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27282</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27248</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26226</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24697</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23546</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23024</span></span>
<span class="line"><span>Validation: Loss 0.20426 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21782</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21603</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21951</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19575</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17581</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17342</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15390</span></span>
<span class="line"><span>Validation: Loss 0.14968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15749 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17385</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15643</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14011</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14229</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12923</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14315</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13237</span></span>
<span class="line"><span>Validation: Loss 0.10889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11544 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11611</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11673</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11049</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10653</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09516</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10229</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09648</span></span>
<span class="line"><span>Validation: Loss 0.07809 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08300 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08119</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08878</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07888</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07669</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07469</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06423</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05633</span></span>
<span class="line"><span>Validation: Loss 0.05467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05798 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05985</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05778</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05870</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05372</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04815</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04624</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04707</span></span>
<span class="line"><span>Validation: Loss 0.04036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04266 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04458</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04204</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04061</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03906</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03945</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04220</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03044</span></span>
<span class="line"><span>Validation: Loss 0.03248 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03434 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03767</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03362</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03057</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03384</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03284</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03570</span></span>
<span class="line"><span>Validation: Loss 0.02745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03194</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02981</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02896</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02718</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02708</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02884</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02567</span></span>
<span class="line"><span>Validation: Loss 0.02380 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02523 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02535</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02598</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02617</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02655</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02420</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02398</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02172</span></span>
<span class="line"><span>Validation: Loss 0.02099 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02229 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02380</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02519</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02419</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02172</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01985</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01971</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02308</span></span>
<span class="line"><span>Validation: Loss 0.01875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02044</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02139</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02057</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01981</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01678</span></span>
<span class="line"><span>Validation: Loss 0.01689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01798 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01849</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01799</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01711</span></span>
<span class="line"><span>Validation: Loss 0.01532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01643</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01799</span></span>
<span class="line"><span>Validation: Loss 0.01397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01586</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01674</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01543</span></span>
<span class="line"><span>Validation: Loss 0.01280 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01368 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01289</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01380</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01471</span></span>
<span class="line"><span>Validation: Loss 0.01177 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01258 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01496</span></span>
<span class="line"><span>Validation: Loss 0.01081 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01156 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01273</span></span>
<span class="line"><span>Validation: Loss 0.00984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01023</span></span>
<span class="line"><span>Validation: Loss 0.00878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00935 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00840</span></span>
<span class="line"><span>Validation: Loss 0.00779 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00841</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00750 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00673</span></span>
<span class="line"><span>Validation: Loss 0.00655 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00694 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
