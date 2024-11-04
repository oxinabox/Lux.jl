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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61245</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59574</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56853</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53893</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51873</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50439</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49904</span></span>
<span class="line"><span>Validation: Loss 0.47153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45510 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47104</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45086</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43882</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42653</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41193</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40034</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39378</span></span>
<span class="line"><span>Validation: Loss 0.37512 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35644 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36858</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35772</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36217</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33080</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30941</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31916</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28197</span></span>
<span class="line"><span>Validation: Loss 0.29083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28303</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27809</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25987</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26208</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24918</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23885</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21604</span></span>
<span class="line"><span>Validation: Loss 0.22140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20489</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21589</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20495</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18650</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18548</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18020</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18339</span></span>
<span class="line"><span>Validation: Loss 0.16573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14599 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15860</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15216</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14344</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14424</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13450</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13874</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14090</span></span>
<span class="line"><span>Validation: Loss 0.12213 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10587 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10757</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11328</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10824</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10899</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09923</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09813</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09926</span></span>
<span class="line"><span>Validation: Loss 0.08748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08188</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08228</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06927</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08169</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07080</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06378</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06867</span></span>
<span class="line"><span>Validation: Loss 0.06061 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06087</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05464</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05616</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04788</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04819</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04958</span></span>
<span class="line"><span>Validation: Loss 0.04473 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04295</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04212</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03921</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03819</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04163</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03930</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03266</span></span>
<span class="line"><span>Validation: Loss 0.03610 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03169 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03660</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03377</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03319</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03268</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03322</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03070</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02951</span></span>
<span class="line"><span>Validation: Loss 0.03062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02678 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03083</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02996</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02770</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02665</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02773</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02855</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02555</span></span>
<span class="line"><span>Validation: Loss 0.02665 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02323 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02465</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02507</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02675</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02398</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02542</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02416</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02255</span></span>
<span class="line"><span>Validation: Loss 0.02359 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02278</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02355</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02185</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02169</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02158</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01941</span></span>
<span class="line"><span>Validation: Loss 0.02112 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01947</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02060</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01980</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01681</span></span>
<span class="line"><span>Validation: Loss 0.01909 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01647 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01942</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01914</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01904</span></span>
<span class="line"><span>Validation: Loss 0.01737 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01492 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01688</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01621</span></span>
<span class="line"><span>Validation: Loss 0.01588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01361 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01611</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01159</span></span>
<span class="line"><span>Validation: Loss 0.01460 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01248 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01373</span></span>
<span class="line"><span>Validation: Loss 0.01347 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01149 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01171</span></span>
<span class="line"><span>Validation: Loss 0.01242 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01069</span></span>
<span class="line"><span>Validation: Loss 0.01135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00970 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00927</span></span>
<span class="line"><span>Validation: Loss 0.01015 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00871 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00884</span></span>
<span class="line"><span>Validation: Loss 0.00896 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00848</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00700 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00708</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00646 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62875</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59549</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58174</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53679</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51532</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49987</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48033</span></span>
<span class="line"><span>Validation: Loss 0.45841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46761 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46548</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45530</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44421</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43611</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41330</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39804</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37029</span></span>
<span class="line"><span>Validation: Loss 0.35868 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36837</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37575</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33375</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32843</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32180</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31139</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33560</span></span>
<span class="line"><span>Validation: Loss 0.27228 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28406</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27627</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27224</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25705</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25315</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22910</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23095</span></span>
<span class="line"><span>Validation: Loss 0.20288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21522 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22445</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19638</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20739</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18698</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18806</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15291</span></span>
<span class="line"><span>Validation: Loss 0.14862 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15963 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15760</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14790</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15842</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15407</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13757</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12802</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11525</span></span>
<span class="line"><span>Validation: Loss 0.10795 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12742</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10978</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10503</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11486</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09552</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09431</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07964</span></span>
<span class="line"><span>Validation: Loss 0.07712 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08365 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08481</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07543</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06830</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08002</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07040</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07568</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06198</span></span>
<span class="line"><span>Validation: Loss 0.05406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05843 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05972</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05439</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05558</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05364</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04803</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04881</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05167</span></span>
<span class="line"><span>Validation: Loss 0.04031 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04239</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04173</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04284</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04049</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04254</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03814</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03159</span></span>
<span class="line"><span>Validation: Loss 0.03258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03507 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03789</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03375</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03407</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03385</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03384</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03044</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03081</span></span>
<span class="line"><span>Validation: Loss 0.02758 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02909</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03029</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03068</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02934</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02732</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02773</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02766</span></span>
<span class="line"><span>Validation: Loss 0.02396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02588 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02647</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02712</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02510</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02507</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02530</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02423</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02275</span></span>
<span class="line"><span>Validation: Loss 0.02116 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02289 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02425</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02114</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02236</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02337</span></span>
<span class="line"><span>Validation: Loss 0.01892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02156</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02065</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02166</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01917</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01995</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01950</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01835</span></span>
<span class="line"><span>Validation: Loss 0.01707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01853</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01787</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01795</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01770</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Validation: Loss 0.01550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01820</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01704</span></span>
<span class="line"><span>Validation: Loss 0.01414 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01583</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01787</span></span>
<span class="line"><span>Validation: Loss 0.01296 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01411 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01394</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01229</span></span>
<span class="line"><span>Validation: Loss 0.01191 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01299 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01183</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01081</span></span>
<span class="line"><span>Validation: Loss 0.01006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01119</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01140</span></span>
<span class="line"><span>Validation: Loss 0.00908 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00876 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00808</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00736</span></span>
<span class="line"><span>Validation: Loss 0.00666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00719 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
