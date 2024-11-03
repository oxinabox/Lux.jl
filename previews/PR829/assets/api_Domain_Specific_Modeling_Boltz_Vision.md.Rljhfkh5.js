import{_ as e,c as t,o as i,a4 as o}from"./chunks/framework.GYfaOXHm.js";const k=JSON.parse('{"title":"Computer Vision Models (Vision API)","description":"","frontmatter":{},"headers":[],"relativePath":"api/Domain_Specific_Modeling/Boltz_Vision.md","filePath":"api/Domain_Specific_Modeling/Boltz_Vision.md","lastUpdated":null}'),s={name:"api/Domain_Specific_Modeling/Boltz_Vision.md"},a=o('<h1 id="Computer-Vision-Models-(Vision-API)" tabindex="-1">Computer Vision Models (<code>Vision</code> API) <a class="header-anchor" href="#Computer-Vision-Models-(Vision-API)" aria-label="Permalink to &quot;Computer Vision Models (`Vision` API) {#Computer-Vision-Models-(Vision-API)}&quot;">​</a></h1><h2 id="Native-Lux-Models" tabindex="-1">Native Lux Models <a class="header-anchor" href="#Native-Lux-Models" aria-label="Permalink to &quot;Native Lux Models {#Native-Lux-Models}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.VGG" href="#Boltz.Vision.VGG">#</a> <b><u>Boltz.Vision.VGG</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(imsize; config, inchannels, batchnorm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nclasses, fcsize, dropout)</span></span></code></pre></div><p>Create a VGG model [1].</p><p><strong>Arguments</strong></p><ul><li><p><code>imsize</code>: input image width and height as a tuple</p></li><li><p><code>config</code>: the configuration for the convolution layers</p></li><li><p><code>inchannels</code>: number of input channels</p></li><li><p><code>batchnorm</code>: set to <code>true</code> to use batch normalization after each convolution</p></li><li><p><code>nclasses</code>: number of output classes</p></li><li><p><code>fcsize</code>: intermediate fully connected layer size</p></li><li><p><code>dropout</code>: dropout level between fully connected layers</p></li></ul><p><strong>References</strong></p><p>[1] Simonyan, Karen, and Andrew Zisserman. &quot;Very deep convolutional networks for large-scale image recognition.&quot; arXiv preprint arXiv:1409.1556 (2014).</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/vgg.jl#L20-L39" target="_blank" rel="noreferrer">source</a></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VGG</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; batchnorm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a VGG model [1] with ImageNet Configuration.</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: the depth of the VGG model. Choices: {<code>11</code>, <code>13</code>, <code>16</code>, <code>19</code>}.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>batchnorm = false</code>: set to <code>true</code> to use batch normalization after each convolution.</p></li><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Simonyan, Karen, and Andrew Zisserman. &quot;Very deep convolutional networks for large-scale image recognition.&quot; arXiv preprint arXiv:1409.1556 (2014).</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/vgg.jl#L62-L80" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.VisionTransformer" href="#Boltz.Vision.VisionTransformer">#</a> <b><u>Boltz.Vision.VisionTransformer</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">VisionTransformer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Creates a Vision Transformer model with the specified configuration.</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: name of the Vision Transformer model to create. The following models are available:</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/vit.jl#L48-L61" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Imported-from-Metalhead.jl" tabindex="-1">Imported from Metalhead.jl <a class="header-anchor" href="#Imported-from-Metalhead.jl" aria-label="Permalink to &quot;Imported from Metalhead.jl {#Imported-from-Metalhead.jl}&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>You need to load <code>Flux</code> and <code>Metalhead</code> before using these models.</p></div><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.AlexNet" href="#Boltz.Vision.AlexNet">#</a> <b><u>Boltz.Vision.AlexNet</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AlexNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create an AlexNet model [1]</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. &quot;Imagenet classification with deep convolutional neural networks.&quot; Advances in neural information processing systems 25 (2012): 1097-1105.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L1-L15" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.ConvMixer" href="#Boltz.Vision.ConvMixer">#</a> <b><u>Boltz.Vision.ConvMixer</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ConvMixer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ConvMixer model [1].</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: The name of the ConvMixer model. Must be one of <code>:base</code>, <code>:small</code>, or <code>:large</code>.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Zhu, Zhuoyuan, et al. &quot;ConvMixer: A Convolutional Neural Network with Faster Depth-wise Convolutions for Computer Vision.&quot; arXiv preprint arXiv:1911.11907 (2019).</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L124-L142" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.DenseNet" href="#Boltz.Vision.DenseNet">#</a> <b><u>Boltz.Vision.DenseNet</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DenseNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a DenseNet model [1].</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the DenseNet model. Must be one of 121, 161, 169, or 201.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. &quot;Densely connected convolutional networks.&quot; Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L77-L95" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.GoogLeNet" href="#Boltz.Vision.GoogLeNet">#</a> <b><u>Boltz.Vision.GoogLeNet</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">GoogLeNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a GoogLeNet model [1].</p><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. &quot;Going deeper with convolutions.&quot; Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L59-L74" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.MobileNet" href="#Boltz.Vision.MobileNet">#</a> <b><u>Boltz.Vision.MobileNet</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">MobileNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Symbol</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a MobileNet model [1, 2, 3].</p><p><strong>Arguments</strong></p><ul><li><code>name::Symbol</code>: The name of the MobileNet model. Must be one of <code>:v1</code>, <code>:v2</code>, <code>:v3_small</code>, or <code>:v3_large</code>.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Howard, Andrew G., et al. &quot;Mobilenets: Efficient convolutional neural networks for mobile vision applications.&quot; arXiv preprint arXiv:1704.04861 (2017). [2] Sandler, Mark, et al. &quot;Mobilenetv2: Inverted residuals and linear bottlenecks.&quot; Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. [3] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. &quot;Searching for MobileNetV3.&quot; arXiv preprint arXiv:1905.02244. 2019.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L98-L121" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.ResNet" href="#Boltz.Vision.ResNet">#</a> <b><u>Boltz.Vision.ResNet</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ResNet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ResNet model [1].</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the ResNet model. Must be one of 18, 34, 50, 101, or 152.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] He, Kaiming, et al. &quot;Deep residual learning for image recognition.&quot; Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L18-L35" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Boltz.Vision.ResNeXt" href="#Boltz.Vision.ResNeXt">#</a> <b><u>Boltz.Vision.ResNeXt</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ResNeXt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Create a ResNeXt model [1].</p><p><strong>Arguments</strong></p><ul><li><code>depth::Int</code>: The depth of the ResNeXt model. Must be one of 50, 101, or 152.</li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>pretrained::Bool=false</code>: If <code>true</code>, returns a pretrained model.</p></li><li><p><code>rng::Union{Nothing, AbstractRNG}=nothing</code>: Random number generator.</p></li><li><p><code>seed::Int=0</code>: Random seed.</p></li><li><p><code>initialized::Val{Bool}=Val(true)</code>: If <code>Val(true)</code>, returns <code>(model, parameters, states)</code>, otherwise just <code>model</code>.</p></li></ul><p><strong>References</strong></p><p>[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He, Ross Gorshick, and Piotr Dollár. &quot;Aggregated residual transformations for deep neural networks.&quot; Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.</p><p><a href="https://github.com/LuxDL/Boltz.jl/blob/v0.3.11/src/vision/extensions.jl#L38-L56" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Pretrained-Models" tabindex="-1">Pretrained Models <a class="header-anchor" href="#Pretrained-Models" aria-label="Permalink to &quot;Pretrained Models {#Pretrained-Models}&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">Tip</p><p>Pass <code>pretrained=true</code> to the model constructor to load the pretrained weights.</p></div><table tabindex="0"><thead><tr><th style="text-align:left;">MODEL</th><th style="text-align:center;">TOP 1 ACCURACY (%)</th><th style="text-align:center;">TOP 5 ACCURACY (%)</th></tr></thead><tbody><tr><td style="text-align:left;"><code>AlexNet()</code></td><td style="text-align:center;">54.48</td><td style="text-align:center;">77.72</td></tr><tr><td style="text-align:left;"><code>VGG(11)</code></td><td style="text-align:center;">67.35</td><td style="text-align:center;">87.91</td></tr><tr><td style="text-align:left;"><code>VGG(13)</code></td><td style="text-align:center;">68.40</td><td style="text-align:center;">88.48</td></tr><tr><td style="text-align:left;"><code>VGG(16)</code></td><td style="text-align:center;">70.24</td><td style="text-align:center;">89.80</td></tr><tr><td style="text-align:left;"><code>VGG(19)</code></td><td style="text-align:center;">71.09</td><td style="text-align:center;">90.27</td></tr><tr><td style="text-align:left;"><code>VGG(11; batchnorm=true)</code></td><td style="text-align:center;">69.09</td><td style="text-align:center;">88.94</td></tr><tr><td style="text-align:left;"><code>VGG(13; batchnorm=true)</code></td><td style="text-align:center;">69.66</td><td style="text-align:center;">89.49</td></tr><tr><td style="text-align:left;"><code>VGG(16; batchnorm=true)</code></td><td style="text-align:center;">72.11</td><td style="text-align:center;">91.02</td></tr><tr><td style="text-align:left;"><code>VGG(19; batchnorm=true)</code></td><td style="text-align:center;">72.95</td><td style="text-align:center;">91.32</td></tr></tbody></table><h3 id="preprocessing" tabindex="-1">Preprocessing <a class="header-anchor" href="#preprocessing" aria-label="Permalink to &quot;Preprocessing&quot;">​</a></h3><p>All the pretrained models require that the images be normalized with the parameters <code>mean = [0.485f0, 0.456f0, 0.406f0]</code> and <code>std = [0.229f0, 0.224f0, 0.225f0]</code>.</p>',27),n=[a];function r(l,d,p,c,h,u){return i(),t("div",null,n)}const m=e(s,[["render",r]]);export{k as __pageData,m as default};