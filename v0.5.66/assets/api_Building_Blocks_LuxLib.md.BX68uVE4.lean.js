import{_ as n,c as e,j as t,a as i,a4 as s,o as a}from"./chunks/framework.yM8ZEq0R.js";const Zi=JSON.parse('{"title":"LuxLib","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/LuxLib.md","filePath":"api/Building_Blocks/LuxLib.md","lastUpdated":null}'),l={name:"api/Building_Blocks/LuxLib.md"},o=s("",29),r={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},d=t("a",{id:"LuxLib.API.batchnorm",href:"#LuxLib.API.batchnorm"},"#",-1),p=t("b",null,[t("u",null,"LuxLib.API.batchnorm")],-1),h=t("i",null,"Function",-1),c=s("",2),u={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.471ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.07ex",height:"2.016ex",role:"img",focusable:"false",viewBox:"0 -683 11080.9 891","aria-hidden":"true"},m=s("",1),k=[m],g=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"D"),t("mn",null,"1")]),t("mo",null,"×"),t("mo",null,"."),t("mo",null,"."),t("mo",null,"."),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"N"),t("mo",null,"−"),t("mn",null,"2")])]),t("mo",null,"×"),t("mn",null,"1"),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mi",null,"N")])])],-1),T=t("p",null,[t("strong",null,"Arguments")],-1),b=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),_=t("code",null,"scale",-1),x={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},L=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),f=[L],v=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),E=t("code",null,"nothing",-1),w=t("code",null,"bias",-1),A={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},C={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},F=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),D=[F],H=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),B=t("code",null,"nothing",-1),M=s("",6),V=t("p",null,[t("strong",null,"Returns")],-1),I=t("p",null,[i("Normalized Array of same size as "),t("code",null,"x"),i(". And a Named Tuple containing the updated running mean and variance.")],-1),P=t("p",null,[t("strong",null,"References")],-1),j=t("p",null,'[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International conference on machine learning. PMLR, 2015.',-1),S=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.50/src/api/batchnorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),Z=t("br",null,null,-1),N={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},z=t("a",{id:"LuxLib.API.groupnorm",href:"#LuxLib.API.groupnorm"},"#",-1),R=t("b",null,[t("u",null,"LuxLib.API.groupnorm")],-1),q=t("i",null,"Function",-1),G=s("",4),O=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),U=t("code",null,"scale",-1),J={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},X={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},K=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),W=[K],$=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),Y=t("code",null,"nothing",-1),tt=t("code",null,"bias",-1),it={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},et={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},at=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),st=[at],nt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),lt=t("code",null,"nothing",-1),ot=s("",3),rt=t("p",null,[t("strong",null,"Returns")],-1),dt=t("p",null,"The normalized array is returned.",-1),pt=t("p",null,[t("strong",null,"References")],-1),ht=t("p",null,'[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference on computer vision (ECCV). 2018.',-1),ct=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.50/src/api/groupnorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),ut=t("br",null,null,-1),Qt={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},mt=t("a",{id:"LuxLib.API.instancenorm",href:"#LuxLib.API.instancenorm"},"#",-1),kt=t("b",null,[t("u",null,"LuxLib.API.instancenorm")],-1),gt=t("i",null,"Function",-1),Tt=s("",2),bt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},_t={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.471ex"},xmlns:"http://www.w3.org/2000/svg",width:"22.72ex",height:"2.016ex",role:"img",focusable:"false",viewBox:"0 -683 10042 891","aria-hidden":"true"},xt=s("",1),yt=[xt],Lt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"D"),t("mn",null,"1")]),t("mo",null,"×"),t("mo",null,"."),t("mo",null,"."),t("mo",null,"."),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"N"),t("mo",null,"−"),t("mn",null,"2")])]),t("mo",null,"×"),t("mn",null,"1"),t("mo",null,"×"),t("mn",null,"1")])],-1),ft=t("p",null,[t("strong",null,"Arguments")],-1),vt=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized (must be atleast 3D)")])],-1),Et=t("code",null,"scale",-1),wt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},At={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},Ct=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),Ft=[Ct],Dt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),Ht=t("code",null,"nothing",-1),Bt=t("code",null,"bias",-1),Mt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Vt={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},It=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),Pt=[It],jt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),St=t("code",null,"nothing",-1),Zt=s("",3),Nt=t("p",null,[t("strong",null,"Returns")],-1),zt=t("p",null,[i("Normalized Array of same size as "),t("code",null,"x"),i(". And a Named Tuple containing the updated running mean and variance.")],-1),Rt=t("p",null,[t("strong",null,"References")],-1),qt=t("p",null,'[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).',-1),Gt=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.50/src/api/instancenorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),Ot=t("br",null,null,-1),Ut={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},Jt=t("a",{id:"LuxLib.API.layernorm",href:"#LuxLib.API.layernorm"},"#",-1),Xt=t("b",null,[t("u",null,"LuxLib.API.layernorm")],-1),Kt=t("i",null,"Function",-1),Wt=s("",2),$t={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Yt={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.294ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 572 453","aria-hidden":"true"},ti=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D465",d:"M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z",style:{"stroke-width":"3"}})])])],-1),ii=[ti],ei=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"x")])],-1),ai={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},si={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.76ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.034ex",height:"6.063ex",role:"img",focusable:"false",viewBox:"0 -1460 11064.9 2680","aria-hidden":"true"},ni=s("",1),li=[ni],oi=t("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[t("mi",null,"y"),t("mo",null,"="),t("mfrac",null,[t("mrow",null,[t("mi",null,"x"),t("mo",null,"−"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",{mathvariant:"double-struck"},"E")]),t("mo",{stretchy:"false"},"["),t("mi",null,"x"),t("mo",{stretchy:"false"},"]")]),t("msqrt",null,[t("mi",null,"V"),t("mi",null,"a"),t("mi",null,"r"),t("mo",{stretchy:"false"},"["),t("mi",null,"x"),t("mo",{stretchy:"false"},"]"),t("mo",null,"+"),t("mi",null,"ϵ")])]),t("mo",null,"∗"),t("mi",null,"γ"),t("mo",null,"+"),t("mi",null,"β")])],-1),ri=t("p",null,[i("and applies the activation function "),t("code",null,"σ"),i(" elementwise to "),t("code",null,"y"),i(".")],-1),di=t("p",null,[t("strong",null,"Arguments")],-1),pi=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),hi=t("code",null,"scale",-1),ci={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},ui={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},Qi=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),mi=[Qi],ki=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),gi=t("code",null,"nothing",-1),Ti=t("code",null,"bias",-1),bi={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},_i={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},xi=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),yi=[xi],Li=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),fi=t("code",null,"nothing",-1),vi=s("",3),Ei=t("p",null,[t("strong",null,"Returns")],-1),wi=t("p",null,[i("Normalized Array of same size as "),t("code",null,"x"),i(".")],-1),Ai=t("p",null,[t("strong",null,"References")],-1),Ci=t("p",null,'[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).',-1),Fi=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.50/src/api/layernorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),Di=s("",4);function Hi(Bi,Mi,Vi,Ii,Pi,ji){return a(),e("div",null,[o,t("div",r,[d,i(" "),p,i(" — "),h,i(". "),c,t("p",null,[i("Batch Normalization computes the mean and variance for each "),t("mjx-container",u,[(a(),e("svg",Q,k)),g]),i(" input slice and normalises the input accordingly.")]),T,t("ul",null,[b,t("li",null,[t("p",null,[_,i(": Scale factor ("),t("mjx-container",x,[(a(),e("svg",y,f)),v]),i(") (can be "),E,i(")")])]),t("li",null,[t("p",null,[w,i(": Bias factor ("),t("mjx-container",A,[(a(),e("svg",C,D)),H]),i(") (can be "),B,i(")")])]),M]),V,I,P,j,S]),Z,t("div",N,[z,i(" "),R,i(" — "),q,i(". "),G,t("ul",null,[O,t("li",null,[t("p",null,[U,i(": Scale factor ("),t("mjx-container",J,[(a(),e("svg",X,W)),$]),i(") (can be "),Y,i(")")])]),t("li",null,[t("p",null,[tt,i(": Bias factor ("),t("mjx-container",it,[(a(),e("svg",et,st)),nt]),i(") (can be "),lt,i(")")])]),ot]),rt,dt,pt,ht,ct]),ut,t("div",Qt,[mt,i(" "),kt,i(" — "),gt,i(". "),Tt,t("p",null,[i("Instance Normalization computes the mean and variance for each "),t("mjx-container",bt,[(a(),e("svg",_t,yt)),Lt]),i(" input slice and normalises the input accordingly.")]),ft,t("ul",null,[vt,t("li",null,[t("p",null,[Et,i(": Scale factor ("),t("mjx-container",wt,[(a(),e("svg",At,Ft)),Dt]),i(") (can be "),Ht,i(")")])]),t("li",null,[t("p",null,[Bt,i(": Bias factor ("),t("mjx-container",Mt,[(a(),e("svg",Vt,Pt)),jt]),i(") (can be "),St,i(")")])]),Zt]),Nt,zt,Rt,qt,Gt]),Ot,t("div",Ut,[Jt,i(" "),Xt,i(" — "),Kt,i(". "),Wt,t("p",null,[i("Given an input array "),t("mjx-container",$t,[(a(),e("svg",Yt,ii)),ei]),i(", this layer computes")]),t("mjx-container",ai,[(a(),e("svg",si,li)),oi]),ri,di,t("ul",null,[pi,t("li",null,[t("p",null,[hi,i(": Scale factor ("),t("mjx-container",ci,[(a(),e("svg",ui,mi)),ki]),i(") (can be "),gi,i(")")])]),t("li",null,[t("p",null,[Ti,i(": Bias factor ("),t("mjx-container",bi,[(a(),e("svg",_i,yi)),Li]),i(") (can be "),fi,i(")")])]),vi]),Ei,wi,Ai,Ci,Fi]),Di])}const Ni=n(l,[["render",Hi]]);export{Zi as __pageData,Ni as default};
