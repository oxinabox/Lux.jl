import{_ as n,c as a,j as t,a as i,a4 as e,o as s}from"./chunks/framework.GYfaOXHm.js";const ji=JSON.parse('{"title":"LuxLib","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/LuxLib.md","filePath":"api/Building_Blocks/LuxLib.md","lastUpdated":null}'),o={name:"api/Building_Blocks/LuxLib.md"},l=e("",29),r={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},d=t("a",{id:"LuxLib.batchnorm",href:"#LuxLib.batchnorm"},"#",-1),p=t("b",null,[t("u",null,"LuxLib.batchnorm")],-1),h=t("i",null,"Function",-1),c=e("",2),u={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.471ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.07ex",height:"2.016ex",role:"img",focusable:"false",viewBox:"0 -683 11080.9 891","aria-hidden":"true"},m=e("",1),k=[m],T=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"D"),t("mn",null,"1")]),t("mo",null,"×"),t("mo",null,"."),t("mo",null,"."),t("mo",null,"."),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"N"),t("mo",null,"−"),t("mn",null,"2")])]),t("mo",null,"×"),t("mn",null,"1"),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mi",null,"N")])])],-1),g=t("p",null,[t("strong",null,"Arguments")],-1),b=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),_=t("code",null,"scale",-1),x={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},L=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),f=[L],v=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),E=t("code",null,"nothing",-1),w=t("code",null,"bias",-1),C={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},F={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},A=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),D=[A],H=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),M=t("code",null,"nothing",-1),V=e("",6),B=e("",7),j=t("br",null,null,-1),S={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},Z=t("a",{id:"LuxLib.groupnorm",href:"#LuxLib.groupnorm"},"#",-1),P=t("b",null,[t("u",null,"LuxLib.groupnorm")],-1),N=t("i",null,"Function",-1),I=e("",4),z=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),R=t("code",null,"scale",-1),q={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},G={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},U=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),O=[U],J=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),X=t("code",null,"nothing",-1),K=t("code",null,"bias",-1),W={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},$={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},Y=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),tt=[Y],it=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),et=t("code",null,"nothing",-1),at=e("",3),st=t("p",null,[t("strong",null,"Returns")],-1),nt=t("p",null,"The normalized array is returned.",-1),ot=t("p",null,[t("strong",null,"References")],-1),lt=t("p",null,'[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference on computer vision (ECCV). 2018.',-1),rt=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.40/src/api/groupnorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),dt=t("br",null,null,-1),pt={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},ht=t("a",{id:"LuxLib.instancenorm",href:"#LuxLib.instancenorm"},"#",-1),ct=t("b",null,[t("u",null,"LuxLib.instancenorm")],-1),ut=t("i",null,"Function",-1),Qt=e("",2),mt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},kt={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.471ex"},xmlns:"http://www.w3.org/2000/svg",width:"22.72ex",height:"2.016ex",role:"img",focusable:"false",viewBox:"0 -683 10042 891","aria-hidden":"true"},Tt=e("",1),gt=[Tt],bt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msub",null,[t("mi",null,"D"),t("mn",null,"1")]),t("mo",null,"×"),t("mo",null,"."),t("mo",null,"."),t("mo",null,"."),t("mo",null,"×"),t("msub",null,[t("mi",null,"D"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",null,"N"),t("mo",null,"−"),t("mn",null,"2")])]),t("mo",null,"×"),t("mn",null,"1"),t("mo",null,"×"),t("mn",null,"1")])],-1),_t=t("p",null,[t("strong",null,"Arguments")],-1),xt=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized (must be atleast 3D)")])],-1),yt=t("code",null,"scale",-1),Lt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},ft={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},vt=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),Et=[vt],wt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),Ct=t("code",null,"nothing",-1),Ft=t("code",null,"bias",-1),At={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Dt={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},Ht=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),Mt=[Ht],Vt=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),Bt=t("code",null,"nothing",-1),jt=e("",3),St=t("p",null,[t("strong",null,"Returns")],-1),Zt=t("p",null,[i("Normalized Array of same size as "),t("code",null,"x"),i(". And a Named Tuple containing the updated running mean and variance.")],-1),Pt=t("p",null,[t("strong",null,"References")],-1),Nt=t("p",null,'[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).',-1),It=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.40/src/api/instancenorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),zt=t("br",null,null,-1),Rt={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},qt=t("a",{id:"LuxLib.layernorm",href:"#LuxLib.layernorm"},"#",-1),Gt=t("b",null,[t("u",null,"LuxLib.layernorm")],-1),Ut=t("i",null,"Function",-1),Ot=e("",2),Jt={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Xt={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.294ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 572 453","aria-hidden":"true"},Kt=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D465",d:"M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z",style:{"stroke-width":"3"}})])])],-1),Wt=[Kt],$t=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"x")])],-1),Yt={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},ti={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.76ex"},xmlns:"http://www.w3.org/2000/svg",width:"25.034ex",height:"6.063ex",role:"img",focusable:"false",viewBox:"0 -1460 11064.9 2680","aria-hidden":"true"},ii=e("",1),ei=[ii],ai=t("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[t("mi",null,"y"),t("mo",null,"="),t("mfrac",null,[t("mrow",null,[t("mi",null,"x"),t("mo",null,"−"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mi",{mathvariant:"double-struck"},"E")]),t("mo",{stretchy:"false"},"["),t("mi",null,"x"),t("mo",{stretchy:"false"},"]")]),t("msqrt",null,[t("mi",null,"V"),t("mi",null,"a"),t("mi",null,"r"),t("mo",{stretchy:"false"},"["),t("mi",null,"x"),t("mo",{stretchy:"false"},"]"),t("mo",null,"+"),t("mi",null,"ϵ")])]),t("mo",null,"∗"),t("mi",null,"γ"),t("mo",null,"+"),t("mi",null,"β")])],-1),si=t("p",null,[i("and applies the activation function "),t("code",null,"σ"),i(" elementwise to "),t("code",null,"y"),i(".")],-1),ni=t("p",null,[t("strong",null,"Arguments")],-1),oi=t("li",null,[t("p",null,[t("code",null,"x"),i(": Input to be Normalized")])],-1),li=t("code",null,"scale",-1),ri={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},di={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.489ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.229ex",height:"1.486ex",role:"img",focusable:"false",viewBox:"0 -441 543 657","aria-hidden":"true"},pi=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FE",d:"M31 249Q11 249 11 258Q11 275 26 304T66 365T129 418T206 441Q233 441 239 440Q287 429 318 386T371 255Q385 195 385 170Q385 166 386 166L398 193Q418 244 443 300T486 391T508 430Q510 431 524 431H537Q543 425 543 422Q543 418 522 378T463 251T391 71Q385 55 378 6T357 -100Q341 -165 330 -190T303 -216Q286 -216 286 -188Q286 -138 340 32L346 51L347 69Q348 79 348 100Q348 257 291 317Q251 355 196 355Q148 355 108 329T51 260Q49 251 47 251Q45 249 31 249Z",style:{"stroke-width":"3"}})])])],-1),hi=[pi],ci=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"γ")])],-1),ui=t("code",null,"nothing",-1),Qi=t("code",null,"bias",-1),mi={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},ki={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.281ex",height:"2.034ex",role:"img",focusable:"false",viewBox:"0 -705 566 899","aria-hidden":"true"},Ti=t("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[t("g",{"data-mml-node":"math"},[t("g",{"data-mml-node":"mi"},[t("path",{"data-c":"1D6FD",d:"M29 -194Q23 -188 23 -186Q23 -183 102 134T186 465Q208 533 243 584T309 658Q365 705 429 705H431Q493 705 533 667T573 570Q573 465 469 396L482 383Q533 332 533 252Q533 139 448 65T257 -10Q227 -10 203 -2T165 17T143 40T131 59T126 65L62 -188Q60 -194 42 -194H29ZM353 431Q392 431 427 419L432 422Q436 426 439 429T449 439T461 453T472 471T484 495T493 524T501 560Q503 569 503 593Q503 611 502 616Q487 667 426 667Q384 667 347 643T286 582T247 514T224 455Q219 439 186 308T152 168Q151 163 151 147Q151 99 173 68Q204 26 260 26Q302 26 349 51T425 137Q441 171 449 214T457 279Q457 337 422 372Q380 358 347 358H337Q258 358 258 389Q258 396 261 403Q275 431 353 431Z",style:{"stroke-width":"3"}})])])],-1),gi=[Ti],bi=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"β")])],-1),_i=t("code",null,"nothing",-1),xi=e("",3),yi=t("p",null,[t("strong",null,"Returns")],-1),Li=t("p",null,[i("Normalized Array of same size as "),t("code",null,"x"),i(".")],-1),fi=t("p",null,[t("strong",null,"References")],-1),vi=t("p",null,'[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).',-1),Ei=t("p",null,[t("a",{href:"https://github.com/LuxDL/LuxLib.jl/blob/v0.3.40/src/api/layernorm.jl#L1",target:"_blank",rel:"noreferrer"},"source")],-1),wi=e("",4);function Ci(Fi,Ai,Di,Hi,Mi,Vi){return s(),a("div",null,[l,t("div",r,[d,i(" "),p,i(" — "),h,i(". "),c,t("p",null,[i("Batch Normalization computes the mean and variance for each "),t("mjx-container",u,[(s(),a("svg",Q,k)),T]),i(" input slice and normalises the input accordingly.")]),g,t("ul",null,[b,t("li",null,[t("p",null,[_,i(": Scale factor ("),t("mjx-container",x,[(s(),a("svg",y,f)),v]),i(") (can be "),E,i(")")])]),t("li",null,[t("p",null,[w,i(": Bias factor ("),t("mjx-container",C,[(s(),a("svg",F,D)),H]),i(") (can be "),M,i(")")])]),V]),B]),j,t("div",S,[Z,i(" "),P,i(" — "),N,i(". "),I,t("ul",null,[z,t("li",null,[t("p",null,[R,i(": Scale factor ("),t("mjx-container",q,[(s(),a("svg",G,O)),J]),i(") (can be "),X,i(")")])]),t("li",null,[t("p",null,[K,i(": Bias factor ("),t("mjx-container",W,[(s(),a("svg",$,tt)),it]),i(") (can be "),et,i(")")])]),at]),st,nt,ot,lt,rt]),dt,t("div",pt,[ht,i(" "),ct,i(" — "),ut,i(". "),Qt,t("p",null,[i("Instance Normalization computes the mean and variance for each "),t("mjx-container",mt,[(s(),a("svg",kt,gt)),bt]),i(" input slice and normalises the input accordingly.")]),_t,t("ul",null,[xt,t("li",null,[t("p",null,[yt,i(": Scale factor ("),t("mjx-container",Lt,[(s(),a("svg",ft,Et)),wt]),i(") (can be "),Ct,i(")")])]),t("li",null,[t("p",null,[Ft,i(": Bias factor ("),t("mjx-container",At,[(s(),a("svg",Dt,Mt)),Vt]),i(") (can be "),Bt,i(")")])]),jt]),St,Zt,Pt,Nt,It]),zt,t("div",Rt,[qt,i(" "),Gt,i(" — "),Ut,i(". "),Ot,t("p",null,[i("Given an input array "),t("mjx-container",Jt,[(s(),a("svg",Xt,Wt)),$t]),i(", this layer computes")]),t("mjx-container",Yt,[(s(),a("svg",ti,ei)),ai]),si,ni,t("ul",null,[oi,t("li",null,[t("p",null,[li,i(": Scale factor ("),t("mjx-container",ri,[(s(),a("svg",di,hi)),ci]),i(") (can be "),ui,i(")")])]),t("li",null,[t("p",null,[Qi,i(": Bias factor ("),t("mjx-container",mi,[(s(),a("svg",ki,gi)),bi]),i(") (can be "),_i,i(")")])]),xi]),yi,Li,fi,vi,Ei]),wi])}const Si=n(o,[["render",Ci]]);export{ji as __pageData,Si as default};
