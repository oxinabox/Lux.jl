import{_ as h,c as s,l as A,a,a3 as n,o as i}from"./chunks/framework.DlIaf7OG.js";const XA=JSON.parse('{"title":"Training a Neural ODE to Model Gravitational Waveforms","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/advanced/1_GravitationalWaveForm.md","filePath":"tutorials/advanced/1_GravitationalWaveForm.md","lastUpdated":null}'),e={name:"tutorials/advanced/1_GravitationalWaveForm.md"},t=n("",7),l={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},p={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"10.819ex",height:"1.658ex",role:"img",focusable:"false",viewBox:"0 -583 4782.1 733","aria-hidden":"true"},k=n("",1),E=[k],d=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"r"),A("mo",null,"="),A("msub",null,[A("mi",null,"r"),A("mn",null,"1")]),A("mo",null,"−"),A("msub",null,[A("mi",null,"r"),A("mn",null,"2")])])],-1),r={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.008ex",height:"1.339ex",role:"img",focusable:"false",viewBox:"0 -442 887.6 592","aria-hidden":"true"},o=n("",1),g=[o],C=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("msub",null,[A("mi",null,"r"),A("mn",null,"1")])])],-1),f={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},v={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.008ex",height:"1.339ex",role:"img",focusable:"false",viewBox:"0 -442 887.6 592","aria-hidden":"true"},y=n("",1),I=[y],u=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("msub",null,[A("mi",null,"r"),A("mn",null,"2")])])],-1),F=n("",2),c={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"24.527ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 10840.9 1000","aria-hidden":"true"},T=n("",1),V=[T],B=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mo",{stretchy:"false"},"("),A("mi",null,"χ"),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")"),A("mo",null,","),A("mi",null,"ϕ"),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")"),A("mo",{stretchy:"false"},")"),A("mo",{stretchy:"false"},"↦"),A("mo",{stretchy:"false"},"("),A("mi",null,"x"),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")"),A("mo",null,","),A("mi",null,"y"),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")"),A("mo",{stretchy:"false"},")")])],-1),D=n("",13),m={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},z={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.117ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3587.6 1000","aria-hidden":"true"},K=n("",1),b=[K],M=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mi",null,"u"),A("mo",{stretchy:"false"},"["),A("mn",null,"1"),A("mo",{stretchy:"false"},"]"),A("mo",null,"="),A("mi",null,"χ")])],-1),U={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},X={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.049ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3557.6 1000","aria-hidden":"true"},Z=n("",1),w=[Z],P=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mi",null,"u"),A("mo",{stretchy:"false"},"["),A("mn",null,"2"),A("mo",{stretchy:"false"},"]"),A("mo",null,"="),A("mi",null,"ϕ")])],-1),R={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},N={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.138ex",height:"1.439ex",role:"img",focusable:"false",viewBox:"0 -442 503 636","aria-hidden":"true"},W=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D45D",d:"M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",style:{"stroke-width":"3"}})])])],-1),x=[W],O=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"p")])],-1),G={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},H={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"2.378ex",height:"1.545ex",role:"img",focusable:"false",viewBox:"0 -683 1051 683","aria-hidden":"true"},L=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D440",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",style:{"stroke-width":"3"}})])])],-1),j=[L],S=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"M")])],-1),J={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.054ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 466 453","aria-hidden":"true"},_=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D452",d:"M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",style:{"stroke-width":"3"}})])])],-1),$=[_],AA=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"e")])],-1),sA=n("",14),iA={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},aA={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.117ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3587.6 1000","aria-hidden":"true"},nA=n("",1),hA=[nA],eA=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mi",null,"u"),A("mo",{stretchy:"false"},"["),A("mn",null,"1"),A("mo",{stretchy:"false"},"]"),A("mo",null,"="),A("mi",null,"χ")])],-1),tA={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},lA={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.049ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3557.6 1000","aria-hidden":"true"},pA=n("",1),kA=[pA],EA=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mi",null,"u"),A("mo",{stretchy:"false"},"["),A("mn",null,"2"),A("mo",{stretchy:"false"},"]"),A("mo",null,"="),A("mi",null,"ϕ")])],-1),dA={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},rA={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.138ex",height:"1.439ex",role:"img",focusable:"false",viewBox:"0 -442 503 636","aria-hidden":"true"},QA=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D45D",d:"M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",style:{"stroke-width":"3"}})])])],-1),oA=[QA],gA=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"p")])],-1),CA={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},fA={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"2.378ex",height:"1.545ex",role:"img",focusable:"false",viewBox:"0 -683 1051 683","aria-hidden":"true"},vA=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D440",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",style:{"stroke-width":"3"}})])])],-1),yA=[vA],IA=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"M")])],-1),uA={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},FA={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.054ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 466 453","aria-hidden":"true"},cA=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D452",d:"M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",style:{"stroke-width":"3"}})])])],-1),qA=[cA],TA=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"e")])],-1),VA=n("",31);function BA(DA,mA,zA,KA,bA,MA){return i(),s("div",null,[t,A("p",null,[a("We need a very crude 2-body path. Assume the 1-body motion is a newtonian 2-body position vector "),A("mjx-container",l,[(i(),s("svg",p,E)),d]),a(" and use Newtonian formulas to get "),A("mjx-container",r,[(i(),s("svg",Q,g)),C]),a(", "),A("mjx-container",f,[(i(),s("svg",v,I)),u]),a(" (e.g. Theoretical Mechanics of Particles and Continua 4.3)")]),F,A("p",null,[a("Next we define a function to perform the change of variables: "),A("mjx-container",c,[(i(),s("svg",q,V)),B])]),D,A("mjx-container",m,[(i(),s("svg",z,b)),M]),A("mjx-container",U,[(i(),s("svg",X,w)),P]),A("p",null,[a("where, "),A("mjx-container",R,[(i(),s("svg",N,x)),O]),a(", "),A("mjx-container",G,[(i(),s("svg",H,j)),S]),a(", and "),A("mjx-container",J,[(i(),s("svg",Y,$)),AA]),a(" are constants")]),sA,A("mjx-container",iA,[(i(),s("svg",aA,hA)),eA]),A("mjx-container",tA,[(i(),s("svg",lA,kA)),EA]),A("p",null,[a("where, "),A("mjx-container",dA,[(i(),s("svg",rA,oA)),gA]),a(", "),A("mjx-container",CA,[(i(),s("svg",fA,yA)),IA]),a(", and "),A("mjx-container",uA,[(i(),s("svg",FA,qA)),TA]),a(" are constants")]),VA])}const ZA=h(e,[["render",BA]]);export{XA as __pageData,ZA as default};
