import{_ as i,c as o,j as t,a as e,a4 as a,o as s}from"./chunks/framework.B66_d8zn.js";const J=JSON.parse('{"title":"Automatic Differentiation Helpers","description":"","frontmatter":{},"headers":[],"relativePath":"api/Lux/autodiff.md","filePath":"api/Lux/autodiff.md","lastUpdated":null}'),d={name:"api/Lux/autodiff.md"},l=a("",4),n={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},r=t("a",{id:"Lux.jacobian_vector_product",href:"#Lux.jacobian_vector_product"},"#",-1),c=t("b",null,[t("u",null,"Lux.jacobian_vector_product")],-1),T=t("i",null,"Function",-1),p=a("",1),h={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.812ex",height:"4.07ex",role:"img",focusable:"false",viewBox:"0 -1149.5 3010.7 1799","aria-hidden":"true"},u=a("",1),m=[u],_=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"u")])],-1),g=a("",8),f=t("br",null,null,-1),b={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},x=t("a",{id:"Lux.vector_jacobian_product",href:"#Lux.vector_jacobian_product"},"#",-1),k=t("b",null,[t("u",null,"Lux.vector_jacobian_product")],-1),y=t("i",null,"Function",-1),w=a("",1),A={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},v={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.126ex",height:"4.536ex",role:"img",focusable:"false",viewBox:"0 -1355.3 3591.5 2004.8","aria-hidden":"true"},L=a("",1),D=[L],j=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msup",null,[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"T")]),t("mi",null,"u")])],-1),P=a("",8),C=a("",6);function E(V,H,M,N,S,B){return s(),o("div",null,[l,t("div",n,[r,e(" "),c,e(" — "),T,e(". "),p,t("p",null,[e("Compute the Jacobian-Vector Product "),t("mjx-container",h,[(s(),o("svg",Q,m)),_]),e(". This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector products efficiently using mixed-mode AD.")]),g]),f,t("div",b,[x,e(" "),k,e(" — "),y,e(". "),w,t("p",null,[e("Compute the Vector-Jacobian Product "),t("mjx-container",A,[(s(),o("svg",v,D)),j]),e(". This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian products efficiently using mixed-mode AD.")]),P]),C])}const Z=i(d,[["render",E]]);export{J as __pageData,Z as default};
