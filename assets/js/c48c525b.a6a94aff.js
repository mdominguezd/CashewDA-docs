"use strict";(self.webpackChunkcashew_da_docs=self.webpackChunkcashew_da_docs||[]).push([[283],{6697:(s,e,a)=>{a.r(e),a.d(e,{assets:()=>r,contentTitle:()=>i,default:()=>h,frontMatter:()=>l,metadata:()=>c,toc:()=>o});var n=a(5893),t=a(1151);const l={},i=void 0,c={id:"Models/Loss_Functions",title:"Loss_Functions",description:"FocalLoss",source:"@site/docs/Models/Loss_Functions.md",sourceDirName:"Models",slug:"/Models/Loss_Functions",permalink:"/CashewDA-docs/docs/Models/Loss_Functions",draft:!1,unlisted:!1,editUrl:"https://github.com/${organizationName}/${projectName}/tree/main/docs/Models/Loss_Functions.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"BuildingBlocks",permalink:"/CashewDA-docs/docs/Models/BuildingBlocks"},next:{title:"U_Net",permalink:"/CashewDA-docs/docs/Models/U_Net"}},r={},o=[{value:"FocalLoss",id:"focalloss",level:2},{value:"Attributes",id:"attributes",level:3},{value:"Methods",id:"methods",level:3},{value:"Source code",id:"source-code",level:3}];function m(s){const e={a:"a",annotation:"annotation",code:"code",h2:"h2",h3:"h3",math:"math",mi:"mi",mn:"mn",mo:"mo",mrow:"mrow",msup:"msup",mtext:"mtext",p:"p",pre:"pre",semantics:"semantics",span:"span",...(0,t.a)(),...s.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(e.h2,{id:"focalloss",children:"FocalLoss"}),"\n",(0,n.jsxs)(e.p,{children:["Class used to calculate the focal loss used for the backward propagation of the segmentation heads of the ",(0,n.jsx)(e.a,{href:"./U_Net#networks-implemented",children:"Networks implemented"}),"."]}),"\n",(0,n.jsx)(e.p,{children:"The Focal loss can be calculated as:\n==CHECK=="}),"\n",(0,n.jsxs)(e.p,{children:[(0,n.jsxs)(e.span,{className:"katex",children:[(0,n.jsx)(e.span,{className:"katex-mathml",children:(0,n.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,n.jsxs)(e.semantics,{children:[(0,n.jsxs)(e.mrow,{children:[(0,n.jsx)(e.mi,{children:"F"}),(0,n.jsx)(e.mi,{children:"o"}),(0,n.jsx)(e.mi,{children:"c"}),(0,n.jsx)(e.mi,{children:"a"}),(0,n.jsx)(e.mi,{children:"l"}),(0,n.jsx)(e.mtext,{children:"\xa0"}),(0,n.jsx)(e.mi,{children:"l"}),(0,n.jsx)(e.mi,{children:"o"}),(0,n.jsx)(e.mi,{children:"s"}),(0,n.jsx)(e.mi,{children:"s"}),(0,n.jsx)(e.mo,{children:"="}),(0,n.jsx)(e.mo,{children:"\u2212"}),(0,n.jsx)(e.mo,{stretchy:"false",children:"("}),(0,n.jsx)(e.mn,{children:"1"}),(0,n.jsx)(e.mo,{children:"\u2212"}),(0,n.jsx)(e.mi,{children:"p"}),(0,n.jsxs)(e.msup,{children:[(0,n.jsx)(e.mo,{stretchy:"false",children:")"}),(0,n.jsx)(e.mi,{children:"\u03b3"})]}),(0,n.jsx)(e.mo,{children:"\u22c5"}),(0,n.jsx)(e.mi,{children:"log"}),(0,n.jsx)(e.mo,{children:"\u2061"}),(0,n.jsx)(e.mo,{stretchy:"false",children:"("}),(0,n.jsx)(e.mi,{children:"p"}),(0,n.jsx)(e.mo,{stretchy:"false",children:")"})]}),(0,n.jsx)(e.annotation,{encoding:"application/x-tex",children:" Focal\\ loss = -(1-p)^\\gamma \\cdot \\log(p)"})]})})}),(0,n.jsxs)(e.span,{className:"katex-html","aria-hidden":"true",children:[(0,n.jsxs)(e.span,{className:"base",children:[(0,n.jsx)(e.span,{className:"strut",style:{height:"0.6944em"}}),(0,n.jsx)(e.span,{className:"mord mathnormal",style:{marginRight:"0.13889em"},children:"F"}),(0,n.jsx)(e.span,{className:"mord mathnormal",children:"oc"}),(0,n.jsx)(e.span,{className:"mord mathnormal",children:"a"}),(0,n.jsx)(e.span,{className:"mord mathnormal",style:{marginRight:"0.01968em"},children:"l"}),(0,n.jsx)(e.span,{className:"mspace",children:"\xa0"}),(0,n.jsx)(e.span,{className:"mord mathnormal",style:{marginRight:"0.01968em"},children:"l"}),(0,n.jsx)(e.span,{className:"mord mathnormal",children:"oss"}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,n.jsx)(e.span,{className:"mrel",children:"="}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,n.jsxs)(e.span,{className:"base",children:[(0,n.jsx)(e.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,n.jsx)(e.span,{className:"mord",children:"\u2212"}),(0,n.jsx)(e.span,{className:"mopen",children:"("}),(0,n.jsx)(e.span,{className:"mord",children:"1"}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2222em"}}),(0,n.jsx)(e.span,{className:"mbin",children:"\u2212"}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2222em"}})]}),(0,n.jsxs)(e.span,{className:"base",children:[(0,n.jsx)(e.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,n.jsx)(e.span,{className:"mord mathnormal",children:"p"}),(0,n.jsxs)(e.span,{className:"mclose",children:[(0,n.jsx)(e.span,{className:"mclose",children:")"}),(0,n.jsx)(e.span,{className:"msupsub",children:(0,n.jsx)(e.span,{className:"vlist-t",children:(0,n.jsx)(e.span,{className:"vlist-r",children:(0,n.jsx)(e.span,{className:"vlist",style:{height:"0.6644em"},children:(0,n.jsxs)(e.span,{style:{top:"-3.063em",marginRight:"0.05em"},children:[(0,n.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,n.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,n.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.05556em"},children:"\u03b3"})})]})})})})})]}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2222em"}}),(0,n.jsx)(e.span,{className:"mbin",children:"\u22c5"}),(0,n.jsx)(e.span,{className:"mspace",style:{marginRight:"0.2222em"}})]}),(0,n.jsxs)(e.span,{className:"base",children:[(0,n.jsx)(e.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,n.jsxs)(e.span,{className:"mop",children:["lo",(0,n.jsx)(e.span,{style:{marginRight:"0.01389em"},children:"g"})]}),(0,n.jsx)(e.span,{className:"mopen",children:"("}),(0,n.jsx)(e.span,{className:"mord mathnormal",children:"p"}),(0,n.jsx)(e.span,{className:"mclose",children:")"})]})]})]}),"\nBig shout out to ",(0,n.jsx)(e.a,{href:"https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b",children:"f1recracker"}),", whose implementation was adapted to create the focal loss function."]}),"\n",(0,n.jsx)(e.h3,{id:"attributes",children:"Attributes"}),"\n",(0,n.jsx)(e.h3,{id:"methods",children:"Methods"}),"\n",(0,n.jsx)(e.h3,{id:"source-code",children:"Source code"}),"\n",(0,n.jsx)(e.pre,{children:(0,n.jsx)(e.code,{className:"language-python",children:"class FocalLoss(nn.Module):\n    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index = None):\n        super(FocalLoss, self).__init__()\n        self.gamma = gamma\n        self.alpha = alpha\n        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])\n        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)\n        self.size_average = size_average\n        self.ignore_index = ignore_index\n\n    def forward(self, input, target):\n\n        if input.dim() > 2:\n            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W\n            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C\n            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C\n        \n        target = target.view(-1,1)\n        \n        if self.ignore_index != None:\n            # Filter predictions with ignore label from loss computation\n            mask = target != self.ignore_index\n\n            target = target[mask[:,0], :]\n            input = input[mask[:,0], :]\n        \n        logpt = F.log_softmax(input, dim=-1)\n        logpt = logpt.gather(1, target)\n        logpt = logpt.view(-1)\n        \n        pt = Variable(logpt.data.exp())\n\n        if self.alpha is not None:\n            if self.alpha.type()!=input.data.type():\n                self.alpha = self.alpha.type_as(input.data)\n            at = self.alpha.gather(0,target.data.view(-1))\n            logpt = logpt * Variable(at)\n\n        loss = -1 * (1-pt)**self.gamma * logpt\n        if self.size_average: return loss.mean()\n        else: return loss.sum()\n"})})]})}function h(s={}){const{wrapper:e}={...(0,t.a)(),...s.components};return e?(0,n.jsx)(e,{...s,children:(0,n.jsx)(m,{...s})}):m(s)}},1151:(s,e,a)=>{a.d(e,{Z:()=>c,a:()=>i});var n=a(7294);const t={},l=n.createContext(t);function i(s){const e=n.useContext(l);return n.useMemo((function(){return"function"==typeof s?s(e):{...e,...s}}),[e,s])}function c(s){let e;return e=s.disableParentContext?"function"==typeof s.components?s.components(t):s.components||t:i(s.components),n.createElement(l.Provider,{value:e},s.children)}}}]);