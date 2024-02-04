"use strict";(self.webpackChunkcashew_da_docs=self.webpackChunkcashew_da_docs||[]).push([[396],{3615:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>d,contentTitle:()=>r,default:()=>p,frontMatter:()=>o,metadata:()=>a,toc:()=>c});var t=i(5893),s=i(1151);const o={},r=void 0,a={id:"Dataset/Unzip_DS",title:"Unzip_DS",description:"Brief description of the submodule",source:"@site/docs/Dataset/Unzip_DS.md",sourceDirName:"Dataset",slug:"/Dataset/Unzip_DS",permalink:"/CashewDA-docs/docs/Dataset/Unzip_DS",draft:!1,unlisted:!1,editUrl:"https://github.com/${organizationName}/${projectName}/tree/main/docs/Dataset/Unzip_DS.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Transforms",permalink:"/CashewDA-docs/docs/Dataset/Transforms"},next:{title:"Models",permalink:"/CashewDA-docs/docs/category/models"}},d={},c=[{value:"Brief description of the submodule",id:"brief-description-of-the-submodule",level:2},{value:"UnzipFolders()",id:"unzipfolders",level:2},{value:"Params",id:"params",level:3},{value:"Outputs",id:"outputs",level:3},{value:"Dependencies used",id:"dependencies-used",level:3},{value:"Source code",id:"source-code",level:3}];function l(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h2,{id:"brief-description-of-the-submodule",children:"Brief description of the submodule"}),"\n",(0,t.jsxs)(n.p,{children:["In this submodule you can find the function used to unzip the datasets previously downloaded using the code ",(0,t.jsx)(n.a,{href:"../intro#download-the-pre-processed-datasets",children:"here"})]}),"\n",(0,t.jsx)(n.h2,{id:"unzipfolders",children:"UnzipFolders()"}),"\n",(0,t.jsx)(n.h3,{id:"params",children:"Params"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"domain:"}),' (str) String with the name of the domain. Options: "Tanzania" or "IvoryCoast".']}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"outputs",children:"Outputs"}),"\n",(0,t.jsx)(n.h3,{id:"dependencies-used",children:"Dependencies used"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"from zipfile import ZipFile\nimport os \nimport warnings\n"})}),"\n",(0,t.jsx)(n.h3,{id:"source-code",children:"Source code"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'def UnzipFolders(domain):\n    """\n        Function to unzip the three folds for each dataset.\n\n        Input:\n            - domain: string with the name of the domain. Options: "Tanzania" or "IvoryCoast"\n    """\n    if (domain != "Tanzania") & (domain != "IvoryCoast"):\n        raise Exception("Domain needs to be Tanzania or IvoryCoast")\n\n    if len([i for i in os.listdir(\'.\') if \'.zip\' in i]) != 0: \n        for i in range(3): \n            if (len([f for f in os.listdir(\'.\') if domain + str(i+1) + ".zip" in f]) != 0):\n                with ZipFile(domain + str(i+1) + ".zip", \'r\') as zipped:\n                    zipped.extractall(path="./")\n                os.remove(domain + str(i+1) + ".zip")\n            else:\n                raise warnings.warn("No zipped file found (" + domain + str(i+1) + ".zip" + ")")\n'})})]})}function p(e={}){const{wrapper:n}={...(0,s.a)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(l,{...e})}):l(e)}},1151:(e,n,i)=>{i.d(n,{Z:()=>a,a:()=>r});var t=i(7294);const s={},o=t.createContext(s);function r(e){const n=t.useContext(o);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:r(e.components),t.createElement(o.Provider,{value:n},e.children)}}}]);