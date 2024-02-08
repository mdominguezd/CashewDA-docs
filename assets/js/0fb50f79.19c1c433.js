"use strict";(self.webpackChunkcashew_da_docs=self.webpackChunkcashew_da_docs||[]).push([[328],{4758:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>u,frontMatter:()=>r,metadata:()=>a,toc:()=>d});var t=i(5893),s=i(1151);const r={sidebar_position:1},o=void 0,a={id:"Training/Hyper_Parameter_Tuning",title:"Hyper_Parameter_Tuning",description:"Brief description of the submodule",source:"@site/docs/Training/Hyper_Parameter_Tuning.md",sourceDirName:"Training",slug:"/Training/Hyper_Parameter_Tuning",permalink:"/CashewDA-docs/docs/Training/Hyper_Parameter_Tuning",draft:!1,unlisted:!1,editUrl:"https://github.com/${organizationName}/${projectName}/tree/main/docs/Training/Hyper_Parameter_Tuning.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"Training",permalink:"/CashewDA-docs/docs/category/training"},next:{title:"Train_DomainOnly",permalink:"/CashewDA-docs/docs/Training/Train_DomainOnly"}},l={},d=[{value:"Brief description of the submodule",id:"brief-description-of-the-submodule",level:2},{value:"HP_Tuning()",id:"hp_tuning",level:2},{value:"Params",id:"params",level:3},{value:"Outputs",id:"outputs",level:3},{value:"Dependencies used",id:"dependencies-used",level:3},{value:"Source code",id:"source-code",level:3},{value:"LoveDA_HP_Tuning()",id:"loveda_hp_tuning",level:2},{value:"Params",id:"params-1",level:3},{value:"Outputs",id:"outputs-1",level:3},{value:"Dependencies used",id:"dependencies-used-1",level:3},{value:"Source code",id:"source-code-1",level:3}];function c(e){const n={a:"a",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h2,{id:"brief-description-of-the-submodule",children:"Brief description of the submodule"}),"\n",(0,t.jsxs)(n.p,{children:["In this submodule the functions used for the selection of hyper-parameters for the training of the ",(0,t.jsx)(n.a,{href:"../Models/U_Net#networks-implemented",children:"Networks implemented"}),"."]}),"\n",(0,t.jsx)(n.h2,{id:"hp_tuning",children:"HP_Tuning()"}),"\n",(0,t.jsxs)(n.p,{children:["Function to execute the hyper-parameter tuning for the ",(0,t.jsx)(n.a,{href:"../Models/U_Net#unet",children:"UNet"})," model on the Cashew Dataset."]}),"\n",(0,t.jsx)(n.p,{children:"It receives the possible values of the hyperparameters as lists and returns a dataframe with the results of each possible combination."}),"\n",(0,t.jsx)(n.p,{children:"The metrics considered are:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Validation F1-Score:"})," Highest value of validation F1-Score obtained during training."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Training time:"})," Time spent on training."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Training rho:"})," Spearman coefficient to check that the training accuracy is continuously increasing with the epochs."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"NO Learning:"})," Boolean indicating if accuracy did improve compared to the one calculated in epoch 0."]}),"\n"]}),"\n",(0,t.jsxs)(n.p,{children:["The calculation of each of the metrics is done using ",(0,t.jsx)(n.strong,{children:"20 epochs"})," and a ",(0,t.jsx)(n.strong,{children:"Linear normalization"})," of the Cashew dataset. For more information of this dataset go ",(0,t.jsx)(n.a,{href:"../Dataset/ReadyToTrain_DS",children:"here"}),"."]}),"\n",(0,t.jsx)(n.h3,{id:"params",children:"Params"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"dir:"})," (dir) Directory with the dataset to be used."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"BS:"})," (list) List with values of batch_size to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"LR"}),": (list) List with values of learning rate to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"STCh:"})," (list) List with values of starting number of channels to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"mu:"})," (list) List with values of momentum to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Bi:"})," (llist) List with values of bilinear to be considered during HP tuning. (Only True or False possible)"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"gamma:"})," (list) List with values of gamma vlaues for the focal loss to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"VI:"})," (list) List with values of vegetation indices (True or False)"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"decay:"})," (list) List with values of the decay rate of learning rate."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"atts:"})," (list) List with booleans for inclusion or not of Attention gates."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"res:"})," (list) List with booleans for inclusion or not of residual connections on the double convolutional blocks."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"tr_size:"})," (float: [0-1]) Amount of training set considered for HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"val_size:"})," (float: [0-1]) Amount of validation set considered for HP tuning."]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"outputs",children:"Outputs"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"HP_values:"})," (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning."]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"dependencies-used",children:"Dependencies used"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"import pandas as pd\nimport time\nfrom torchmetrics.classification import BinaryF1Score\n\nfrom Dataset.Transforms import getTransforms\nfrom Dataset.ReadyToTrain_DS import getDataLoaders\nfrom Models.U_Net import UNet\nfrom Models.Loss_Functions import FocalLoss\n"})}),"\n",(0,t.jsx)(n.h3,{id:"source-code",children:"Source code"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"def HP_Tuning(dir, BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.15, val_size = 0.75):\n    \"\"\"\n        Function to perform Hyperparameter tuning for the networks to be trained.\n\n        Input:\n            - dir: Directory with the dataset to be used.\n            - BS: List with values of batch_size to be considered during HP tuning.\n            - LR: List with values of learning rate to be considered during HP tuning.\n            - STCh: List with values of starting number of channels to be considered during HP tuning.\n            - mu: List with values of momentum to be considered during HP tuning.\n            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)\n            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.\n            - VI: List with values of vegetation indices (True or False)\n            - decay: decay rate of learning rate.\n            - atts: Inclusion or not of Attention gates.\n            - res: Inclusion or not of residual connections on convolutional blocks.\n            - tr_size: Amount of training set considered.\n            - val_size: Amount of validation et cosidered.\n            \n        Output:\n            - HP_values: (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.\n    \"\"\"\n\n    transforms = get_transforms()\n    normalization = 'Linear_1_99'\n    epochs = 12\n\n    rows = []\n\n    for bs in BS:\n        for lr in LR:\n            for stch in STCh:\n                for mu in MU:\n                    for bi in Bi:\n                        for g in gamma:\n                            for vi in VI:\n                                for de in decay:\n                                    for at in atts:\n                                        for re in res:\n                                            train_loader, val_loader, test_loader = get_DataLoaders(dir, bs, transforms, normalization, vi, train_split_size = tr_size, val_split_size = val_size)\n                                            n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number fomr actual data\n                                            n_classes = 2\n                \n                                            loss_function = FocalLoss(gamma = g)\n                \n                                            # Define the network\n                                            network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)\n            \n                                            start = time.time()\n                                            f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False)\n                                            end = time.time()\n            \n                                            rows.append([bs, lr, stch, mu, bi, g, vi, de, at, re, f1_val, end-start, spearman, no_l])\n            \n                                            HP_values = pd.DataFrame(rows)\n                                            HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'VI', 'decay', 'attention', 'resnet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']\n                                            HP_values.to_csv('TempHyperParamTuning_'+dir+'.csv')\n    \n    return HP_values\n"})}),"\n",(0,t.jsx)(n.h2,{id:"loveda_hp_tuning",children:"LoveDA_HP_Tuning()"}),"\n",(0,t.jsxs)(n.p,{children:["Function to execute the hyper-parameter tuning for the ",(0,t.jsx)(n.a,{href:"../Models/U_Net#unet",children:"UNet"})," model on the ",(0,t.jsx)(n.strong,{children:"LoveDA"})," dataset"]}),"\n",(0,t.jsx)(n.p,{children:"It receives the possible values of the hyperparameters as lists and returns a dataframe with the results of each possible combination."}),"\n",(0,t.jsx)(n.p,{children:"The metrics used are:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Validation mIOU:"})," Highest value of mIOU obtained during training."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Training time:"})," Time spent on training."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Training rho:"})," Spearman coefficient to check that the training accuracy is continuously increasing with the epochs."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"NO Learning:"})," Boolean indicating if accuracy did improve compared to the one calculated in epoch 0."]}),"\n"]}),"\n",(0,t.jsxs)(n.p,{children:["The calculation of each of the metrics is done using ",(0,t.jsx)(n.strong,{children:"15 epochs"}),"."]}),"\n",(0,t.jsx)(n.h3,{id:"params-1",children:"Params"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"BS:"})," (list) List with values of batch_size to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"LR"}),": (list) List with values of learning rate to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"STCh:"})," (list) List with values of starting number of channels to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"mu:"})," (list) List with values of momentum to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Bi:"})," (llist) List with values of bilinear to be considered during HP tuning. (Only True or False possible)"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"gamma:"})," (list) List with values of gamma values for the focal loss to be considered during HP tuning."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"decay:"})," (list) List with values of the decay rate of learning rate."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"atts:"})," (list) List with booleans for inclusion or not of Attention gates."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"res:"})," (list) List with booleans for inclusion or not of residual connections on the double convolutional blocks"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"tr_size:"})," (float: [0-1]) Amount of training set considered for HP tuning. ",(0,t.jsx)(n.em,{children:"Default"})," is 0.15."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"val_size:"})," (float: [0-1]) Amount of validation set considered for HP tuning. ",(0,t.jsx)(n.em,{children:"Default"})," is 0.75."]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"outputs-1",children:"Outputs"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"HP_values:"})," (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning."]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"dependencies-used-1",children:"Dependencies used"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"import pandas as pd\nimport time\nfrom torchmetrics.classification import BinaryF1Score\n\nfrom Dataset.Transforms import getTransforms\nfrom Dataset.ReadyToTrain_DS import get_LOVE_DataLoaders\nfrom Models.U_Net import UNet\nfrom Models.Loss_Functions import FocalLoss\n"})}),"\n",(0,t.jsx)(n.h3,{id:"source-code-1",children:"Source code"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"def LoveDA_HP_Tuning(domain, BS, LR, STCh, MU, Bi, gamma, decay, atts, res, tr_size = 0.15, val_size = 0.75):\n    \"\"\"\n        Function to perform Hyperparameter tuning for the networks to be trained on LoveDA dataset.\n\n        Input:\n            - dir: Directory with the dataset to be used.\n            - BS: List with values of batch_size to be considered during HP tuning.\n            - LR: List with values of learning rate to be considered during HP tuning.\n            - STCh: List with values of starting number of channels to be considered during HP tuning.\n            - mu: List with values of momentum to be considered during HP tuning.\n            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)\n            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.\n            - decay: decay of learning rate.\n            - atts: Boolean indicating if attention gates are used or not.\n            - res: Boolean indicating if residua connections on convolutional blocks are used or not.\n            \n        Output:\n            - HP_values: (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.\n    \"\"\"\n\n    transforms = get_transforms()\n    # normalization = 'Linear_1_99'\n    epochs = 15\n\n    rows = []\n\n    for bs in BS:\n        for lr in LR:\n            for stch in STCh:\n                for mu in MU:\n                    for bi in Bi:\n                        for g in gamma:\n                            for de in decay:\n                                for at in atts:\n                                    for re in res:\n                                        train_loader, val_loader, test_loader = get_LOVE_DataLoaders(domain, bs, train_split_size = tr_size, val_split_size = val_size)\n                                        n_channels = next(enumerate(train_loader))[1]['image'].shape[1] #get band number fomr actual data\n                                        n_classes = 8\n            \n                                        loss_function = FocalLoss(gamma = g, ignore_index = 0)\n            \n                                        # Define the network\n                                        network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)\n        \n                                        start = time.time()\n                                        f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False, accu_function=JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0) , Love = True)\n                                        end = time.time()\n        \n                                        rows.append([bs, lr, stch, mu, bi, g, de, at, re, f1_val, end-start, spearman, no_l])\n        \n                                        HP_values = pd.DataFrame(rows)\n                                        HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'decay', 'attention', 'resunet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']\n                                        HP_values.to_csv('TempHyperParamTuning_LOVE.csv')\n    \n    return HP_values\n"})})]})}function u(e={}){const{wrapper:n}={...(0,s.a)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},1151:(e,n,i)=>{i.d(n,{Z:()=>a,a:()=>o});var t=i(7294);const s={},r=t.createContext(s);function o(e){const n=t.useContext(r);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),t.createElement(r.Provider,{value:n},e.children)}}}]);