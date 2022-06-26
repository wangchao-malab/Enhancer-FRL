# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""
def features_layer2():
    import os

    # 1 kmer
    input_dir="./sequence/seq_layer2.fasta "
    output_dir="./features/layer2_features/t_"

      
    for x in [3]:
        comm_line="python  ./feature_descriptors/descnucleotide/Kmer.py --file " + input_dir\
            +"--kmer " + str(x) +"  --format csv " +' --out ' + output_dir+'kmer_'+str(x)+'.csv'
        
        os.system(comm_line)


    # 2 CKSNAP
    for x in [22]:
        comm_line="python ./feature_descriptors/descnucleotide/CKSNAP.py --file " + input_dir\
                  +"--gap "+str(x)+"  --format csv " +' --out ' + output_dir+'cksnap_'+str(x)+'.csv'

        os.system(comm_line)

        
    # 3 DAC 
    for x in [4]:      
        comm_line="python ./feature_descriptors/iLearn-nucleotide-acc.py --file " + input_dir\
                    +'--method '+ "DAC" + ' --type DNA --lag '+str(x)+' --out ' + output_dir+'dac_'+str(x)+'.csv --format csv'

        os.system(comm_line)

        
    # 4 TACC 
    for x in[5]:      
        comm_line="python ./feature_descriptors/iLearn-nucleotide-acc.py --file " + input_dir\
                    +'--method '+ "TACC" + ' --type DNA --lag '+str(x)+' --out ' + output_dir+'tacc_'+str(x)+'.csv --format csv'

        os.system(comm_line)


    # 5 SCPseDNC
    for x in [4]:
      comm_line="python ./feature_descriptors/iLearn-nucleotide-Pse.py --file " + input_dir\
                +'--method  SCPseDNC'+ ' --type DNA --lamada '+str(x)+ ' --weight 0.1 --out ' + output_dir+'scpsednc_'+str(x)+'.csv --format csv'

      os.system(comm_line)


    # 6 SCPseTNC
    for x in [7]:
      comm_line="python ./feature_descriptors/iLearn-nucleotide-Pse.py --file " + input_dir\
                +'--method  SCPseTNC'+ ' --type DNA --lamada '+str(x)+ ' --weight 0.1 --out ' + output_dir+'scpsetnc_'+str(x)+'.csv --format csv'

      os.system(comm_line)
      

    # feature 7-10
    feature_list4=["ENAC","NCP","ANF","PseEIIP",]
    for x in feature_list4:   
        comm_line="python ./feature_descriptors/iLearn-nucleotide-basic.py --file " + input_dir\
                    +'--method '+ str(x) + ' --out ' +output_dir+str(x)+'.csv --format csv'

        os.system(comm_line)


    import pandas as pd
    from sklearn import preprocessing
    import numpy as np

    dir_list=os.listdir("./model/model_features/layer_2/")
    for x in dir_list:
        if x.split(".")[-1]=="csv":

            xx=x.split(".")[0]
            dfx_train=pd.read_csv('./model/model_features/layer_2/'+str(x),sep=',',header=None,index_col=None).iloc[:,1:]
            
            min_max_scaler = preprocessing.MinMaxScaler().fit(dfx_train)       
            
            dfx_test=pd.read_csv('./features/layer2_features/t_'+str(x),sep=',',header=None,index_col=None).iloc[:,1:]
            dfx_ind=dfx_test.index
            dfx_col=dfx_test.columns
            
            s = min_max_scaler.transform(dfx_test)
            ss=pd.DataFrame(s,index=dfx_ind,columns=dfx_col)
            ss.to_csv('./features/layer2_features/mm/'+"m_t_"+str(xx)+".csv",sep=',')


def pred_proba_of_testing_layer2():
    import numpy as np
    import pandas as pd
    import os
    import joblib
    feature_list=['svm','rf','knn','adab','lgb']
    for classfier in feature_list:
        for fea in ['m_ANF.csv', 'm_cksnap_22.csv', 'm_dac_4.csv' ,'m_ENAC.csv', 'm_kmer_3.csv',
                    'm_NCP.csv', 'm_PseEIIP.csv', 'm_scpsednc_4.csv', 'm_scpsetnc_7.csv', 'm_tacc_5.csv']:
            test_pre_proba=[]
            test_pre=[]
            fea_name="test_order_"+str(classfier)+"_"+str(fea)

            
            X=pd.read_csv('./features/layer2_features/mm/m_t_'+str(fea.split("m_")[1]),sep=',',index_col=0).to_numpy()
            
            model=joblib.load("./model/single_feature_models/layer_2/train_model_"+str(classfier)+"_"+str(fea.split(".")[0])+".m")
            
            y_pred_proba=model.predict_proba(X)[:,1]
            test_pre_proba.append(y_pred_proba)
            
            pd.DataFrame(test_pre_proba).to_csv("./features/layer2_features/proba/testing_pre_proba_"+str(classfier)+"_"+str(fea.split(".")[0])+".csv")
            

def Enhancer_FRL_layer2():
    import os
    import joblib
    import numpy as np
    import pandas as pd

    feature_list=['svm','rf','knn','adab','lgb']
    test_pre_proba=[]
    test_pred_all_testing=[]  
    for y in feature_list:
        for fea in ['m_ANF.csv', 'm_cksnap_22.csv', 'm_dac_4.csv' ,'m_ENAC.csv', 'm_kmer_3.csv',
                    'm_NCP.csv', 'm_PseEIIP.csv', 'm_scpsednc_4.csv', 'm_scpsetnc_7.csv', 'm_tacc_5.csv']:
                    
                fea_name_testing_prob="testing_pre_proba_"+str(y)+"_"+str(fea)
                test_pre_testing=pd.read_csv('./features/layer2_features/proba/'+str(fea_name_testing_prob),sep=',',header=0,index_col=0).to_numpy().reshape(-1)
                test_pre_testing=test_pre_testing[~np.isnan(test_pre_testing)]
                test_pred_all_testing.append(list(test_pre_testing))
  
    pd.DataFrame(np.array(test_pred_all_testing).T).to_csv("./model/pre_combine_testing2.csv")
              
    X_testing=pd.read_csv('./model/pre_combine_testing2.csv',sep=',',index_col=0).to_numpy()
        
    model=joblib.load("./model/single_feature_models/layer_2/Enhancer_FRL_layer2_SVM.m")
        
    y_pred_proba=model.predict_proba(X_testing)[:,1]
    test_pre_proba.append(y_pred_proba)
    pd.DataFrame(np.array(test_pre_proba).T).to_csv("./results/out_layer2.csv")
    
    df1=pd.read_csv('./results/out_layer2.csv',sep=",").to_numpy()[:,1]
    with open("./results/out_layer2.txt", "w") as fil2:
                        fr = open('./sequence/seq_layer2.fasta','r')
                        sequence=fr.readlines()
                        pos=1
                        for num_line in range(0,len(sequence),2):
                            seq_name=sequence[num_line].rstrip("\n")
                            if df1[int(num_line/2)] >0.5:
                                fil2.write(seq_name+","+str(df1[int(num_line/2)])+","+"Strong enhancer"+"\n")
                            else:
                                fil2.write(seq_name+","+str(df1[int(num_line/2)])+","+"Weak enhancer"+"\n")  
