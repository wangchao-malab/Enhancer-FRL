# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""

def Enhancer_FRL(input_file,output_file):
    import os
    from Enhancer_layer1 import features_layer1, pred_proba_of_testing_layer1, Enhancer_FRL_layer1
    from Enhancer_layer2 import features_layer2, pred_proba_of_testing_layer2, Enhancer_FRL_layer2
    print("---------------------------")
    print("   layer1 job running......")
    features_layer1(input_file)
    pred_proba_of_testing_layer1()
    Enhancer_FRL_layer1(input_file)
    print("   layer1 job finished")
    
    print("   layer2 job running......")
    features_layer2()
    pred_proba_of_testing_layer2()
    Enhancer_FRL_layer2()
    print("   layer2 job finished")
    print("---------------------------")
    import pandas as pd
    df1=pd.read_csv('./results/out_layer1.txt',sep=",",index_col=0,header=None)
    df2=pd.read_csv('./results/out_layer2.txt',sep=",",index_col=0,header=None)

    seq_index1=df1.index

    with open('./results/'+str(output_file)+'.csv', "w") as fil1:
        fil1.write("seq_id"+","+"Enhancer_prediction"+","+"probability"+","+"Enhancer classification"+","+"probability"+"\n")
        for x in seq_index1:
            layer1=df1.loc[x].to_numpy()
            fil1.write(str(x)+","+layer1[1]+","+str('%.4f' %layer1[0])+",")
            if layer1[0]>0.5:
                layer2=df2.loc[x].to_numpy() 
                fil1.write(str(layer2[1])+","+str('%.4f'% layer2[0])+"\n")
            else:
                fil1.write(str("-")+","+str("-")+"\n") 
 
def file_remove():
    import os,shutil
    dir_list1=os.listdir("./features/layer1_features/")
    for x in dir_list1:
        if x.split(".")[-1]=="csv":
            os.remove("./features/layer1_features/"+str(x))
    
    dir_list2=os.listdir("./features/layer1_features/proba/")
    for x in dir_list2:
            os.remove("./features/layer1_features/proba/"+str(x))

    dir_list3=os.listdir("./features/layer1_features/mm/")
    for x in dir_list3:
              os.remove("./features/layer1_features/mm/"+str(x))


    dir_list4=os.listdir("./features/layer2_features/")
    for x in dir_list4:
        if x.split(".")[-1]=="csv":
            os.remove("./features/layer2_features/"+str(x))
    
    dir_list5=os.listdir("./features/layer2_features/proba/")
    for x in dir_list5:
            os.remove("./features/layer2_features/proba/"+str(x))

    dir_list6=os.listdir("./features/layer2_features/mm/")
    for x in dir_list6:
              os.remove("./features/layer2_features/mm/"+str(x)) 

              
if __name__=='__main__':
    import argparse
    import pandas as pd
    import os
    os.chdir(r'C:\Users\Administer\Desktop\Enhancer-FRL_master') # seting the working dir
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--inputfile", help="input fasta file",)
    parser.add_argument("-o","--outputfile", help="output fasta file",)
    args = parser.parse_args()
    input_file=args.inputfile
    output_file=args.outputfile
    print("work launched")
    Enhancer_FRL(input_file,output_file)
    file_remove()
    print("job finished !")

