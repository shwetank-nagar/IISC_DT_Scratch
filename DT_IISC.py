#Step 1 - function call required
from collections import Counter
import itertools
from math import log
import time
import os
import string
import sys

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        pass

    def read_files(self, filepath):
        '''
        input - File path of input file. 
        read file and convert into 2d array.
        handles only csv format.
        returns Column header and dataset into two different variable.
        
        '''
        datastore=[]
        final_dataset ={}
        with open(filepath,'r') as files:
            for line in files.readlines():
                lines=line.strip().split(',')
                for i in range(len(lines)):
                    if len(lines[i])==0 or len(lines[i])== None:
                        lines[i]=None
                    elif lines[i][0]=='"' or lines[i][-1]=='"':
                        lines[i]=lines[i][1:-1]
                        value=self.digit_check(lines[i])
                    else:
                        value=self.digit_check(lines[i])
                    if value=='int':
                        lines[i]=int(lines[i])
                    elif value=='float':
                        lines[i]=float(lines[i])
                    else:
                        lines[i]=lines[i]
                datastore.append(lines)

        columns=datastore[0]
        dataset=datastore[1:]


        return columns,dataset


    #Step 1.1
    def digit_check(self,user_input):
        '''
        convert string digits into integer / float.
        inputs each element of 2d array.
        return int/float/string to read_files function.
        required - 2d array reads all elements as string char.
        '''
        try:
           val = int(user_input)
           return 'int'
        except ValueError:
          try:
            val = float(user_input)
            return 'float'
          except ValueError:
              return 'string'



    #Step 3.1
    
    def random_number(self,low, high):
        """
        a time based random number generator 
        uses the random time between a user's input events
        returns an integer between low and high-1
        """
        return int(low + int(time.time()*1000) % (high - low))



    #Step 3.2
    def random_indices(self,high,test_size):
        '''
        generates random sample index.
        inputs length of dataset and sample size
        returns radom indices
        
        '''
        test_indices=[]
        while len(test_indices)<test_size:
            indices=self.random_number(len(test_indices),high)
            test_indices.append(indices)
            test_indices = list( dict.fromkeys(test_indices) )
        test_indices.sort()
        return test_indices




    # Step 3 - function call required
    def test_train_split(self,df,test_size):
        '''
        input dataset and test size
        returns test and training data set
        
        '''
        high=len(df)
        train_df=[]
        if isinstance(test_size,float):
            test_size=round(test_size*len(df))
        test_indices = self.random_indices(high,test_size)
        test_df=[df[value] for value in test_indices]
        for i in range(len(df)):
            if i not in test_indices:
                train_df.append(df[i])
        return test_df,train_df

    def convert_feature_to_x_y_relation(self, feature):
      feature_tuple = (tuple(f) for f in feature)
      #print(feature_tuple)


      ls = []
      c= Counter(feature_tuple)
      #print("Counter: ",c)
      for l in c:
          ls.append([l[0] , l[1], c[l]])
      #print("ls",ls)

      key_func = lambda x: x[0].strip()
      ls.sort()
      gr = []  
      
      for key, group in itertools.groupby(ls, key_func):
          #print("key_func",key, list(group))
          gr.append(list(group))
      #print("gr", gr)

      gr.sort()
      final_list = []
      for i,grs in enumerate(gr):
          if len(grs) == 2 :
              final_list.append([gr[i][0][2],gr[i][1][2]])
          elif gr[i][0][1] == 0:
              final_list.append([gr[i][0][2],0])
          elif gr[i][0][1] == 1:
              final_list.append([0,gr[i][0][2]])        
      return final_list




    
    def entropy(self,pi):
        '''
        return the Entropy of a probability distribution:
        entropy(p) = − SUM (Pi * log(Pi) )
        defintion:
                entropy is a metric to measure the uncertainty of a probability distribution.
        entropy ranges between 0 to 1
        Low entropy means the distribution varies (peaks and valleys).
        High entropy means the distribution is uniform.
        
        '''

        total = 0
        for p in pi:
            p = p / sum(pi)
            if p != 0:
                total += p * log(p, 2)
            else:
                total += 0
        total *= -1
        return total


    def gain(self,Y_count_list, feature_list):
        '''
        return the information gain:
        gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
        '''

        total = 0
        for v in feature_list:
            total += sum(v) / sum(Y_count_list) * self.entropy(v)

        gain = self.entropy(Y_count_list) - total
        return gain



#############################################Variable dist is a dictionary with Key as feature and value as a list giving feature value and dependent variable#######################################################
    def get_dictionary_with_x_groups(self,columname,X,Y):
      Variable_dict ={}
      for colm in  range(len(columname)-1):
        temp=[]
        for x,y in zip([x[colm] for x in X] ,Y):
          temp.append([x,y])
        Variable_dict[columname[colm]] = temp  
        temp.append([x,y])
      return Variable_dict



#print(Y_count_list)

#print(Variable_dict['Outlook'])
#print(dt.convert_feature_to_x_y_relation(Variable_dict['Outlook']))




#############################################Below gives a root dictionary, which has key as the feature and value as information gain######################################################
    def give_x_information_gain(self,Variable_dict):
      root={}
      groups={}
      for key in Variable_dict.keys():
        #print("Groups for Feature %s is %s" %(key,  dt.convert_feature_to_x_y_relation(Variable_dict[key])))
        #print("Information Gain for Feature %s is %s" %(key,  dt.gain(Y_count_list,dt.convert_feature_to_x_y_relation(Variable_dict[key]))))
        root[key] =  dt.gain(Y_count_list,dt.convert_feature_to_x_y_relation(Variable_dict[key]))
        groups[key] = Variable_dict[key]
      Max_IG = max(root, key=root.get) 
      root_index = columname.index(Max_IG)
      Max_IG_Value = root[Max_IG]
      return Max_IG,root_index,Max_IG_Value

#############################################Below gives a sub tree on the basis of a node# with respect to Y#####################################################
    def find_subtree_for_node(self,root_index, X, Y):
      all_val_node = list(set([x[root_index] for x in X]))
      child_tree={}
      for l in all_val_node:
        temp_l =[]
        if l not in child_tree.keys():
            child_tree[l] =[]
            for x,y in zip(X,Y):
              if x[root_index] == l:
                temp_l.append(x+[y]) 
            child_tree[l] = temp_l

      return   child_tree    




##################################################


'''
##Demo - Categtorical Dataset
columname,dataset=read_files('AppleStore.csv')
test_df,train_df=test_train_split(dataset,.10)
print (len(dataset))
print (len(train_df))
print (len(test_df))
'''

###Demo - Numerical Dataset
#####Creation of on object#####
dt = DecisionTreeClassifier()
#columname,dataset=dt.read_files('/Python_Projects/Python_Learning/Decision_Tree/data/UCI_Credit_Card.csv')
#####To get current working director and then joining with appropriate data under data folder#####
path =os.path.abspath(os.getcwd())
# load and prepare data

#filename = os.path.join(path,"data/data_banknote_authentication.csv")
filename = os.path.join(path,"data/Golf_data_set.csv")
columname,dataset=dt.read_files(filename)



#####Breaking down the dataset into train_df and test_df, and giving 10% as the split
test_df,train_df=dt.test_train_split(dataset,0)
#this is the dependent variable, or label (y) as a list)
Y =  [item[-1] for item in train_df]
#these are the independent variables (x) as a list of list)
X = [item[:-1] for item in train_df]
print("X is:", X)

Variable_dict ={}
Variable_dict = dt.get_dictionary_with_x_groups(columname,X,Y)
print("\n")
Y_count_list = [Y.count(y) for y in set(Y)]#0,1, although , we need to make sure that it shd come up in a order
Max_IG,root_index,Max_IG_Value = dt.give_x_information_gain(Variable_dict)
print("Max_IG is '", Max_IG ,"' & root_index is ",root_index )
print("\n")
child_tree ={}
child_tree = dt.find_subtree_for_node(root_index, X, Y)
print("child tree is ", child_tree)



child_1 ={}
for key in child_tree.keys():
  print("Child_tree values testing", [x[:-1] for x in child_tree[key]], [x[-1] for x in child_tree[key]])
  Variable_dict_child_1 = dt.get_dictionary_with_x_groups(columname,[x[:-1] for x in child_tree[key]],[x[-1] for x in child_tree[key]])
  Max_IG,root_index,Max_IG_Value = dt.give_x_information_gain(Variable_dict_child_1)
  print("Key is",key, " & Max_IG is '", Max_IG ,"' & root_index is ",root_index , " Max gain value is ", Max_IG_Value)
  if len(set(x[-1] for x in child_tree[key])) == 1:
    child_tree_1 = child_tree[key]
    print("child_tree_1", child_tree_1)
  else:  
    child_tree_1 = dt.find_subtree_for_node(root_index, [x[:-1] for x in child_tree[key]],[x[-1] for x in child_tree[key]]) 
    print("child_tree_1", child_tree_1)
  child_1[key] = child_tree_1
  print("Max_IG is '", Max_IG ,"' & root_index is ",root_index , " Max gain value is ", Max_IG_Value)
  print("\n")



print("child_1 ",child_1)  

