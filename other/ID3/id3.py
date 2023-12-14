import numpy as np
import copy 

def entropy(labels):                                                    # 计算数据集的信息熵                                                                          
    unique_labels, label_counts = np.unique(labels, return_counts=True) # 获取目标变量的唯一值以及统计次数
    total_count = len(labels)                                           # 总样本量个数
    prob = label_counts / total_count                                   # 该唯一值的占比
    ent = -np.sum(prob * np.log2(prob))                                 # 计算信息熵
    return ent

def majority_vote(labels):                                              # 多数表决确定叶节点标签                             
    unique_values, counts = np.unique(labels, return_counts=True)       # 使用numpy.unique()函数统计每个元素的出现次数
    majority_label = unique_values[np.argmax(counts)]                   # 找到出现次数最多的元素
    return majority_label

def choose_best_feature(data):                                          # 选择最佳划分特征,返回索引    
    labels   = data[:,-1]                                               # 获取当前样本集的目标变量，固定-1表示最后一列是标签
    base_entropy = entropy(labels)                                      # 获取当前样本集的信息熵
 
    best_info_gain = 0.0                                                # 设定初始信息增益为0
    best_column  = -1                                                   # 设定初始最优特征列为-1
 
    for index in range(data.shape[1]-1):                                # 获取当前样本集的每一列
        unique_values = np.unique(data[:,index])                        # 获取当前样本集的第index列的唯一值
 
        new_entropy = 0.0                                               # 当前index列的条件熵初始设为0
        sub_data = data[:,[index,-1]]                                   # 获取子数据集(仅包含两列)：第index列,目标变量
 
        for value in unique_values:                                     # 遍历第index列的所有特征值
            label_i = sub_data[sub_data[:,0]==value,1]                  # 根据特定的特征值value划分出的样本子集，并只获取目标变量这一列
            new_entropy += len(label_i)/len(labels) * entropy(label_i)  # 计算index列的特征值value对应的节点的条件熵，并进行累加
        info_gain = base_entropy - new_entropy                          # index列的信息增益 = 当前样本集的信息熵-index列的条件熵
 
 
        if info_gain > best_info_gain:                                  # 获取信息增益最大对应的index列
            best_info_gain = info_gain
            best_column = index
 
    return best_column                      

def split_data(data,index_column,value_column):                         # 给定index列，以及其某个特征值value
    # 筛选满足条件的样本
    filtered_data = data[data[:, index_column] == value_column]         # 获取index列中所有特征值为value的样本
    return np.delete(filtered_data, index_column, axis=1)

def create_decision_tree(data,columns):    # data是numpy格式，包含目标变量
                                           # 目标变量索引。从0开始,且固定为最后一列
                                           # columns为列名称   
    labels   = data[:,-1]
    # 如果数据集中的所有实例属于同一类别，则返回该类别
    if len(set(labels)) == 1:
        return labels[0]
    # 如果没有特征可用，返回实例集中最多的类别
    if data.shape[1] == 1:                 
        return majority_vote(labels)
  
    best_column_index = choose_best_feature(data)     
                                                      # 获取的是信息增益最大的特征列对应的索引
    bestcolumn=columns[best_column_index]             # 依据索引获取特征列名称
    decision_tree = {bestcolumn: {}}                  # 构建当前节点下的决策树
    del(columns[best_column_index] )                  # 列名称列表剔除已选取的最优特征
    
    uniqueVals=set(data[:,best_column_index])         # 获取已选取的最优特征的唯一值

    for value in uniqueVals:                          # 依据已选取的最优特征的唯一值，进行数据切割，并进行下一节点的树构建
        sub_columns = copy.deepcopy(columns)
        decision_tree[bestcolumn][value] = create_decision_tree(split_data(data, best_column_index, value), sub_columns)

    return decision_tree

def print_decision_tree(decision_tree, indent=''):
    # 遍历并输出决策树
    
    # 当前节点为叶节点时，直接输出结果
    if isinstance(decision_tree, str):
        print(indent + decision_tree)
        return
 
    # 当前节点为内部节点时，继续遍历子节点
    for key, value in decision_tree.items():
        print(indent + key + ":")
        if isinstance(value, dict):
            print_decision_tree(value, indent + '  ')
        else:
            print(indent + '  ' + value)

def DecisionTree_predict(decision_tree, Columns, test_sample):
    root_feature = list(decision_tree.keys())[0]  # 获取根节点的特征
    root_dict = decision_tree[root_feature]  # 获取根节点的取值对应的子树
    
    root_feature_index = Columns.index(root_feature)  # 获取根节点特征在特征列表中的索引

    print('root_feature_index:', root_feature_index)
    for value in root_dict:  # 遍历根节点取值对应的子树的所有可能取值
        if test_sample[root_feature_index] == value:  # 如果测试样本的特征值与当前取值相等
            if isinstance(root_dict[value], dict):  # 如果该取值对应的子树还是一个字典（非叶子节点）
                class_label = DecisionTree_predict(root_dict[value], Columns, test_sample)  # 递归向下查询子树
            else:  # 如果该取值对应的子树是一个标签（叶子节点）
                class_label = root_dict[value]  # 将该标签作为分类结果
    return class_label  # 返回分类结果


def createDataSet():    # 创造示例数据
    dataSet=[['青绿','蜷缩','浊响','清晰','凹陷','硬滑','好瓜'],
             ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','好瓜'],
             ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','好瓜'],
             ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','好瓜'],                
             ['青绿','稍蜷','浊响','清晰','稍凹','软粘','好瓜'],               
             ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','好瓜'],                
             ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','好瓜'],
             ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','坏瓜'],
             ['青绿','硬挺','清脆','清晰','平坦','软粘','坏瓜'],
             ['浅白','蜷缩','浊响','模糊','平坦','软粘','坏瓜'],
             ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','坏瓜'],  
             ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','坏瓜'],
             ['乌黑','稍蜷','浊响','清晰','稍凹','软粘','坏瓜'],
             ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','坏瓜']]
    labels = ['色泽','根蒂','敲声','纹理','脐部','触感',"目标"]  #六个特征+1个特征列

    return dataSet,labels

if __name__ == "__main__":
    data,Columns = createDataSet()
    data = np.array(data)
    columns = copy.deepcopy(Columns)
    decision_tree = create_decision_tree(data,columns)
    print_decision_tree(decision_tree) # 输出决策树结果

    testSample=['浅白','硬挺','清脆','清晰','平坦','软粘']  # 待测样本
    class_label = DecisionTree_predict(decision_tree,Columns,testSample)
    print('class_label:', class_label)
