具体实验流程为
1、pytorch生成pth模型。
2、pth模型转换为onnx结构。
3、qnn工具将onnx模型转换为手机硬件支持格式。
4、手机上调用模型运行生成结果。

首先需要一个实验模型：
我这里直接用pytorch的mnist模型做相关实验，因为它简单，会额外引入网络不兼容的概率低。
相关代码和模型训练，参考https://zhuanlan.zhihu.com/p/137571225
最终得到训练好的模型：model.pth

代码onnx_convert.py将pth模型转换为onnx格式,得到模型Network.onnx
代码onnx_test.py用来验证转换后onnx模型输出是否正常。

按照如下教程安装qnn相关工具和使用该工具进行onnx模型转换
https://zhuanlan.zhihu.com/p/641013796
https://zhuanlan.zhihu.com/p/671145477
https://zhuanlan.zhihu.com/p/671160585
资料：https://blog.csdn.net/weixin_38498942/article/details/135278345 介绍了qnn工具常用命令细节。
注意：xxx.tofile('xx.raw') 用来生成qnn测试时候需要的输入文件。


source /opt/qcom/aistack/qnn/2.19.2.240210/bin/envsetup.sh
