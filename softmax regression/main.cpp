#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include"mnist.h"
#include"softmax.h"
using namespace std;

int main()
{
    v2uc image,test;
    vi label,testlabel;

    mnist mst;

    //载入训练集
    mst.load(image,label,mst.train_image,mst.train_label);

    softMax<float> sm;
    sm.init(image[0].size(),10);
    sm.train(image,label,0.003,0.001);

    //载入测试集
    mst.load(test,testlabel,mst.test_image,mst.test_label);

    printf("Correct Rate: %.3f%%\n",sm.predict(image,label)*100);
    printf("Correct Rate: %.3f%%\n",sm.predict(test,testlabel)*100);
    sm.show("weight.txt");
	return 0;
}
