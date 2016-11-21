#ifndef SOFTMAX_H_INCLUDED
#define SOFTMAX_H_INCLUDED

#include<cmath>
template<class T>
class softMax{
public:

    vector<T> weight; //第i类: weight[input_size*i,input_size*(i+1)-1]
    vector<T> thetax; //临时空间
    int input_size,output_size;
    T learningRate; //学习率
    T lambda; //正则项系数
    T eps; //迭代终止的精度

    //input_size: 特征向量维数(加偏置), output_size: 类别数目
    void init(int _input_size,int _output_size){
        input_size=_input_size,output_size=_output_size;
        weight.clear();
        weight.resize(input_size*output_size);
        for(int i=0;i<input_size*output_size;i++)
            weight[i]=0;
    }


    template<typename T1> //calculate exp/sigma(exp)
    vector<T>& softmax(vector<T1> &feature){
        assert(feature.size()==input_size);
        thetax.assign(output_size,0);
        for(int i=0;i<output_size;i++){
            int index=i*input_size;
            for(int j=0;j<input_size;j++)
                thetax[i]+=weight[index+j]*feature[j];
        }
        //trick: exp(x)/sigma(exp(x)) = exp(x-max)/sigma(exp(x-max))
        T thetaxMax=thetax[0],sum=0;
        for(int i=1;i<thetax.size();i++)
            thetaxMax=max(thetaxMax,thetax[i]);
        for(int i=0;i<thetax.size();i++){
            thetax[i]=exp(thetax[i]-thetaxMax);
            sum+=thetax[i];
        }
        for(int i=0;i<thetax.size();i++)
            thetax[i]/=sum;
        return thetax;
    }

    T length(vector<T> &temp){
        T len=0;
        for(int i=0;i<temp.size();i++)
            len+=temp[i]*temp[i];
        return sqrt(len);
    }

    //批梯度
    template<typename T1>
    bool batch_gradient(vector<vector<T1> > &feature,vi &label,int batchSize=100){
        assert(feature.size()==label.size());
        vector<int> index(feature.size(),0);
        for(int i=1;i<index.size();i++) index[i]=i;
        random_shuffle(index.begin(),index.end());
        vector<T> temp(weight.size(),0);
        for(int j=0,k=-1;j<feature.size();j++){
            if(++k==batchSize){
                if(length(temp)<eps) return true;
                for(int i=0;i<weight.size();i++){
                    weight[i]+=learningRate*temp[i]/batchSize+lambda*weight[i];
                    temp[i]=0;
                }
                k=0;
            }
            softmax(feature[index[j]]);
            for(int i=0;i<thetax.size();i++)
                thetax[i]=(i==label[index[j]])-thetax[i];
            for(int g=0;g<output_size;g++){
                int curPos=g*input_size;
                for(int i=0;i<input_size;i++)
                    temp[curPos+i]+=feature[index[j]][i]*thetax[g];
            }
        }
        return false;
    }

    //训练
    template<typename T1>
    void train(vector<vector<T1> > &feature,vi &label,T _learningRate,T _lambda,int maxIter=5,T _eps=1e-6){
        learningRate=_learningRate;
        lambda=_lambda;
        eps=_eps;
        for(int i=0;i<maxIter;i++)
            if(batch_gradient(feature,label)) break;
    }

    int MaxIndex(vector<T> thetax){
        int maxId=0;
        for(int i=1;i<thetax.size();i++)
            if(thetax[i]>thetax[maxId]) maxId=i;
        return maxId;
    }

    //评估测试集准确率
    template<typename T1>
    double predict(vector<vector<T1> > &testSet,vi &label){
        int identifier;
        int correct=0;
        for(int i=0;i<testSet.size();i++){
            identifier=MaxIndex(softmax(testSet[i]));
            correct+=identifier==label[i];
            //printf("Test %d: predict: %d, fact: %d\n",i+1,identifier,label[i]);
        }
        return correct*1.0/label.size();
    }

    void show(string path){
        FILE* file;
        if((file=fopen(path.c_str(),"w"))==NULL){
            printf("Can't open the %s.\n",path.c_str());
            return;
        }
        for(int i=0;i<output_size;i++){
            fprintf(file,"%d %d\n",28,28);
            int index=i*input_size;
            for(int j=0;j<input_size;j++)
                fprintf(file,"%.3f ",weight[index+j]);
            fprintf(file,"\n");
        }
        fclose(file);
    }

};

#endif // SOFTMAX_H_INCLUDED
