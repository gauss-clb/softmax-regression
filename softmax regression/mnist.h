#ifndef MNIST_H_INCLUDED
#define MNIST_H_INCLUDED

#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include<cstdlib>
#include<cassert>
using namespace std;

/**
 * THE MNIST DATABASE of handwritten digits
 *
 * Author: GAUSS_CLB
 * Date: 2016/9/5
 *
*/
typedef vector<vector<unsigned char> > v2uc;
typedef vector<int> vi;
class mnist{
public:
    string train_image,train_label;
    string test_image,test_label;
    mnist():
        train_image("train-images.idx3-ubyte"),
        train_label("train-labels.idx1-ubyte"),
        test_image("t10k-images.idx3-ubyte"),
        test_label("t10k-labels.idx1-ubyte"){}

    /**
      *imageName: imageFile
      *labelName: labelFile
    */

    void load(v2uc &image,vi &label,string imageName,string labelName){
        for(int i=0;i<image.size();i++) image[i].clear();
        image.clear();
        label.clear();
        int magicNumber;
        int imageNumber;
        int labelNumber;
        int rows;
        int cols;
        FILE* file;
        if((file=fopen(imageName.c_str(),"rb"))==NULL){
            printf("Can't open the %s.\n",imageName.c_str());
            return;
        }
        fread(&magicNumber,sizeof(int),1,file);
        fread(&imageNumber,sizeof(int),1,file);
        fread(&rows,sizeof(int),1,file);
        fread(&cols,sizeof(int),1,file);
        magicNumber=Big2Little(magicNumber);
        imageNumber=Big2Little(imageNumber);
        rows=Big2Little(rows);
        cols=Big2Little(cols);
        assert(magicNumber==2051);
        assert(imageNumber>=0);
        assert(rows==28);
        assert(cols==28);
        image.resize(imageNumber);
        unsigned char pixel;
        for(int i=0;i<imageNumber;i++){
            image[i].resize(rows*cols+1);
            image[i][0]=1;
            for(int j=1;j<=rows*cols;j++){
                fread(&pixel,1,1,file);
                image[i][j]=pixel;
            }
        }
        fclose(file);
        if((file=fopen(labelName.c_str(),"rb"))==NULL){
            printf("Can't open the %s.\n",labelName.c_str());
            return;
        }
        fread(&magicNumber,sizeof(int),1,file);
        fread(&labelNumber,sizeof(int),1,file);
        magicNumber=Big2Little(magicNumber);
        labelNumber=Big2Little(labelNumber);
        assert(magicNumber==2049);
        assert(imageNumber==labelNumber);
        int identifier=0;
        label.resize(labelNumber);
        for(int i=0;i<labelNumber;i++){
            fread(&identifier,1,1,file);
            label[i]=identifier;
        }
        fclose(file);
    }

    int Big2Little(int value){ //Big Endian -> Little Endian
        return (value&0x000000FF)<<24|
                (value&0x0000FF00)<<8|
                (value&0x00FF0000)>>8|
                (value&0xFF000000)>>24;
    }

};

#endif // MNIST_H_INCLUDED

