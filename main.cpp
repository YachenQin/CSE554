//
//  main.cpp
//  cv554
//
//  Created by 秦雅琛 on 2018/10/25.
//  Copyright © 2018 秦雅琛. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;


//use Scharr filter
Mat energy_function_Scharr(Mat image){
    
    Mat dx, dy;
    Mat output;
    
    Mat gray;
    Mat blur;
    GaussianBlur(image, blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(blur, gray, CV_BGR2GRAY);
    
    //calculate gradient on direction x
    Scharr(gray, dx, CV_64F,  1, 0);
    //calculate gradient on direction y
    Scharr(gray, dy, CV_64F, 0, 1);
    
    magnitude(dx,dy, output);
    
    double min_value, max_value, Z;
    
    minMaxLoc(output, &min_value, &max_value);
    
    Z = 1/max_value * 255;
    
    output = output * Z;
    
    output.convertTo(output, CV_8U);
    
    return output;
}

//use sobel filter
Mat energy_function_sobel(Mat image){
    Mat dx, dy;
    Mat output;
    
    Mat gray;
    Mat blur;
    GaussianBlur(image, blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(blur, gray, CV_BGR2GRAY);
    Sobel(gray, dx, CV_64F, 1, 0);
    Sobel(gray, dy, CV_64F, 0, 1);
    magnitude(dx,dy, output);
    
    magnitude(dx,dy, output);
    
    double min_value, max_value, Z;
    
    minMaxLoc(output, &min_value, &max_value);
    
    Z = 1/max_value * 255;
    
    output = output * Z;
    
    output.convertTo(output, CV_8U);
    
    return output;
}



int* find_seam(Mat &image,int time){
    int height = image.rows, width = image.cols;
    
    int dp[height][width];
    for(int col = 0; col < width; col++){
        dp[0][col] = (int)image.at<uchar>(0,col);
    }
    
    for(int row = 1; row < height ; row++){
        for(int col = 0; col < width; col++){
            if (col == 0)
                dp[row][col] = min(dp[row-1][col+1], dp[row-1][col]);
            else if (col == width-1)
                dp[row][col] = min(dp[row-1][col-1], dp[row-1][col]);
            else
                dp[row][col] = min({dp[row-1][col-1], dp[row-1][col], dp[row-1][col+1]});
            dp[row][col] += (int)image.at<uchar>(row,col);
        }
    }
    
    int min_value = INT_MAX;
    int min_index = -1;
    
    map<int,vector<int>> dic;
    for(int col = 0; col < width; col++){
        dic[dp[height-1][col]].push_back(col);
        if (dp[height-1][col] < min_value) {
            min_value = dp[height - 1][col];
            min_index = col;
        }
    }
    
    if(time>1){
        time=rand() % dic.size();
        map<int,vector<int>>::iterator it=dic.begin();
        for(int i=time;i>0;i--){
            it++;
        }
        min_value=it->first;
        int length=it->second.size();
        int a=rand() % length;
        min_index=it->second[a];
    }
    
    int path[height];
    Point pos(height-1,min_index);
    path[pos.x] = pos.y;
    
    while (pos.x != 0){
        int value = dp[pos.x][pos.y] - (int)image.at<uchar>(pos.x,pos.y);
        int r = pos.x, c = pos.y;
        if (c == 0){
            if (value == dp[r-1][c+1])
                pos = Point(r-1,c+1);
            else
                pos = Point(r-1,c);
        }
        else if (c == width-1){
            if (value == dp[r-1][c-1])
                pos = Point(r-1,c-1);
            else
                pos = Point(r-1,c);
        }
        else{
            if (value == dp[r-1][c-1])
                pos = Point(r-1,c-1);
            else if (value == dp[r-1][c+1])
                pos = Point(r-1,c+1);
            else
                pos = Point(r-1,c);
        }
        path[pos.x] = pos.y;
    }
    
    return path;
}


void remove_pixels(Mat& image, Mat& output, int *seam){
    for(int row = 0; row < image.rows; row++ ) {
        for (int col = 0; col < image.cols; col++){
//            if(col==seam[row]){
//                output.at<Vec3b>(row,col)[0] = 255;
//                output.at<Vec3b>(row,col)[1] = 255;
//                output.at<Vec3b>(row,col)[2] = 255;
//            }
            if (col >= seam[row])
                output.at<Vec3b>(row,col) = image.at<Vec3b>(row,col+1);
            else
                output.at<Vec3b>(row,col) = image.at<Vec3b>(row,col);
        }
    }
}

void add_pixels(Mat& image, Mat& output, int *seam){
    for(int row = 0; row < output.rows; row++ ) {
        for (int col = 0; col < output.cols; col++){
            if (col >= seam[row]+1){
                output.at<Vec3b>(row,col) = image.at<Vec3b>(row,col-1);
            }
            //else if(col == seam[row]+1){
              //  output.at<Vec3b>(row,col)=output.at<Vec3b>(row,col-1);
            //}
            else
                output.at<Vec3b>(row,col) = image.at<Vec3b>(row,col);
        }
    }
}

void rot90(Mat &matImage, int rotflag){
    if (rotflag == 1){
        transpose(matImage, matImage);
        flip(matImage, matImage,1);
    } else {
        transpose(matImage, matImage);
        flip(matImage, matImage,0);
    }
}

void modify_seam(Mat& image,int energy,int operation, char orientation = 'v'){
    if (orientation == 'h')
        rot90(image,1);
    int H = image.rows, W = image.cols;
    Mat eimage;
    if(energy==1){
      eimage=energy_function_sobel(image);
    }
    else{
        eimage=energy_function_Scharr(image);
    }
    Mat result;
    if(operation==0){
        int* seam = find_seam(eimage,1);
        Mat output(H,W-1, CV_8UC3);
        //Mat output(H,W, CV_8UC3);
        remove_pixels(image, output, seam);
        result=output;
    }
    
   else{
       int* seam = find_seam(eimage,2);
       Mat output(H,W+1, CV_8UC3);
       add_pixels(image, output, seam);
       result=output;
    }
    
    if (orientation == 'h')
        rot90(result,2);
    image = result;
}

void modify_image(Mat& image, int new_cols, int new_rows, int width, int height,int operation,int energy){
    cout << endl << "Processing image..." << endl;
    if(operation==0){
        for(int i = 0; i < width - new_cols; i++){
            modify_seam(image,energy,operation, 'v');
        }
        for(int i = 0; i < height - new_rows; i++){
            modify_seam(image,energy,operation, 'h');
        }
    }
    else{
        for(int i = 0; i < new_cols - width ; i++){
             modify_seam(image,energy,operation, 'v');
        }
        for(int i = 0; i < new_rows - height ; i++){
             modify_seam(image,energy,operation, 'h');
        }
    }
}


void realTime(Mat& image){
    cout << "S ARROW: Shrink vertically" << endl;
    cout << "A ARROW: Shrink horizontally" << endl;
    cout << "D ARROW: Expand horizontally" << endl;
    cout << "W ARROW: Expand vertically" << endl;
    
    cout << "q: Quit" << endl;
    
    int key;
    while(1) {
        namedWindow("Display window", WINDOW_AUTOSIZE);
        imshow("Display window", image);
        key = waitKey(0);
        if (key == 'q')
            break;
        else if (key == 'a'){
             modify_seam(image,1,0,'v');
        }
        else if (key == 's'){
            modify_seam(image,1,0,'h');
        }
        else if (key == 'd'){
            modify_seam(image,1,1,'v');
        }
        else if (key == 'w'){
            modify_seam(image,1,1,'h');
        }
    }
}



int main() {
    Mat image;
    string file;//="/Users/qinyachen/Documents/554/final/inputA.jpg";
    int energy=1;
    int operation=1;
    cout<<"provide the file path and name you want to modified:"<<endl;
    cin>>file;
    image = imread(file, 1);
//    Mat output=energy_function_sobel(image);
//    imshow("Display window", output);
//    waitKey(0);
    
    int real;
    cout<< " want real time? (1 for yes, 0 for no)"<<endl;
    cin >> real;
    if(real == 1){
        realTime(image);
    }
    if(real == 0 )
    {
        cout <<"What energy function you want to use? ( 1 for Sobel, 2 for Scharr)"<<endl;
        cin>> energy;
        cout <<"What operation you want to provide? ( 1 for expand, 0 for shrink)"<<endl;
        cin>> operation;

        cout << "The size of the image is: (" << image.cols << ", " << image.rows << ")" << endl;

        int new_rows, new_cols;
        cout << "Enter new width: ";
        cin >> new_cols;
        cout << "Enter new height: ";
        cin >> new_rows;

        modify_image(image, new_cols, new_rows,image.cols,image.rows,operation,energy);

        cout << "Done!" << endl;
        int save;
        cout << "save image? (1 for yes,0 for no)"<< endl;
        cin>>save;
        if(save==1){
            cout << "Input path and name " << endl;
            string output;
            cin >> output;
            imwrite("/Users/qinyachen/Documents/554/final/"+output+".jpg",image);
            imshow("Display window", image);
            waitKey(0);
        }
        else if(save==0){
            imshow("Display window", image);
            waitKey(0);
        }
        else{
            cout<<"wrong command"<<endl;
        }
    }
    else{
      return 0;
    }

    return 0;
}
