#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
//#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std::chrono;
using namespace std;



Mat MyBlur(Mat img, std::vector<std::vector<float>> kernel) {

    int n = kernel.size(); //??
    int border = n / 2;
    
    cv::Mat image= img.clone();

    // прогоняем фильтр первый раз
    for (int r = border; r < img.rows - border; r++)
        for (int c = border; c < img.cols-border; c++) {
       
            Vec3b& color = img.at<Vec3b>(r, c);
            float red = 0, green = 0, blue = 0;

            for (int k_row = -border, i = 0; k_row <= border; k_row++, i++) {
                for (int k_col = -border, j = 0; k_col <= border; k_col++, j++) {

                    int _row = r + k_row;
                    int _col = c + k_col;

                    color = img.at<Vec3b>(_row, _col);//получаем пиксель

                    // Синий
                    blue += color[0] * kernel[i][j];
                    // Зеленый
                    green += color[1] * kernel[i][j];
                    // Красный
                    red += color[2] * kernel[i][j];
                }
            }
            // Записываем пиксель
            color = img.at<Vec3b>(r, c);
            color[0] = blue;
            color[1] = green;
            color[2] = red;
            image.at<Vec3b>(r, c) = color;
        }

    //imwrite("tmp.jpg", image);

    cv::Rect roi(border, border, image.cols-border*2, image.rows - border*2);

    Mat blurImage = image(roi);//выходное без карйних пикселей.

    //imwrite("crop.jpg", blurImage);
    // прогоняем фильтр второй раз
    for (int r = border; r < image.rows-border; r++)
        for (int c = border; c < image.cols-border; c++) {       
            Vec3b& color = image.at<Vec3b>(r, c); 
            float red = 0, green = 0, blue = 0;

            for (int k_row = -border, i = 0; k_row <= border; k_row++, i++) {
                for (int k_col = -border, j = 0; k_col <= border; k_col++, j++) {

                    int _row = r + k_row;
                    int _col = c + k_col;

                    color = image.at<Vec3b>(_row, _col);//получаем пиксель

                    // Синий
                    blue += color[0] * kernel[i][j];
                    // Зеленый
                    green += color[1] * kernel[i][j];
                    // Красный
                    red += color[2] * kernel[i][j];
                }
            }
            // Записываем пиксель
            color = image.at<Vec3b>(r, c);
            color[0] = blue;
            color[1] = green;
            color[2] = red;
            blurImage.at<Vec3b>(r-border, c-border) = color;
        }
    return blurImage;
}

std::vector<std::vector<float>> readKernel(std::string filepath) {

    std::ifstream file(filepath); 
    std::vector<std::vector<float>> kernel;
    
    int n;
    file >> n;

    for (int i = 0; i < n; i++) {
        std::vector<float> row;
        float v;
        for (int j = 0; j < n; j++) {
            file >> v;
            row.push_back(v);
        }
        kernel.push_back(row); 
    }
    file.close();
    
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++)
    //        cout << kernel[i][j] << " ";
    //    cout << endl;
    //}


    return kernel;
}

int main()
{
    const std::string matrixPath = "mas.txt";
    //"car30.jpg", "car64.jpg",  "car120.jpg", "car200.jpg","car300.jpg","car550.jpg","car800.jpg","car1000.jpg"
    std::string fileName = "car3.jpg";
    std::string outFilename = "out_" + fileName;
    int maxIter = 1;


    Mat image = imread(fileName);
    Mat output_image;

    std::vector<std::vector<float>> GaussKernel = readKernel(matrixPath);
    int n = GaussKernel.size();
    int border = n / 2;

   
    double fullTime = 0;
    for (int iter = 0; iter < maxIter; iter++) {
        auto start = steady_clock::now();
        // Создайте новое изображение с продублированной строкой сверху
        
        
        cv::Mat duplicatedImage;
        cv::repeat(image.row(0), border, 1, duplicatedImage);

        // Добавить дубликаты сверху
        cv::vconcat(duplicatedImage, image, duplicatedImage);

        cv::Mat botDublicate;
        cv::repeat(image.row(image.rows - 1), border, 1, botDublicate);

        // Добавьте дубликаты снизу
        cv::vconcat(duplicatedImage, botDublicate, duplicatedImage);


        cv::Mat leftDuble;
        cv::repeat(duplicatedImage.col(0), 1, border, leftDuble);

         // Добавить дубликаты слева
        cv::hconcat(leftDuble, duplicatedImage, duplicatedImage);

        cv::Mat rightDubl;
        cv::repeat(duplicatedImage.col(duplicatedImage.cols - 1), 1, border, rightDubl);

         // Добавить дубликаты справа
        cv::hconcat(duplicatedImage, rightDubl, duplicatedImage);

        imwrite("test.jpg", duplicatedImage);

        output_image = MyBlur(duplicatedImage, GaussKernel);
        auto end = steady_clock::now();
        double time = (double)duration_cast<microseconds>(end - start).count() / maxIter / pow(10, 6);
        fullTime += time;
    }

    std::ofstream ftime("time_"+ fileName+".txt");
    ftime<<fixed<<setprecision(6)<< fullTime<<endl;
    ftime.close();
    std::cout << "Time: "<< fixed << setprecision(6)<< fullTime <<endl;

    imwrite(outFilename, output_image);

    return 0;
}