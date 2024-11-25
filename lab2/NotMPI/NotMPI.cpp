#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
//#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std::chrono;
using namespace std;



Mat MyBlur(Mat image, std::vector<std::vector<float>> kernel) {
    int n = kernel.size();
    int border = n / 2;
    cv::Mat blurImage(image.rows - border * 2, image.cols, image.type());

    // прогоняем фильтр 
    for (int r = border; r < image.rows - border; r++)
        for (int c = 0; c < image.cols; c++) {
            uchar red = 0, green = 0, blue = 0;

            for (int k_row = -border, i = 0; k_row <= border; k_row++, i++) {
                for (int k_col = -border, j = 0; k_col <= border; k_col++, j++) {

                    int _row = r + k_row;
                    int _col = c + k_col;

                    if (_col < 0) {
                        _col = 0;
                    }
                    else if (_col > image.cols - 1) {
                        _col = image.cols - 1;
                    }

                    cv::Vec3b pixel = image.at<cv::Vec3b>(_row, _col);
                    blue += pixel[0] * kernel[i][j];
                    green += pixel[1] * kernel[i][j];
                    red += pixel[2] * kernel[i][j];
                }
            }

            // Установите новое значение цвета пикселя (например, синий)
            cv::Vec3b newColor(blue, green, red); // (B, G, R)
            blurImage.at<cv::Vec3b>(r - border, c) = newColor;
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
    


    return kernel;
}

int main()
{
    const std::string matrixPath = "mas.txt";
    //"car10.jpg","car20.jpg","car30.jpg", "car64.jpg",  "car120.jpg", "car200.jpg","car300.jpg","car400.jpg","car550.jpg","car800.jpg"
    std::string fileName = "car64.jpg";
    std::string outFilename = "out_" + fileName;
    int maxIter = 1000;


    Mat image = imread(fileName);
    Mat output_image;

    std::vector<std::vector<float>> GaussKernel = readKernel(matrixPath);
    int n = GaussKernel.size();
    int border = n / 2;

   
    double fullTime = 0;
    for (int iterImg = 0; iterImg < 2; iterImg++) {
        // Создайте новое изображение с продублированной строкой сверху
        cv::Mat duplicatedImage;
        cv::repeat(image.row(0), border, 1, duplicatedImage);

        // Добавить дубликаты сверху
        cv::vconcat(duplicatedImage, image, duplicatedImage);

        cv::Mat botDublicate;
        cv::repeat(image.row(image.rows - 1), border, 1, botDublicate);

        // Добавьте дубликаты снизу
        cv::vconcat(duplicatedImage, botDublicate, duplicatedImage);

        for (int iter = 0; iter < maxIter; iter++) {
            auto start = steady_clock::now();
            
            // imwrite("test.jpg", duplicatedImage);

            output_image = MyBlur(duplicatedImage, GaussKernel);
            auto end = steady_clock::now();
            double time = (double)duration_cast<microseconds>(end - start).count();
            fullTime += time;
        }
        image = output_image;
    }
    fullTime = fullTime / pow(10, 6) / maxIter;

    std::ofstream ftime("time_"+ fileName+".txt");
    ftime<<fixed<<setprecision(6)<< fullTime<<endl;
    ftime.close();
    std::cout << "Time: "<< fixed << setprecision(6)<< fullTime <<endl;

    imwrite(outFilename, output_image);

    return 0;
}