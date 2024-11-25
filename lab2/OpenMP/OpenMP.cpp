#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
//#include <sstream>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

int maxIter = 1000;

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


Mat MyBlurV1(Mat image, std::vector<std::vector<float>> kernel, double* procTime) {

    int n = kernel.size();
    int border = n / 2;
    cv::Mat blurImage(image.rows - border * 2, image.cols, image.type());
    //int chunk_size=50;
#pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        double sumTime = 0;
        for (int iter = 0; iter < maxIter; iter++) {
            double startTime = omp_get_wtime();
//#pragma omp for schedule(static)
//#pragma omp for schedule(dynamic)
//#pragma omp for schedule(guided)
//#pragma omp for schedule(static, chunk_size)
//#pragma omp for schedule(dynamic, chunk_size)
//#pragma omp for schedule(guided, chunk_size)
#pragma omp for schedule(dynamic)
            for (int r = border; r < image.rows - border; r++) {
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
            }
            double endTime = omp_get_wtime();
            sumTime += endTime - startTime;
        }
        sumTime = sumTime / maxIter;
        procTime[thread_id] = sumTime;
    }

    return blurImage;
}

Mat MyBlurV2(Mat image, std::vector<std::vector<float>> kernel, double* procTime) {

    int n = kernel.size();
    int border = n / 2;
    cv::Mat blurImage(image.rows - border * 2, image.cols, image.type());

    int step = image.cols / 4;
    if (image.cols - step * 3 > step)
        step++;
#pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        double sumTime = 0;
        int start_col= thread_id * step;
        int end_col = start_col + step;
        if (thread_id == 3)
            end_col = image.cols;
        for (int iter = 0; iter < maxIter; iter++) {
            double startTime = omp_get_wtime();
            for (int r = border; r < image.rows - border; r++) {
                for (int c = start_col; c < end_col; c++) {
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
            }
            double endTime = omp_get_wtime();
            sumTime += endTime - startTime;
        }
        sumTime = sumTime / maxIter;
        procTime[thread_id] = sumTime;
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
#define useSimple 0
int main()
{


    const std::string matrixPath = "mas.txt";
    //"car10.jpg","car20.jpg","car30.jpg", "car64.jpg",  "car120.jpg", "car200.jpg","car300.jpg","car400.jpg","car550.jpg","car800.jpg"
    std::string fileName = "car64.jpg";
   

    Mat img = imread(fileName);


    std::vector<std::vector<float>> GaussKernel = readKernel(matrixPath);
    int n = GaussKernel.size();
    int border = n / 2;

    Mat output_image;
    double Times[4]{ 0,0,0,0 };
    for (int iterImg = 0; iterImg < 2; iterImg++) {
        double* tmpTime=new double[4];
        // Создайте новое изображение с продублированной строкой сверху

        cv::Mat duplicatedImage;
        cv::repeat(img.row(0), border, 1, duplicatedImage);

        // Добавить дубликаты сверху
        cv::vconcat(duplicatedImage, img, duplicatedImage);

        cv::Mat botDublicate;
        cv::repeat(img.row(img.rows - 1), border, 1, botDublicate);

        // Добавьте дубликаты снизу
        cv::vconcat(duplicatedImage, botDublicate, duplicatedImage);

        // imwrite("test.jpg", duplicatedImage);
#if useSimple 1
        output_image = MyBlurV2(duplicatedImage, GaussKernel, tmpTime);
#else 
        output_image = MyBlurV1(duplicatedImage, GaussKernel, tmpTime);
#endif
        for (int i = 0; i < 4; i++) {
            Times[i] += tmpTime[i];
        }
        delete tmpTime;
        img = output_image;

    }
    double fullTime = Times[0];
    for (int i = 0; i < 4; i++) {
        cout << "Time" << i << " " << fixed << setprecision(6)<< Times[i]<<endl;
        if (fullTime < Times[i])
            fullTime = Times[i];
    }


    std::ofstream ftime("time_"+std::to_string(useSimple) + fileName + ".txt");
    ftime << fixed << setprecision(6) << fullTime << endl;
    ftime.close();
    std::cout << "Time: " << fixed << setprecision(6) << fullTime << endl;

#if useSimple 1
    std::string outFilename = "ompV2_" + fileName;
#else 
    std::string outFilename = "ompV1_" + fileName;
#endif
    imwrite(outFilename, output_image);
    return 0;
}