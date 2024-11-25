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

int maxIter = 1;


Mat MyBlur(Mat img, std::vector<std::vector<float>> kernel) {

    int n = kernel.size();
    int border = n / 2;

    cv::Mat image = img.clone();


        // прогоняем фильтр первый раз
        #pragma omp parallel for
        for (int r = 0; r < img.rows; r++)
            for (int c = 0; c < img.cols; c++) {

                Vec3b& color = img.at<Vec3b>(r, c);
                float red = 0, green = 0, blue = 0;

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
                        if (_row < 0) {
                            _row = 0;
                        }
                        else if (_row > image.rows - 1) {
                            _row = image.rows - 1;
                        }
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

    cv::Rect roi(0, border, image.cols, image.rows - border * 2);

    Mat blurImage = image(roi);//выходное без карйних пикселей.

    //imwrite("crop.jpg", blurImage);
    // прогоняем фильтр второй раз
    #pragma omp parallel for
    for (int r = border; r < image.rows - border; r++)
        for (int c = 0; c < image.cols; c++) {
            Vec3b& color = image.at<Vec3b>(r, c);
            float red = 0, green = 0, blue = 0;

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
            blurImage.at<Vec3b>(r - border, c) = color;
        }

    return blurImage;
}

Mat MyBlurV2(Mat img, std::vector<std::vector<float>> kernel) {

    int n = kernel.size();
    int border = n / 2;

    cv::Mat image = img.clone();


    // прогоняем фильтр первый раз
    int step = img.rows / 4;
#pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        int start_row = thread_id * step;
        int end_row = (thread_id == 3) ? img.rows : (start_row + step);


      
            for (int r = start_row; r < end_row; r++)
                for (int c = 0; c < img.cols; c++) {

                    Vec3b& color = img.at<Vec3b>(r, c);
                    float red = 0, green = 0, blue = 0;

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
                            if (_row < 0) {
                                _row = 0;
                            }
                            else if (_row > image.rows - 1) {
                                _row = image.rows - 1;
                            }
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
    }
    //imwrite("tmp.jpg", image);

    cv::Rect roi(0, border, image.cols, image.rows - border * 2);

    Mat blurImage = image(roi);//выходное без карйних пикселей.

    //imwrite("crop.jpg", blurImage);
    // 
    step = (img.rows - 2 * border) / 4;
    // прогоняем фильтр второй раз
#pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        int start_row = thread_id * step + border;
        int end_row = (thread_id == 3) ? (img.rows - border) : (start_row + step);

        for (int r = start_row; r < end_row; r++)
            for (int c = 0; c < image.cols; c++) {
                Vec3b& color = image.at<Vec3b>(r, c);
                float red = 0, green = 0, blue = 0;

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
                blurImage.at<Vec3b>(r - border, c) = color;
            }
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
//    const long n = 10000;
//    int block_n = 100;
//    double pi;
//    double time;
//
//    // 1. Static
//    pi = 1;
//    time = omp_get_wtime();
//#pragma omp parallel num_threads(4) shared(pi,n)
//#pragma omp for reduction(*:pi) schedule(static,block_n)
//    for (long int i = 1; i < n; i++)
//        pi *= 4.0 * i * i / (2.0 * i - 1) / (2.0 * i + 1);
//    printf("Type Static. Time = %0.6f. Pi = %0.6f\n", omp_get_wtime() - time, 2 * pi);
//
//    // 2. Dynamic
//    pi = 1;
//    time = omp_get_wtime();
//#pragma omp parallel num_threads(4) shared(pi,n)
//#pragma omp for reduction(*:pi) schedule(dynamic,block_n)
//    for (long int i = 1; i < n; i++)
//        pi *= 4.0 * i * i / (2.0 * i - 1) / (2.0 * i + 1);
//    printf("Type Dynamic. Time = %0.6f. Pi = %0.6f\n", omp_get_wtime() - time, 2 * pi);
//
//    // 3. Guided
//    pi = 1;
//    time = omp_get_wtime();
//#pragma omp parallel num_threads(4) shared(pi,n)
//#pragma omp for reduction(*:pi) schedule(guided,block_n)
//    for (long int i = 1; i < n; i++)
//        pi *= 4.0 * i * i / (2.0 * i - 1) / (2.0 * i + 1);
//    printf("Type Guided. Time = %0.6f. Pi = %0.6f\n", omp_get_wtime() - time, 2 * pi);
//
//    // 4. Runtime
//    pi = 1;
//    time = omp_get_wtime();
//#pragma omp parallel num_threads(4) shared(pi,n)
//#pragma omp for reduction(*:pi) schedule(runtime)
//    for (long int i = 1; i < n; i++)
//        pi *= 4.0 * i * i / (2.0 * i - 1) / (2.0 * i + 1);
//    printf("Type Runtime. Time = %0.6f. Pi = %0.6f\n", omp_get_wtime() - time, 2 * pi);
//
//    return 0;
//}

    const std::string matrixPath = "mas.txt";
    //"car10.jpg","car20.jpg","car30.jpg", "car64.jpg",  "car120.jpg", "car200.jpg","car300.jpg","car400.jpg","car550.jpg","car800.jpg"
    std::string fileName = "car10.jpg";
    std::string outFilename = "out_" + fileName;


   /* std::cout << "posl do" << endl;
    #pragma omp parallel
    {
        printf("paral\n");
    }
    std::cout << "posl posle" << std::endl;*/

    Mat img = imread(fileName);


    std::vector<std::vector<float>> GaussKernel = readKernel(matrixPath);
    int n = GaussKernel.size();
    int border = n / 2;

#if useSimple 1
    Mat output_image;
    double fullTime = 0;
    for (int iter = 0; iter < maxIter; iter++) {
        auto start = steady_clock::now();
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

        output_image = MyBlur(duplicatedImage, GaussKernel);
        auto end = steady_clock::now();
        double time = (double)duration_cast<microseconds>(end - start).count();
        fullTime += time;
    }
    fullTime = fullTime / pow(10, 6) / maxIter;

    std::ofstream ftime("time_" + fileName + ".txt");
    ftime << fixed << setprecision(6) << fullTime << endl;
    ftime.close();
    std::cout << "Time: " << fixed << setprecision(6) << fullTime << endl;

    imwrite(outFilename, output_image);
#else

    
    
    // Создайте новое изображение с продублированной строкой сверху
    cv::Mat duplicatedImage;
    cv::repeat(img.row(0), border, 1, duplicatedImage);

    // Добавить дубликаты сверху
    cv::vconcat(duplicatedImage, img, duplicatedImage);

    cv::Mat botDublicate;
    cv::repeat(img.row(img.rows - 1), border, 1, botDublicate);

    // Добавьте дубликаты снизу
    cv::vconcat(duplicatedImage, botDublicate, duplicatedImage);

    double stratTime = omp_get_wtime();
    Mat output_image;
    for (int iter = 0; iter < maxIter; iter++) {
        output_image = MyBlurV2(duplicatedImage, GaussKernel);
    }
    double endTime = omp_get_wtime();
    cout << endTime - stratTime;
    //std::ofstream ftime("time_" + fileName + ".txt");
   // ftime << fixed << setprecision(6) << fullTime << endl;
   // ftime.close();
   // std::cout << "Time: " << fixed << setprecision(6) << fullTime << endl;

    imwrite(outFilename, output_image);
#endif

    

    return 0;
}