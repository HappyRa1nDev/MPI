#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include<opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;
using namespace cv;

int fix(int value, int min, int max) {
    return value < min ? min : value > max ? max : value;
}


Mat MyBlur(Mat srcImg, std::vector<std::vector<float>> kernel) {

    int n = kernel.size(); //??
    int border = n / 2;

    cv::Mat image = srcImg.clone();

    // прогоняем фильтр первый раз
    for (int r = border; r < srcImg.rows - border; r++)
        for (int c = border; c < srcImg.cols - border; c++) {

            Vec3b& color = srcImg.at<Vec3b>(r, c);
            float red = 0, green = 0, blue = 0;

            for (int k_row = -border, i = 0; k_row <= border; k_row++, i++) {
                for (int k_col = -border, j = 0; k_col <= border; k_col++, j++) {

                    int _row = r + k_row;
                    int _col = c + k_col;

                    color = srcImg.at<Vec3b>(_row, _col);//получаем пиксель

                    // Синий
                    blue += color[0] * kernel[i][j];
                    // Зеленый
                    green += color[1] * kernel[i][j];
                    // Красный
                    red += color[2] * kernel[i][j];
                }
            }
            // Записываем пиксель
            color = srcImg.at<Vec3b>(r, c);
            color[0] = blue;
            color[1] = green;
            color[2] = red;
            image.at<Vec3b>(r, c) = color;
        }

    //imwrite("tmp.jpg", image);

    cv::Rect roi(border, border, image.cols - border * 2, image.rows - border * 2);

    Mat blurImage = image(roi);//выходное без карйних пикселей.

    //imwrite("crop.jpg", blurImage);
    // прогоняем фильтр второй раз
    for (int r = border; r < image.rows - border; r++)
        for (int c = border; c < image.cols - border; c++) {
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
            blurImage.at<Vec3b>(r - border, c - border) = color;
        }
    return blurImage;
}


std::vector<std::vector<float>> readMatrix(std::string filepath) {

    std::ifstream file(filepath); // открыть файл для чтения
    std::string line;
    std::vector<std::vector<float>> matrix;

    while (std::getline(file, line)) { // читать построчно из файла
        std::vector<float> row;
        std::istringstream iss(line);
        float value;
        while (iss >> value) { // читать числа из строки
            row.push_back(value);
        }
        matrix.push_back(row); // добавить строку в матрицу
    }
    file.close();

    //// вывести матрицу на экран для проверки
    //for (const auto& row : matrix) {
    //    for (const auto& value : row) {
    //        std::cout << value << " ";
    //    }
    //    std::cout << std::endl;
    //}

    return matrix;
}

std::vector<std::vector<float>> GaussMatrix;

int main(int argc, char** argv) {


    const std::string matrixPath = "mas.txt";
    std::vector<std::string> resolutions({ "64.jpg", "128.jpg", "300.jpg", "574.jpg", "900.jpg" });
    std::string resolution = "car.jpg";
    std::string imagePath = resolution;
    std::string output = "answ_" + resolution;
    int repeats = 1;

    GaussMatrix = readMatrix(matrixPath);
    int border = GaussMatrix.size() / 2;
    cout << border << "keeeeek";

    Mat original_image = imread(imagePath);


    MPI_Init(&argc, &argv);



    const int original_n= original_image.cols;
    const int original_m= original_image.rows;
    cout << "m" << original_m<<endl;
    int step = original_m / 4;
    if (original_m - step * 3 > step) {
        cout << "kek" << endl;
        step++;
    }
    //int step = original_step;
    cout << step<<endl;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Создаю новое изображение дублями сверху и снизу
    cv::Mat img;
    cv::repeat(original_image.row(0), border, 1, img);

    // Добавить дубликаты сверху
    cv::vconcat(img, original_image, img);

    cv::Mat botDubl;
    cv::repeat(original_image.row(original_image.rows - 1), border, 1, botDubl);

    // Добавьте дубликаты снизу
    cv::vconcat(img, botDubl, img);


    cv::Mat leftDuble;
    cv::repeat(img.col(0), 1, border, leftDuble);

    // Добавить дубликаты слева
    cv::hconcat(leftDuble, img, img);

    cv::Mat rightDubl;
    cv::repeat(img.col(img.cols - 1), 1, border, rightDubl);

    // Добавить дубликаты справа
    cv::hconcat(img, rightDubl, img);
    cv::imwrite("concate.jpg", img);

    const int n = img.cols;
    const int m = img.rows;


    int rows = step + border*2;
    if (rank == 3) {
        rows = m - step * 3;
    }
    int cols = n;

   

    // Создаем массив для хранения собранных данных
    int sendcounts[4];   // Количество элементов для каждого процесса
    int displs[4];       // Смещение для каждого процесса
    for (int i = 0; i < 3; i++) {
        sendcounts[i] = (step+border*2) * img.channels();
        displs[i] = i * step * img.channels();
    }
    displs[3] = 3 * step * img.channels();
    sendcounts[3] = m * img.channels() - step * 3 * img.channels();
 
    MPI_Datatype rowChunk;
    MPI_Type_contiguous(cols, MPI_UNSIGNED_CHAR, &rowChunk);
    MPI_Type_commit(&rowChunk);

    MPI_Datatype myChunk;
    MPI_Type_vector(rows * img.channels(), cols, cols, MPI_UNSIGNED_CHAR, &myChunk);
    MPI_Type_commit(&myChunk);

    Mat recv_img(Size(cols, rows), img.type());

    //распределяем по процессам
    MPI_Scatterv(&img.data[0], sendcounts, displs, rowChunk, &recv_img.data[0], 1, myChunk, 0, MPI_COMM_WORLD);

    std::string proc_in = "input_" + std::to_string(rank) + resolution;
    cv::imwrite(proc_in, recv_img);

    Mat output_image= MyBlur(recv_img, GaussMatrix);
    std::string proc_out = "answ_"+ std::to_string(rank) + resolution;
    cv::imwrite(proc_out, output_image);

    int original_rows = step;
    if (rank == 3) {
        original_rows = original_m - step * 3;
    }
    MPI_Datatype out_chunk;
    MPI_Type_vector(original_rows * img.channels(), original_n, original_n, MPI_UNSIGNED_CHAR, &out_chunk);
    MPI_Type_commit(&out_chunk);


    int sendcounts_final[4];   // Количество элементов для каждого процесса
    int displs_final[4];       // Смещение для каждого процесса
    for (int i = 0; i < 3; i++) {
        sendcounts_final[i] = step * img.channels();
        displs_final[i] = i * step * img.channels();
    }
    displs_final[3] = 3 * step * img.channels();
    sendcounts_final[3] = original_m * img.channels() - step * 3 * img.channels();
    for (int i = 0; i < 4; i++) {
        cout << " " << sendcounts_final[i];
    }
    cout << endl;

    MPI_Datatype outRowChunk;
    MPI_Type_contiguous(original_n, MPI_UNSIGNED_CHAR, &outRowChunk);
    MPI_Type_commit(&outRowChunk);

    Mat final_image(original_image.size(), original_image.type());
    //MPI_Gather(&output_image.data[0], original_rows* original_n * img.channels(), MPI_UNSIGNED_CHAR, &final_image.data[0], original_n* original_rows *img.channels(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final, displs_final, outRowChunk, 0, MPI_COMM_WORLD);
    //MPI_Gather(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final[rank], rowChunk, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Сохраните изображение с граничным пикселем, скопированным n раз
        cv::imwrite("final_out.jpg", final_image);
    }
    else {
        cout << "hi"<<rank << endl;
    }
    MPI_Type_free(&outRowChunk);
    MPI_Type_free(&myChunk);
    MPI_Type_free(&rowChunk);
    MPI_Type_free(&out_chunk);
    MPI_Finalize();
    return 0;
}
