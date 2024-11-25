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


Mat F1(Mat image, std::vector<std::vector<float>> GaussMatrix) {

    int n = GaussMatrix.size(); //??
    int border = n / 2;

    cv::Rect roi(0, border, image.cols, image.rows - border * 2);

    Mat blurImage = image(roi);//выходное без карйних пикселей.


    // Разделяем картинку на цвета и сохраняем каждый цвет в свой массив RGB
    for (int r = border; r < image.rows - border; r++)
        for (int c = 0; c < image.cols; c++) {

            //// Получаем массив цветов для пикселя i-j            
            Vec3b& color = image.at<Vec3b>(r, c); // определить дефолнтым значением
            float red = 0, green = 0, blue = 0;

            for (int k_row = -border, k = 0; k_row <= border; k_row++, k++) {
                for (int k_col = -border, l = 0; k_col <= border; k_col++, l++) {

                    int _row = r + k_row;
                    int _col = c + k_col;
                    //_row = fix(_row, 0, image.rows - 1);
                    _col = fix(_col, 0, image.cols - 1);

                    color = image.at<Vec3b>(_row, _col);

                    // Синий
                    blue += color[0] * GaussMatrix[k][l];
                    // Зеленый
                    green += color[1] * GaussMatrix[k][l];
                    // Красный
                    red += color[2] * GaussMatrix[k][l];
                }
            }
            // Записываем массив цветов, как "пиксель" в новой картинке
            color = image.at<Vec3b>(r, c);
            color[0] = blue;
            color[1] = green;
            color[2] = red;
            blurImage.at<Vec3b>(r - border, c) = color;
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string matrixPath = "mas.txt";
    std::vector<std::string> resolutions({ "64.jpg", "128.jpg", "300.jpg", "574.jpg", "900.jpg" });
    std::string resolution = "car2.jpg";
    std::string imagePath = resolution;
    std::string output = "answ_" + resolution;
    int repeats = 1;

    GaussMatrix = readMatrix(matrixPath);
    int border = GaussMatrix.size() / 2;

    Mat original_image = imread(imagePath);
    Mat final_image(original_image.size(), original_image.type());


    const int original_n= original_image.cols;
    const int original_m= original_image.rows;
    cout << "m" << original_m<<endl;
    const int original_step = original_m / 4;
    int step = original_step;
    cout << step<<endl;
    
    // Создайте новое изображение с продублированной строкой сверху
    cv::Mat img;
    cv::repeat(original_image.row(0), border, 1, img);

    // Добавить дубликаты сверху
    cv::vconcat(img, original_image, img);

    cv::Mat botDubl;
    cv::repeat(original_image.row(original_image.rows - 1), border, 1, botDubl);

    // Добавьте дубликаты снизу
    cv::vconcat(img, botDubl, img);
    cv::imwrite("concate.jpg", img);

    const int n = img.cols;
    const int m = img.rows;


   
  /*  if (m - step * 3 > 4)
        step++;*/
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
    // cout << img.channels() << endl;
    MPI_Type_commit(&rowChunk);

    MPI_Datatype mychank;
    MPI_Type_vector(rows * img.channels(), cols, cols, MPI_UNSIGNED_CHAR, &mychank);
    MPI_Type_commit(&mychank);


    // Выполняем разделение данных
    //int* recv_data = nullptr;
    //recv_data = new int[rows * cols];
    Mat recv_img(Size(cols, rows), img.type());


    MPI_Scatterv(&img.data[0], sendcounts, displs, rowChunk, &recv_img.data[0], 1, mychank, 0, MPI_COMM_WORLD);

    std::string proc_in = "input_" + std::to_string(rank) + resolution;
    cv::imwrite(proc_in, recv_img);

    Mat output_image= F1(recv_img, GaussMatrix);
    std::string proc_out = "answ_"+ std::to_string(rank) + resolution;
    cv::imwrite(proc_out, output_image);

    int original_rows = original_step;
    if (rank == 3) {
        original_rows = original_m - original_step * 3;
    }
    MPI_Datatype out_chunk;
    MPI_Type_vector(original_rows * img.channels(), cols, cols, MPI_UNSIGNED_CHAR, &out_chunk);
    MPI_Type_commit(&out_chunk);


    int sendcounts_final[4];   // Количество элементов для каждого процесса
    int displs_final[4];       // Смещение для каждого процесса
    for (int i = 0; i < 3; i++) {
        sendcounts_final[i] = original_step * img.channels();
        displs_final[i] = i * original_step * img.channels();
    }
    displs_final[3] = 3 * original_step * img.channels();
    sendcounts_final[3] = original_m * img.channels() - original_step * 3 * img.channels();

    MPI_Gatherv(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final, displs_final, rowChunk, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Сохраните изображение с граничным пикселем, скопированным n раз
        cv::imwrite("final_out.jpg", final_image);
    }
    else {
        cout << "hi"<<rank << endl;
    }
    //MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
