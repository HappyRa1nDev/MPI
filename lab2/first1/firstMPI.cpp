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

std::vector<std::vector<float>> GaussKernel;

int main(int argc, char** argv) {


    const std::string matrixPath = "mas.txt";
    //"car10.jpg","car20.jpg","car30.jpg", "car64.jpg",  "car120.jpg", "car200.jpg","car300.jpg","car400.jpg","car550.jpg","car800.jpg"
    std::string fileName = "car64.jpg";
    std::string outFilename = "out_" + fileName;
    
    int maxIter = 10;

    GaussKernel = readKernel(matrixPath);
    int border = GaussKernel.size() / 2;

    Mat original_image = imread(fileName);
    Mat final_image(original_image.size(), original_image.type());

    MPI_Init(&argc, &argv);



    const int original_n = original_image.cols;
    const int original_m = original_image.rows;
    int step = original_m / 4;
    if (original_m - step * 3 > step) {
        step++;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double sumProcTime = 0;
    for (int iterImg = 0; iterImg < 2; iterImg++) {
        double procTime = 0;
        double startTime;
        double endTime;
        // Создаю новое изображение дублями сверху и снизу
        cv::Mat img;
        cv::repeat(original_image.row(0), border, 1, img);

        // Добавить дубликаты сверху
        cv::vconcat(img, original_image, img);

        cv::Mat botDubl;
        cv::repeat(original_image.row(original_image.rows - 1), border, 1, botDubl);

        // Добавьте дубликаты снизу
        cv::vconcat(img, botDubl, img);
        //cv::imwrite("concate.jpg", img);
        for (int iter = 0; iter < maxIter; iter++) {
            startTime = MPI_Wtime();
            const int n = img.cols;
            const int m = img.rows;


            int rows = step + border * 2;

            if (rank == 3) {
                rows = m - step * 3;
            }
            int cols = n;



            // Создаем массив для хранения собранных данных
            int sendcounts[4];   // Количество элементов для каждого процесса
            int displs[4];       // Смещение для каждого процесса
            for (int i = 0; i < 3; i++) {
                sendcounts[i] = (step + border * 2) * img.channels();

                displs[i] = i * (step)*img.channels();
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

            //std::string proc_in = "input_" + std::to_string(rank) + fileName;
            //cv::imwrite(proc_in, recv_img);

            Mat output_image = MyBlur(recv_img, GaussKernel);
            //std::string proc_out = "answ_" + std::to_string(rank) + fileName;
            //cv::imwrite(proc_out, output_image);

            int original_rows = step;
            if (rank == 3) {
                original_rows = original_m - step * 3;
            }
            MPI_Datatype out_chunk;
            MPI_Type_vector(original_rows * img.channels(), cols, cols, MPI_UNSIGNED_CHAR, &out_chunk);
            MPI_Type_commit(&out_chunk);


            int sendcounts_final[4];   // Количество элементов для каждого процесса
            int displs_final[4];       // Смещение для каждого процесса
            for (int i = 0; i < 3; i++) {
                sendcounts_final[i] = step * img.channels();
                displs_final[i] = i * step * img.channels();
            }
            displs_final[3] = 3 * step * img.channels();
            sendcounts_final[3] = original_m * img.channels() - step * 3 * img.channels();
            //обираем все вместе
           if(iterImg!=1)
                MPI_Allgatherv(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final, displs_final, rowChunk, MPI_COMM_WORLD);
           else {
               MPI_Gatherv(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final, displs_final, rowChunk, 0, MPI_COMM_WORLD);
           }
            //MPI_Gatherv(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final, displs_final, rowChunk,0, MPI_COMM_WORLD);
            //MPI_Gather(&output_image.data[0], 1, out_chunk, &final_image.data[0], sendcounts_final[rank], rowChunk, 0, MPI_COMM_WORLD);
            MPI_Type_free(&myChunk);
            MPI_Type_free(&rowChunk);
            MPI_Type_free(&out_chunk);
            endTime = MPI_Wtime();
            procTime += endTime - startTime;
        }
        procTime = procTime / maxIter;
        sumProcTime += procTime;
        MPI_Barrier(MPI_COMM_WORLD);
        if(iterImg!=1)
            original_image = final_image.clone();
    }
    cout << "Time" <<rank<<" "<< fixed << setprecision(6) << sumProcTime << endl;
    double maxtime;
    MPI_Reduce(&sumProcTime, &maxtime, size - 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);//получаем максимальное время

    if (rank == 0) {
        cout << "MaxTime " << fixed << setprecision(6)<<maxtime << endl;
        std::ofstream ftime("ptime_" + fileName + ".txt");
        ftime << fixed << setprecision(6) << maxtime << endl;
        ftime.close();
        cv::imwrite("pout_"+fileName, final_image);
    }
    //MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
