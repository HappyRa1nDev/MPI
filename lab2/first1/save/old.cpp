//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "mpi.h"

using namespace std;
using namespace cv;


int fix(int value, int min, int max) {
    return value < min ? min : value > max ? max : value;
}
Mat F1(Mat image, std::vector<std::vector<float>> GaussMatrix) {

    int n = GaussMatrix.size();
    int border = n / 2;

    Mat blurImage = image.clone();

    // Разделяем картинку на цвета и сохраняем каждый цвет в свой массив RGB
    for (int x = 0; x < image.rows; x++)
        for (int y = 0; y < image.cols; y++) {

            //// Получаем массив цветов для пикселя i-j            
            Vec3b& color = image.at<Vec3b>(x, y); // определить дефолнтым значением
            uchar r = 0, g = 0, b = 0;

            for (int fx = -border, k = 0; fx <= border; fx++, k++) {
                for (int fy = -border, l = 0; fy <= border; fy++, l++) {

                    int rx = x + fx;
                    int ry = y + fy;
                    rx = fix(rx, 0, image.rows - 1);
                    ry = fix(ry, 0, image.cols - 1);

                    color = image.at<Vec3b>(rx, ry);
                    // Синий
                    b += saturate_cast<uchar>(color[0] * GaussMatrix[k][l]);
                    // Зеленый
                    g += saturate_cast<uchar>(color[1] * GaussMatrix[k][l]);
                    // Красный
                    r += saturate_cast<uchar>(color[2] * GaussMatrix[k][l]);
                }
            }
            // Записываем массив цветов, как "пиксель" в новой картинке
            color[0] = b;
            color[1] = g;
            color[2] = r;
            blurImage.at<Vec3b>(x, y) = color;
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

    return matrix;
}

int main(int argc, char** argv)
{
    const std::string matrixPath = "mas.txt";
    std::vector<std::string> resolutions({ "64.jpg", "128.jpg", "300.jpg", "574.jpg", "900.jpg","1200.jpg" ,"2000.jpg" });
    std::string resolution = "car.jpg";
    std::string imagePath = /*"C:/Users/darkm/source/repos/MPI_Pproject/MPI_Pproject/fortest/" +*/ resolution;
    std::string output = "out_" + resolution;

    double tStart, tEnd, totalTime; int repeats = 1;
    double allTotalTime, resultTime = 0.0;

    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image, finalImage;
    std::vector<std::vector<float>> GaussMatrix;

    image = imread(imagePath);
    GaussMatrix = readMatrix(matrixPath);

    finalImage = image.clone();

    totalTime = 0;
    for (int i = 0; i < repeats; i++)
    {
        tStart = MPI_Wtime();

        int deltaBottom;
        if ((finalImage.rows) % size == 0)
            deltaBottom = 0;
        else
            deltaBottom = size - (finalImage.rows) % size; // для уравнивания кусков

        copyMakeBorder(image, finalImage, 0, 0 + deltaBottom, 0, 0, BORDER_REPLICATE); // Наращиваем картинку значениями окраин

        int blockCount = finalImage.rows * finalImage.channels() / size;  // Количество блоков(строк)
        int blockLength = finalImage.cols;       // Количество элементов в каждом из блоков
        int blockStride = finalImage.cols;      //количество элементов между началами соседних блоков.

        int chunkCount = (GaussMatrix.size() / 2) * finalImage.channels();
        int chunkLength = blockLength;
        int chunkStride = blockStride;

        // Объявляем переменные для хранения производного типа
        MPI_Datatype blockType, chunk;
        // Создаем производные типы на основе существующего 
        MPI_Type_vector(blockCount, blockLength, blockStride, MPI_UNSIGNED_CHAR, &blockType);
        MPI_Type_commit(&blockType);

        MPI_Type_vector(chunkCount, chunkLength, chunkStride, MPI_UNSIGNED_CHAR, &chunk);
        MPI_Type_commit(&chunk);

        Mat blockImage(Size(blockLength, blockCount / finalImage.channels()), finalImage.type());
        Mat chunkToBotImage(Size(chunkLength, chunkCount / finalImage.channels()), finalImage.type());
        Mat chunkToTopImage(Size(chunkLength, chunkCount / finalImage.channels()), finalImage.type());
        MPI_Scatter(&finalImage.data[0], 1, blockType, &blockImage.data[0], 1, blockType, 0, MPI_COMM_WORLD);

        // Обработка блока изображения 

        int dest = rank - 1;   // получатель
        int source = rank + 1; // отправитель
        if (rank == 0) dest = MPI_PROC_NULL;
        if (rank == size - 1) source = MPI_PROC_NULL;
        MPI_Recv(&chunkToBotImage.data[0], 1, chunk, source, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&blockImage.data[0], 1, chunk, dest, 1, MPI_COMM_WORLD);

        dest = rank + 1;   // получатель
        source = rank - 1; // отправитель
        if (rank == 0) source = MPI_PROC_NULL;
        if (rank == size - 1) dest = MPI_PROC_NULL;
        MPI_Recv(&chunkToTopImage.data[0], 1, chunk, source, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&blockImage.data[0 + blockLength * blockCount - chunkLength * chunkCount], 1, chunk, dest, 1, MPI_COMM_WORLD);

        if (rank != 0) vconcat(chunkToTopImage, blockImage, blockImage);
        if (rank != size - 1) vconcat(blockImage, chunkToBotImage, blockImage);


        blockImage = F1(blockImage, GaussMatrix);

        if (rank == 0) vconcat(chunkToTopImage, blockImage, blockImage);

        // Сбор блоков изображения со всех процессов

        Mat outputImage(Size(blockLength, size * blockCount / finalImage.channels()), image.type());
        MPI_Gather(&blockImage.data[0 + chunkLength * chunkCount], 1, blockType, &outputImage.data[0], 1, blockType, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            outputImage = outputImage(Rect(0, 0, outputImage.cols, outputImage.rows - deltaBottom));

            imwrite(output, outputImage);

            //imshow("", outputImage);                  // Для теста можем вывести картинку на окно
            //waitKey(0);						         // Ждем нажатия клавиши для закрытия окна с картинкой
            //destroyAllWindows();				    // Освобождаем память из-под окон
        }
        tEnd = MPI_Wtime();
        totalTime = tEnd - tStart;


        MPI_Reduce(&totalTime, &allTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            resultTime += allTotalTime;
        }
    }

    resultTime /= repeats;

    //printf("R%d time: %f\n", rank, totalTime);

    if (rank == 0) {
        printf("Result time: %f\n", resultTime);
    }

    MPI_Finalize();

    return 0;
}
