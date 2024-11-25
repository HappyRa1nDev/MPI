#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
using namespace std;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 12;
    const int m = 10;
    int rows = 4;
    int cols = n;
 /*   if (rank == 3) {
        cols = n - cols * 3;
    }*/

    // Создаем локальные двумерные массивы на каждом процессе
    int* local_data = new int[n * m];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            local_data[i * n + j] = i;
        }
    }
    //int local_data[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

    // Создаем массив для хранения собранных данных
   

    int sendcounts[4];   // Количество элементов для каждого процесса
    int displs[4];       // Смещение для каждого процесса
    for (int i = 0; i < 4; i++) {
        sendcounts[i] = rows;
        displs[i] = i * 2;
    }
 
    MPI_Datatype rowChunk;
    MPI_Type_contiguous(cols, MPI_INT, &rowChunk);
    MPI_Type_commit(&rowChunk);

    MPI_Datatype mychank;
    MPI_Type_vector(rows, cols, cols, MPI_INT, &mychank);
    MPI_Type_commit(&mychank);
    // Выполняем разделение данных
    int* recv_data = nullptr;
    recv_data = new int[rows * cols];

    MPI_Scatterv(local_data, sendcounts, displs, rowChunk, recv_data, 1, mychank, 0, MPI_COMM_WORLD);

    int* r2= nullptr;
    r2 = new int[n * m];
    MPI_Gatherv(recv_data, sendcounts[rank], rowChunk, r2, sendcounts,displs, rowChunk, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "hi" << endl;
        printf("Gathered data on root process:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", local_data[i * n + j]);
            }
            printf("\n");
        }
        printf("Gathered data on root process:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", recv_data[i * cols + j]);
            }
            printf("\n");
        }
        printf("Gathered data on root process:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", r2[i * n + j]);
            }
            printf("\n");
        }
    }
    else {
        cout << "hi"<<rank << endl;
    }
    //MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
