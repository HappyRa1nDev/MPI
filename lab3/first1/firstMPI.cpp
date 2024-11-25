#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>


using namespace std;

long double a = 1.0;
long double b = 0.1;
long double c = 0.5;
long double bc2 = (c + b) / 2;//0.3

const long double x_max = 1;
const long double x_min = 0;
int n = 30;
const long double h = (x_max - x_min) / (n - 1);;//шаг по x


const  long double T = 5;//заданное T
const int m = 4.0 * T / h;//задаем m 
long double tau = h * h / 4;//задаем tau с учетом условия устойчиовости
long double y = tau / h;
long double y2 = y * y;

int step = n / 4;

long double fx(long double x) {
	if (x < 0 || x>1) {
		cout << "bad x in fx" << endl;//в случае если x вышел за диапазон
		return -1;
	}
	if (x <= b) {
		return 0;
	}
	else if (x > b && x <= bc2) {
		return 5 * x - 0.5;
	}
	else  if (x > bc2 && x <= c) {
		return (-5) * x + 2.5;
	}
	else
		return 0;

}
long double gx(long double x) {
	return 0;
}

enum eTag {
	eRes = 2,
	eChunk
};
template <typename T>
std::string toString(T val)
{
	std::ostringstream oss;
	oss << val;
	return oss.str();
}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4)
	{
		printf("This application is meant to be run with 4 processes, not %d.\n", size);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Заполняем массив измерений количеством элементов в каждом направлении
	int dims[1];
	dims[0] = size;

	// делаем без периодичности (не соединяем конец и начало размерности)
	int periods[1] = { false };

	// Настраиваем, чтобы MPI перенумеровал процессы
	// Если reorder = false, то номер каждого процесса в новой группе идентичен номеру в старой группе.
	// Если reorder = true, функция может переупорядочивать процессы (возможно, чтобы обеспечить хорошее наложение виртуальной топологии на физическую систему).
	int reorder = true;

	// Создаем декартовый коммуникатор
	MPI_Comm comm_2D;
	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &comm_2D);

	// Создаем описание соседей
	enum DIRECTIONS { LEFT, RIGHT };
	//const char* neighbours_names[] = { "left", "right" };
	int neighbours_ranks[2];

	//найдем соседей слева и справа
	MPI_Cart_shift(comm_2D, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

	//Возвращаем номер текущего процесса в рамках декартова коммуникатора
	int my_rank;
	MPI_Comm_rank(comm_2D, &my_rank);



	int num = step;
	if (my_rank == 3) {
		//последний может быть чуть больше
		num = n - step * 3;
	}


	MPI_Status status;

	MPI_Datatype sendrecvItem;//тип из одного элемента, чтобы подошел под чек лист.
	MPI_Type_contiguous(1, MPI_DOUBLE, &sendrecvItem);
	MPI_Type_commit(&sendrecvItem);

	int maxIter = 1000;//максимальное число итераций для замера времени
	double* u_final = nullptr;

	double procTime = 0;
	double startTime;
	double endTime;
	for (int iter = 0; iter < maxIter; iter++) {
		startTime = MPI_Wtime();
		double* u = new double[m * num];

		//заполняем первые две строки
		for (int i = 0; i < num; i++) {
			u[0 * num + i] = fx((my_rank * step + i) * h);
			u[1 * num + i] = u[0 * num + i] + tau * gx((my_rank * step + i) * h);
		}




		double sendRes;//то что отправляем
		double getRes;//то что получаем
		//высылаемвычисленные значения
		if (neighbours_ranks[LEFT] != MPI_PROC_NULL) {
			sendRes = u[1 * num + 0];
			MPI_Send(&sendRes, 1, sendrecvItem, neighbours_ranks[LEFT], eRes, comm_2D);
		}
		if (neighbours_ranks[RIGHT] != MPI_PROC_NULL) {
			sendRes = u[1 * num + num - 1];
			MPI_Send(&sendRes, 1, sendrecvItem, neighbours_ranks[RIGHT], eRes, comm_2D);
		}
		for (int index = 2; index < m; index++) {

			if (neighbours_ranks[LEFT] != MPI_PROC_NULL) {
				//получаем посылку слева
				MPI_Recv(&getRes, 1, sendrecvItem, neighbours_ranks[LEFT], eRes, comm_2D, &status);
				u[index * num + 0] = 2 * (1 - y2) * u[(index - 1) * num + 0] + y2 * (u[(index - 1) * num + 0 + 1] - getRes) - u[(index - 2) * num + 0];
				//высылаем вычисленное значение слева
				sendRes = u[index * num + 0];
				if (index != m - 1)
					MPI_Send(&sendRes, 1, sendrecvItem, neighbours_ranks[LEFT], eRes, comm_2D);
			}
			else {
				u[index * num + 0] = 0;
			}


			if (neighbours_ranks[RIGHT] != MPI_PROC_NULL) {
				//получаем посылку справа
				MPI_Recv(&getRes, 1, sendrecvItem, neighbours_ranks[RIGHT], eRes, comm_2D, &status);
				u[index * num + num - 1] = 2 * (1 - y2) * u[(index - 1) * num + num - 1] + y2 * (getRes - u[(index - 1) * num + num - 1 - 1]) - u[(index - 2) * num + num - 1];
				//высылаем вычисленное значение справа
				sendRes = u[index * num + num - 1];
				if (index != m - 1)
					MPI_Send(&sendRes, 1, sendrecvItem, neighbours_ranks[RIGHT], eRes, comm_2D);
			}
			else {
				u[index * num + num - 1] = 0;
			}

			//считаем промежуточные на слое
			for (int i = 1; i < num - 1; i++) {
				u[index * num + i] = 2 * (1 - y2) * u[(index - 1) * num + i] + y2 * (u[(index - 1) * num + i + 1] - u[(index - 1) * num + i - 1]) - u[(index - 2) * num + i];
			}

		}


		MPI_Datatype chunk_old, chunk;//производный тип для получения результатов работы процессов
		MPI_Type_vector(m, step, n, MPI_DOUBLE, &chunk_old);
		MPI_Type_create_resized(chunk_old, 0, step * sizeof(MPI_DOUBLE), &chunk);//корректируем
		MPI_Type_commit(&chunk);

		MPI_Datatype chunkLast_old, chunkLast;//производный тип для последнего процесса, так как он может быть больше других
		MPI_Type_vector(m, (n - 3 * step), n, MPI_DOUBLE, &chunkLast_old);
		MPI_Type_create_resized(chunkLast_old, 0, (n - 3 * step) * sizeof(MPI_DOUBLE), &chunkLast);
		MPI_Type_commit(&chunkLast);

		MPI_Datatype sendType;
		MPI_Type_vector(m, num, num, MPI_DOUBLE, &sendType);
		MPI_Type_commit(&sendType);



		if (my_rank == 0) {
			u_final = new double[m * n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < num; j++) {
					u_final[i * n + j] = u[i * num + j];
				}
			}
			//собираем результаты
			MPI_Recv(u_final + step, 1, chunk, 1, eChunk, comm_2D, MPI_STATUSES_IGNORE);
			MPI_Recv(u_final + step * 2, 1, chunk, 2, eChunk, comm_2D, MPI_STATUSES_IGNORE);
			MPI_Recv(u_final + step * 3, 1, chunkLast, 3, eChunk, comm_2D, MPI_STATUSES_IGNORE);
		}
		else {
			//отправляем результат
			MPI_Send(u, 1, sendType, 0, eChunk, comm_2D);
		}
		double endTime = MPI_Wtime();
		procTime += endTime - startTime;
		MPI_Type_free(&chunk);
		MPI_Type_free(&chunkLast);
		MPI_Type_free(&sendType);
		delete[] u;
	}
	MPI_Type_free(&sendrecvItem);
	procTime = procTime / maxIter;
	double maxtime;
	MPI_Reduce(&procTime, &maxtime, size - 1, MPI_DOUBLE, MPI_MAX, 0, comm_2D);//получаем максимальное время

	//вывод времени на каждом процессе
	cout << "Time " << my_rank << " " << std::fixed << std::setprecision(6) << procTime << endl;
	if (my_rank == 0) {
		string timeFile = "pTime" + toString(n) + ".txt";
		ofstream timef(timeFile);
		timef << std::fixed << std::setprecision(6) << maxtime << endl;
		timef.close();
		cout << maxtime << endl;

		string fileName = "ptest" + toString(n) + ".txt";
		ofstream fout(fileName);

		fout << "0,";
		for (int i = 0; i < n - 1; i++) {
			fout << i * h << ",";
		}
		fout << (n - 1) * h << endl;
		for (int i = 0; i < m; i++) {
			fout << i << ",";
			for (int j = 0; j < n - 1; j++) {
				fout << std::fixed << std::setprecision(8) << u_final[i * n + j] << ",";
			}
			fout << std::fixed << std::setprecision(8) << u_final[i * n + n - 1] << endl;
		}
		delete[] u_final;
	}

	MPI_Comm_free(&comm_2D);

	MPI_Finalize();

	return EXIT_SUCCESS;
}