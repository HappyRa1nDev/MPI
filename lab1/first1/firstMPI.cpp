#include "math.h"
#include "mpi.h"
#include <iomanip>
#include <iostream>
using namespace std;

#define use_more 0
double xmin = 1;
double xmax = 4;

const double w_int = 0.75;
int n = 5;

double myintg(double a, double b, double h,int num) {
	double sum = 0;
	for (int i = 1; i <= num; i++) {
		double xi = a + h * i;
		sum += h / (xi * xi);
	}
	return sum;
}
enum eTag {
	eResAndTime = 2
};


int main(int argc, char** argv)
{

		MPI_Init(&argc, &argv);


		int size;
		int cur_proc;

		const int main_proc = 0;

		MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);//получили ранк текущего просесса
		MPI_Comm_size(MPI_COMM_WORLD, &size);
#if use_more
		if(cur_proc== main_proc)
			cout << "N " << " sum " << " time " << " pogreshost " << endl;
		for (n = 5; n < 5000; n += 67) {
#endif
			double h = (xmax - xmin) / n;
		
		MPI_Status status;
		double sum = 0;//результат
		double ResAndTime[2];//массив для перессылки результата и времени		
		double t1, t2, t3;
		int len = n / size;

		double a, b;//интервалы
		a = xmin + h * cur_proc * len;
		b = a + h * len;
		if (cur_proc == size - 1) {
			b = xmax;
			len += n % size;
		}
		double time = 0;
		for (int k = 0; k < 1000; k++) {
			t1 = MPI_Wtime();
			sum = myintg(a, b, h, len);
			t2 = MPI_Wtime();
			time += t2 - t1;
		}
		t3 = time / 1000;
#if !use_more
		cout << fixed << setprecision(8) << "proc " << cur_proc << " sum " << sum << " time " << t3 << endl;
#endif
		//вычиляем и передаем
		if (cur_proc != main_proc) {
			ResAndTime[0] = sum;
			ResAndTime[1] = t3;
			MPI_Send(&ResAndTime, 2, MPI_DOUBLE, main_proc, eResAndTime, MPI_COMM_WORLD);
		}
		else {
			//bool mas[3];//для вычисления большого числа n важно понимать какой именно процесс выслал

			for (int k = 1; k < size;k++) {
				MPI_Recv(&ResAndTime, 2, MPI_DOUBLE, k, eResAndTime, MPI_COMM_WORLD, &status);
				sum += ResAndTime[0];
				if (t3 < ResAndTime[1]) {
					t3 = ResAndTime[1];
				}
			}
#if !use_more
			cout  << "N " << n << fixed << setprecision(8) << " sum: " << sum << " time: " << t3 << " pogreshost " << abs(w_int - sum) << endl;
#else
			cout << n << fixed << setprecision(8) <<" " << sum << " " << t3 << " " << abs(w_int - sum) << endl;
#endif
		}
#if use_more
		}
#endif
		MPI_Finalize();

		
	return 0;
}