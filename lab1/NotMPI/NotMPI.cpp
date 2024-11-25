
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>
#define use_more 0
using namespace std;
const double w_int = 0.75;
double myintg(double a, double b, int n) {
	double h = (b - a) / n;
	double sum = 0;
	for (int i = 1; i <= n; i++) {
		double xi = a + h * i;
		sum += h / (xi * xi);
	}
	return sum;
}
int main()
{
#if use_more
	cout << "N " << " sum " << " time " << " pogreshost " << endl;
#endif
	double sum_time = 0;
	double sum;
	int n = 5000;
#if use_more
	for (n = 5; n < 5000; n += 67) {
#endif
		for (int i = 0; i < 10; i++) {
			int a = 1;
			int b = 4;
			double start = clock();
			sum = myintg(a, b, n);
			double end = clock();

			sum_time += (end - start)/ CLOCKS_PER_SEC;
		}
		//cout << "posledovatelnoe" << endl;
#if use_more
		cout << fixed << setprecision(8) << n << " " << sum << " " << sum_time / 1000 << " " << abs(w_int - sum) << endl;
	}
#else
		cout<<"N "<< n << fixed << setprecision(8)<<  " sum: " << sum << " time: " << sum_time / 1000 << " pogreshost " << abs(w_int - sum) << endl;
#endif

}

