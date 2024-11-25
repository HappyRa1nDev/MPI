#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>
#include <chrono>

double a=1.0;
double b=0.1;
double c=0.5;
double bc2 = (c + b) / 2;//0.3

const double x_max = 1;
const double x_min = 0;
int n=500;
const double h = (x_max-x_min) /(n-1);//шаг по x


const  double T = 5;//заданное T

double tau = h/4;//задаем tau с учетом условия устойчиовости
const int m = 4.0 * T / h;//нормальный
double y = tau / h;
double y2 = y*y;

//сетка
double** u;

using namespace std;
//f(x) из начальных условий
double fx(double x) {
    if (x < 0 || x>1) {
        cout << "bad x in fx"<<endl;//в случае если x вышел за диапазон
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
//g(x) из начальных условий
double gx(double x) {
    return 0;
}

//высчитваем по индексу слоя
void pp(int index) {

    for (int i = 1; i < n-1; i++) { 
        //используем полученную формулу
        u[index][i] = 2 * (1 - y2) * u[index - 1][i] + y2 * (u[index - 1][i + 1] + u[index - 1][i - 1]) - u[index - 2][i];
    }
}

void Calculate() {
    //здаем значения на первых 2-х слоях
    for (int i = 0; i < n; i++) {
        u[0][i] = fx(i * h);
        u[1][i] = u[0][i] + tau * gx(i * h);
    }
    //задаем граничные условия
    for (int i = 2; i < m; i++) {
        u[i][0] = 0;
        u[i][n - 1] = 0;
    }
    //счиатем промежуточные
    for (int index = 2; index < m; index++) {
        pp(index);
    }
}
template <typename T>
std::string toString(T val)
{
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

int main()
{
    cout <<n<<"x"<< m << endl;
    u = new double* [m];
    for (int i = 0; i < m; i++) {
        u[i] = new double[n];
    }
    cout <<"h " << h << endl;
    cout <<"tau " << tau << endl;
    cout << "(h^2)/2 " << pow(h, 2) / 2 << endl;
    cout << "ystoichivost: " << (tau < (pow(h, 2) / 2))<<endl;

    int maxIter=1;//число итераций
    std::chrono::time_point<std::chrono::system_clock> startTime;
    std::chrono::time_point<std::chrono::system_clock> endTime;
    double sumTime = 0;
    for (int iter = 0; iter < maxIter; iter++) {
        startTime = chrono::system_clock::now();
        Calculate();
        endTime = chrono::system_clock::now();
        sumTime += chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
    }
    double finalTime = sumTime / maxIter;
    finalTime = finalTime / (pow(10, 6));

    //время
    cout << "Time "<< std::fixed << std::setprecision(6) <<finalTime<< endl;
    string timeFile = "time" + toString(n) + ".txt";
    ofstream timef(timeFile);
    timef<< std::fixed << std::setprecision(6) << finalTime << endl;
    timef.close();

    //сетка
    string fileName = "test" + toString(n) +".txt";
    ofstream fout(fileName);

    fout << "0,";
    for (int i = 0; i < n - 1; i++) {
        fout << i * h<<",";
    }
    fout << (n-1) * h << endl;
    for (int i = 0; i < m; i++) {
        fout << i << ",";
        for (int j = 0; j < n-1; j++) {
            fout << std::fixed << std::setprecision(8) << u[i][j] << ",";
        }
        fout << std::fixed << std::setprecision(8)<<  u[i][(n - 1)]<< endl;
    }
    fout.close();
    delete[] u;
    return 0;
}