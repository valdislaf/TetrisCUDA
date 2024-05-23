#include <iostream>
#include <limits> 
#include <stdio.h>
#include <vector>
#include <string>
#define NOMINMAX
#include <Windows.h>
#include <random>
#include <cmath>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

std::random_device rd; // Инициализация генератора случайных чисел
#define ID_NEW_GAME_BUTTON 15001
#define ID_EXIT_BUTTON 15002
// В глобальной области видимости объявляем идентификатор таймера
#define IDT_TIMER1 1

// Структура Point для работы на устройстве и хосте
struct Point {
    int x;
    int y;

    __host__ __device__ Point(int x_ = 0, int y_ = 0) : x(x_), y(y_) {}

    __host__ __device__ void setX(int newX) { x = newX; }
    __host__ __device__ void setY(int newY) { y = newY; }
};

// Класс Tetromino для работы на устройстве и хосте
class Tetromino {
public:
    std::vector<Point> pointsRect1;
    std::vector<Point> pointsRect2;
    std::vector<Point> pointsRect3;
    std::vector<Point> pointsRect4;

    __host__ __device__ Tetromino() {}

    // Конструктор
    __host__ __device__ Tetromino(const std::vector<Point>& p1, const std::vector<Point>& p2,
        const std::vector<Point>& p3, const std::vector<Point>& p4)
        : pointsRect1(p1), pointsRect2(p2), pointsRect3(p3), pointsRect4(p4) {}

    // Методы для доступа к точкам
    __host__ __device__ const std::vector<Point>& getPointsRect1() const { return pointsRect1; }
    __host__ __device__ const std::vector<Point>& getPointsRect2() const { return pointsRect2; }
    __host__ __device__ const std::vector<Point>& getPointsRect3() const { return pointsRect3; }
    __host__ __device__ const std::vector<Point>& getPointsRect4() const { return pointsRect4; }

    // Методы для изменения точек
    __host__ __device__ void setPointsRect1(const std::vector<Point>& points) { pointsRect1 = points; }
    __host__ __device__ void setPointsRect2(const std::vector<Point>& points) { pointsRect2 = points; }
    __host__ __device__ void setPointsRect3(const std::vector<Point>& points) { pointsRect3 = points; }
    __host__ __device__ void setPointsRect4(const std::vector<Point>& points) { pointsRect4 = points; }
};

// Ядро для масштабирования точек
__global__ void ScaleKernel(int scale, Point* points, int* offsets, int numTetrominos, int totalPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPoints) return;

    int tetrominoIdx = 0;
    while (tetrominoIdx < numTetrominos - 1 && offsets[tetrominoIdx + 1] <= idx) {
        ++tetrominoIdx;
    }

    points[idx].x *= scale;
    points[idx].y *= scale;
}

// Основная функция для масштабирования
void Scale(int scale, std::vector<std::vector<Tetromino>>& Figs) {
    int numFigs = Figs.size();
    std::vector<int> offsets;
    int totalPoints = 0;

    for (const auto& tetrominoList : Figs) {
        for (const auto& tetromino : tetrominoList) {
            offsets.push_back(totalPoints);
            totalPoints += tetromino.getPointsRect1().size();
            totalPoints += tetromino.getPointsRect2().size();
            totalPoints += tetromino.getPointsRect3().size();
            totalPoints += tetromino.getPointsRect4().size();
        }
    }
    offsets.push_back(totalPoints);

    Point* d_points;
    int* d_offsets;
    cudaMalloc(&d_points, totalPoints * sizeof(Point));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<Point> allPoints;
    for (const auto& tetrominoList : Figs) {
        for (const auto& tetromino : tetrominoList) {
            allPoints.insert(allPoints.end(), tetromino.getPointsRect1().begin(), tetromino.getPointsRect1().end());
            allPoints.insert(allPoints.end(), tetromino.getPointsRect2().begin(), tetromino.getPointsRect2().end());
            allPoints.insert(allPoints.end(), tetromino.getPointsRect3().begin(), tetromino.getPointsRect3().end());
            allPoints.insert(allPoints.end(), tetromino.getPointsRect4().begin(), tetromino.getPointsRect4().end());
        }
    }
    cudaMemcpy(d_points, allPoints.data(), totalPoints * sizeof(Point), cudaMemcpyHostToDevice);

    // Запускаем ядро
    int blockSize = 256;
    int numBlocks = (totalPoints + blockSize - 1) / blockSize;
    ScaleKernel << <numBlocks, blockSize >> > (scale, d_points, d_offsets, numFigs, totalPoints);
    cudaDeviceSynchronize();

    // Копируем данные обратно на хост
    cudaMemcpy(allPoints.data(), d_points, totalPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    // Обновляем исходные данные
    int pointIdx = 0;
    for (auto& tetrominoList : Figs) {
        for (auto& tetromino : tetrominoList) {
            auto size1 = tetromino.getPointsRect1().size();
            std::vector<Point> pointsRect1(allPoints.begin() + pointIdx, allPoints.begin() + pointIdx + size1);
            pointIdx += size1;

            auto size2 = tetromino.getPointsRect2().size();
            std::vector<Point> pointsRect2(allPoints.begin() + pointIdx, allPoints.begin() + pointIdx + size2);
            pointIdx += size2;

            auto size3 = tetromino.getPointsRect3().size();
            std::vector<Point> pointsRect3(allPoints.begin() + pointIdx, allPoints.begin() + pointIdx + size3);
            pointIdx += size3;

            auto size4 = tetromino.getPointsRect4().size();
            std::vector<Point> pointsRect4(allPoints.begin() + pointIdx, allPoints.begin() + pointIdx + size4);
            pointIdx += size4;

            tetromino.setPointsRect1(pointsRect1);
            tetromino.setPointsRect2(pointsRect2);
            tetromino.setPointsRect3(pointsRect3);
            tetromino.setPointsRect4(pointsRect4);
        }
    }

    // Освобождаем память на устройстве
    cudaFree(d_points);
    cudaFree(d_offsets);
}


static class  TetrominoFactory
{
public: static   Tetromino CreateFig1()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(22, 0);
    Point p10 = Point(32, 0);
    Point p11 = Point(32, 10);
    Point p12 = Point(22, 10);

    Point p13 = Point(33, 0);
    Point p14 = Point(43, 0);
    Point p15 = Point(43, 10);
    Point p16 = Point(33, 10);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public: static Tetromino CreateFig2()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(22, 0);
    Point p10 = Point(32, 0);
    Point p11 = Point(32, 10);
    Point p12 = Point(22, 10);

    Point p13 = Point(22, 11);
    Point p14 = Point(32, 11);
    Point p15 = Point(32, 21);
    Point p16 = Point(22, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:  static  Tetromino CreateFig3()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(22, 0);
    Point p10 = Point(32, 0);
    Point p11 = Point(32, 10);
    Point p12 = Point(22, 10);

    Point p13 = Point(0, 11);
    Point p14 = Point(10, 11);
    Point p15 = Point(10, 21);
    Point p16 = Point(0, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public:   static    Tetromino CreateFig4()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(22, 0);
    Point p10 = Point(32, 0);
    Point p11 = Point(32, 10);
    Point p12 = Point(22, 10);

    Point p13 = Point(11, 11);
    Point p14 = Point(21, 11);
    Point p15 = Point(21, 21);
    Point p16 = Point(11, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public:   static   Tetromino CreateFig5()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(0, 11);
    Point p10 = Point(10, 11);
    Point p11 = Point(10, 21);
    Point p12 = Point(0, 21);

    Point p13 = Point(11, 11);
    Point p14 = Point(21, 11);
    Point p15 = Point(21, 21);
    Point p16 = Point(11, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public: static   Tetromino CreateFig6()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 0);
    Point p6 = Point(21, 0);
    Point p7 = Point(21, 10);
    Point p8 = Point(11, 10);

    Point p9 = Point(11, 11);
    Point p10 = Point(21, 11);
    Point p11 = Point(21, 21);
    Point p12 = Point(11, 21);

    Point p13 = Point(22, 11);
    Point p14 = Point(32, 11);
    Point p15 = Point(32, 21);
    Point p16 = Point(22, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public:   static  Tetromino CreateFig7()
{
    Point p1 = Point(11, 0);
    Point p2 = Point(21, 0);
    Point p3 = Point(21, 10);
    Point p4 = Point(11, 10);

    Point p5 = Point(22, 0);
    Point p6 = Point(32, 0);
    Point p7 = Point(32, 10);
    Point p8 = Point(22, 10);

    Point p9 = Point(0, 11);
    Point p10 = Point(10, 11);
    Point p11 = Point(10, 21);
    Point p12 = Point(0, 21);

    Point p13 = Point(11, 11);
    Point p14 = Point(21, 11);
    Point p15 = Point(21, 21);
    Point p16 = Point(11, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:  static  Tetromino CreateFig1_1()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(0, 11);
    Point p6 = Point(10, 11);
    Point p7 = Point(10, 21);
    Point p8 = Point(0, 21);

    Point p9 = Point(0, 22);
    Point p10 = Point(10, 22);
    Point p11 = Point(10, 32);
    Point p12 = Point(0, 32);

    Point p13 = Point(0, 33);
    Point p14 = Point(10, 33);
    Point p15 = Point(10, 43);
    Point p16 = Point(0, 43);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:  static   Tetromino CreateFig2_1()
{
    Point p1 = Point(11, 0);
    Point p2 = Point(21, 0);
    Point p3 = Point(21, 10);
    Point p4 = Point(11, 10);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(11, 22);
    Point p10 = Point(21, 22);
    Point p11 = Point(21, 32);
    Point p12 = Point(11, 32);

    Point p13 = Point(0, 22);
    Point p14 = Point(10, 22);
    Point p15 = Point(10, 32);
    Point p16 = Point(0, 32);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}


public:  static  Tetromino CreateFig2_2()
{
    Point p1 = Point(0, 11);
    Point p2 = Point(10, 11);
    Point p3 = Point(10, 21);
    Point p4 = Point(0, 21);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(22, 11);
    Point p10 = Point(32, 11);
    Point p11 = Point(32, 21);
    Point p12 = Point(22, 21);

    Point p13 = Point(0, 0);
    Point p14 = Point(10, 0);
    Point p15 = Point(10, 10);
    Point p16 = Point(0, 10);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:  static Tetromino CreateFig2_3()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(0, 11);
    Point p6 = Point(10, 11);
    Point p7 = Point(10, 21);
    Point p8 = Point(0, 21);

    Point p9 = Point(0, 22);
    Point p10 = Point(10, 22);
    Point p11 = Point(10, 32);
    Point p12 = Point(0, 32);

    Point p13 = Point(11, 0);
    Point p14 = Point(21, 0);
    Point p15 = Point(21, 10);
    Point p16 = Point(11, 10);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public:  static  Tetromino CreateFig3_1()
{

    Point p1 = Point(11, 0);
    Point p2 = Point(21, 0);
    Point p3 = Point(21, 10);
    Point p4 = Point(11, 10);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(11, 22);
    Point p10 = Point(21, 22);
    Point p11 = Point(21, 32);
    Point p12 = Point(11, 32);

    Point p13 = Point(0, 0);
    Point p14 = Point(10, 0);
    Point p15 = Point(10, 10);
    Point p16 = Point(0, 10);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public: static   Tetromino CreateFig3_2()
{

    Point p1 = Point(0, 11);
    Point p2 = Point(10, 11);
    Point p3 = Point(10, 21);
    Point p4 = Point(0, 21);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(22, 11);
    Point p10 = Point(32, 11);
    Point p11 = Point(32, 21);
    Point p12 = Point(22, 21);

    Point p13 = Point(22, 0);
    Point p14 = Point(32, 0);
    Point p15 = Point(32, 10);
    Point p16 = Point(22, 10);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public: static   Tetromino CreateFig3_3()
{

    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(0, 11);
    Point p6 = Point(10, 11);
    Point p7 = Point(10, 21);
    Point p8 = Point(0, 21);

    Point p9 = Point(0, 22);
    Point p10 = Point(10, 22);
    Point p11 = Point(10, 32);
    Point p12 = Point(0, 32);

    Point p13 = Point(11, 22);
    Point p14 = Point(21, 22);
    Point p15 = Point(21, 32);
    Point p16 = Point(11, 32);


    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public: static  Tetromino CreateFig4_1()
{
    Point p1 = Point(11, 0);
    Point p2 = Point(21, 0);
    Point p3 = Point(21, 10);
    Point p4 = Point(11, 10);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(11, 22);
    Point p10 = Point(21, 22);
    Point p11 = Point(21, 32);
    Point p12 = Point(11, 32);

    Point p13 = Point(0, 11);
    Point p14 = Point(10, 11);
    Point p15 = Point(10, 21);
    Point p16 = Point(0, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}
public:  static  Tetromino CreateFig4_2()
{
    Point p1 = Point(0, 11);
    Point p2 = Point(10, 11);
    Point p3 = Point(10, 21);
    Point p4 = Point(0, 21);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(22, 11);
    Point p10 = Point(32, 11);
    Point p11 = Point(32, 21);
    Point p12 = Point(22, 21);

    Point p13 = Point(11, 0);
    Point p14 = Point(21, 0);
    Point p15 = Point(21, 10);
    Point p16 = Point(11, 10);


    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:  static   Tetromino CreateFig4_3()
{
    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(0, 11);
    Point p6 = Point(10, 11);
    Point p7 = Point(10, 21);
    Point p8 = Point(0, 21);

    Point p9 = Point(0, 22);
    Point p10 = Point(10, 22);
    Point p11 = Point(10, 32);
    Point p12 = Point(0, 32);

    Point p13 = Point(11, 11);
    Point p14 = Point(21, 11);
    Point p15 = Point(21, 21);
    Point p16 = Point(11, 21);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public: static   Tetromino CreateFig6_1()
{

    Point p1 = Point(11, 0);
    Point p2 = Point(21, 0);
    Point p3 = Point(21, 10);
    Point p4 = Point(11, 10);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(0, 11);
    Point p10 = Point(10, 11);
    Point p11 = Point(10, 21);
    Point p12 = Point(0, 21);

    Point p13 = Point(0, 22);
    Point p14 = Point(10, 22);
    Point p15 = Point(10, 32);
    Point p16 = Point(0, 32);


    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

public:   static Tetromino CreateFig7_1()
{

    Point p1 = Point(0, 0);
    Point p2 = Point(10, 0);
    Point p3 = Point(10, 10);
    Point p4 = Point(0, 10);

    Point p5 = Point(11, 11);
    Point p6 = Point(21, 11);
    Point p7 = Point(21, 21);
    Point p8 = Point(11, 21);

    Point p9 = Point(0, 11);
    Point p10 = Point(10, 11);
    Point p11 = Point(10, 21);
    Point p12 = Point(0, 21);

    Point p13 = Point(11, 22);
    Point p14 = Point(21, 22);
    Point p15 = Point(21, 32);
    Point p16 = Point(11, 32);

    return  Tetromino({ p1, p2, p3, p4 }, { p5, p6, p7, p8 }, { p9, p10, p11, p12 }, { p13, p14, p15, p16 });
}

};



int height = 1023;
int width = 660;
int Score = 0;
bool StopGame = false;
int sizes[] = { 2, 4, 4, 4, 1, 2, 2 };
int scale = 3;
std::vector<Tetromino> Figrnd_out;
std::vector<Tetromino> Figrnd_group;
Tetromino Figrnd{};
int mod = 0;
int current_mod = 0;
std::vector<std::vector<Tetromino>>Figs;
int speed_moveY = 2;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
// Ядро CUDA для нахождения минимального значения Y для заданного X1
__global__ void minYKernel(Point* d_points, int* d_offsets, int numTetrominos, int X1, int* d_minYList) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTetrominos) return;

    int startOffset = d_offsets[idx];
    int endOffset = d_offsets[idx + 1];
    int minY = INT_MAX;

    for (int i = startOffset; i < endOffset; ++i) {
        if (d_points[i].x == X1 && d_points[i].y < minY) {
            minY = d_points[i].y;
        }
    }
    d_minYList[idx] = minY;
}

// Основная функция для нахождения минимального Y
int minYfromOutbyX(const std::vector<Tetromino>& Figrnd_out, int X1) {
    int numTetrominos = Figrnd_out.size();
    int totalPoints = 0;

    // Подсчет общего количества точек и заполнение offset-ов
    std::vector<int> offsets(numTetrominos + 1);
    for (int i = 0; i < numTetrominos; ++i) {
        offsets[i] = totalPoints;
        totalPoints += Figrnd_out[i].getPointsRect1().size();
        totalPoints += Figrnd_out[i].getPointsRect2().size();
        totalPoints += Figrnd_out[i].getPointsRect3().size();
        totalPoints += Figrnd_out[i].getPointsRect4().size();
    }
    offsets[numTetrominos] = totalPoints;

    Point* d_points;
    int* d_offsets;
    int* d_minYList;
    cudaMalloc(&d_points, totalPoints * sizeof(Point));
    cudaMalloc(&d_offsets, (numTetrominos + 1) * sizeof(int));
    cudaMalloc(&d_minYList, numTetrominos * sizeof(int));

    // Сбор всех точек в один массив
    std::vector<Point> allPoints;
    for (const auto& tetromino : Figrnd_out) {
        allPoints.insert(allPoints.end(), tetromino.getPointsRect1().begin(), tetromino.getPointsRect1().end());
        allPoints.insert(allPoints.end(), tetromino.getPointsRect2().begin(), tetromino.getPointsRect2().end());
        allPoints.insert(allPoints.end(), tetromino.getPointsRect3().begin(), tetromino.getPointsRect3().end());
        allPoints.insert(allPoints.end(), tetromino.getPointsRect4().begin(), tetromino.getPointsRect4().end());
    }

    cudaMemcpy(d_points, allPoints.data(), totalPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (numTetrominos + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра
    int blockSize = 256;
    int numBlocks = (numTetrominos + blockSize - 1) / blockSize;
    minYKernel << <numBlocks, blockSize >> > (d_points, d_offsets, numTetrominos, X1, d_minYList);
    cudaDeviceSynchronize();

    // Копирование результатов обратно на хост
    std::vector<int> minYList(numTetrominos);
    cudaMemcpy(minYList.data(), d_minYList, numTetrominos * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве
    cudaFree(d_points);
    cudaFree(d_offsets);
    cudaFree(d_minYList);

    // Возвращение минимального Y среди всех найденных
    return *std::min_element(minYList.begin(), minYList.end());
}

//
//void Scale(int scale, std::vector<std::vector<Tetromino>>& Figs)
//{
//    // Перебираем все фигуры в массиве Figs
//    for (int i = 0; i < Figs.size(); i++)
//    {
//        std::vector<Tetromino> tetrominos = Figs[i]; // Получаем массив фигур
//        int size = sizes[i]; // Получаем размер текущего массива фигур
//
//        // Перебираем каждую фигуру в массиве фигур
//        for (int j = 0; j < size; j++)
//        {
//            Tetromino& tetromino = tetrominos[j]; // Получаем фигуру по ссылке
//
//            // Создаем векторы точек перед циклом
//            std::vector<Point> Rect1, Rect2, Rect3, Rect4;
//
//            // Получаем векторы точек перед циклом
//            const std::vector<Point>& pointsRect1 = tetromino.getPointsRect1();
//            const std::vector<Point>& pointsRect2 = tetromino.getPointsRect2();
//            const std::vector<Point>& pointsRect3 = tetromino.getPointsRect3();
//            const std::vector<Point>& pointsRect4 = tetromino.getPointsRect4();
//
//            // Перебираем все точки в каждом прямоугольнике фигуры
//            for (int k = 0; k < pointsRect1.size(); k++)
//            {
//                // Масштабируем координаты каждой точки на указанный масштаб
//                Rect1.push_back(Point(pointsRect1[k].x * scale, pointsRect1[k].y * scale));
//                Rect2.push_back(Point(pointsRect2[k].x * scale, pointsRect2[k].y * scale));
//                Rect3.push_back(Point(pointsRect3[k].x * scale, pointsRect3[k].y * scale));
//                Rect4.push_back(Point(pointsRect4[k].x * scale, pointsRect4[k].y * scale));
//            }
//
//            // Устанавливаем новые векторы точек
//            tetromino.setPointsRect1(Rect1);
//            tetromino.setPointsRect2(Rect2);
//            tetromino.setPointsRect3(Rect3);
//            tetromino.setPointsRect4(Rect4);
//        }
//        Figs[i] = tetrominos;
//    }
//}
//
//int minYfromOutbyX(const std::vector<Tetromino>& Figrnd_out, int X1)
//{
//    // Создаем вектор для хранения минимальных значений Y для каждого массива точек
//    std::vector<int> minYList;
//
//    // Перебираем все элементы Figrnd_out
//    for (const Tetromino& fig : Figrnd_out)
//    {
//        // Находим минимальное значение Y для каждого массива точек в текущем элементе Figrnd_out
//        int aYt1 = std::numeric_limits<int>::max();
//        int aYt2 = std::numeric_limits<int>::max();
//        int aYt3 = std::numeric_limits<int>::max();
//        int aYt4 = std::numeric_limits<int>::max();
//
//        for (const Point& point : fig.getPointsRect1()) {
//            if (point.x == X1 && point.y < aYt1) {
//                aYt1 = point.y;
//            }
//        }
//        for (const Point& point : fig.getPointsRect2()) {
//            if (point.x == X1 && point.y < aYt2) {
//                aYt2 = point.y;
//            }
//        }
//        for (const Point& point : fig.getPointsRect3()) {
//            if (point.x == X1 && point.y < aYt3) {
//                aYt3 = point.y;
//            }
//        }
//        for (const Point& point : fig.getPointsRect4()) {
//            if (point.x == X1 && point.y < aYt4) {
//                aYt4 = point.y;
//            }
//        }
//
//        // Находим минимальное значение Y среди найденных значений
//        int minY = std::min({ aYt1, aYt2, aYt3, aYt4 });
//
//        // Добавляем найденное минимальное значение Y в вектор
//        minYList.push_back(minY);
//    }
//
//    // Находим минимальное значение Y среди всех найденных значений
//    return *std::min_element(minYList.begin(), minYList.end());
//}



bool Moving(Tetromino Figrnd, std::vector<Tetromino> Figrnd_out)
{
    auto X1 = std::min_element(Figrnd.getPointsRect1().begin(), Figrnd.getPointsRect1().end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
        });
    auto Y1 = std::max_element(Figrnd.getPointsRect1().begin(), Figrnd.getPointsRect1().end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
        });
    int Y1min = minYfromOutbyX(Figrnd_out, X1->x);

    auto X2 = std::min_element(Figrnd.getPointsRect2().begin(), Figrnd.getPointsRect2().end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
        });
    auto Y2 = std::max_element(Figrnd.getPointsRect2().begin(), Figrnd.getPointsRect2().end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
        });
    int Y2min = minYfromOutbyX(Figrnd_out, X2->x);

    auto X3 = std::min_element(Figrnd.getPointsRect3().begin(), Figrnd.getPointsRect3().end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
        });
    auto Y3 = std::max_element(Figrnd.getPointsRect3().begin(), Figrnd.getPointsRect3().end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
        });
    int Y3min = minYfromOutbyX(Figrnd_out, X3->x);

    auto X4 = std::min_element(Figrnd.getPointsRect4().begin(), Figrnd.getPointsRect4().end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
        });
    auto Y4 = std::max_element(Figrnd.getPointsRect4().begin(), Figrnd.getPointsRect4().end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
        });
    int Y4min = minYfromOutbyX(Figrnd_out, X4->x);

    if (
        Y1->y + 1 < Y1min &&
        Y2->y + 1 < Y2min &&
        Y3->y + 1 < Y3min &&
        Y4->y + 1 < Y4min
        )
    {
        return true;
    }
    return false;
}

__global__ void maxFixYKernel(Point* d_pointsRect1, Point* d_pointsRect2, Point* d_pointsRect3, Point* d_pointsRect4, int numPoints, int scale, int h, int* d_delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point* pointsRects[] = { d_pointsRect1, d_pointsRect2, d_pointsRect3, d_pointsRect4 };
    int maxY = 0;

    for (int i = 0; i < 4; ++i) {
        Point* pointsRect = pointsRects[i];
        for (int j = 0; j < numPoints; ++j) {
            int y = pointsRect[j].y;
            if (y > maxY) {
                maxY = y;
            }
        }
    }

    maxY -= 10 * scale;

    int delta = h;

    for (int i = 0; i <= h / (11 * scale); ++i) {
        int d = maxY - i * 11 * scale;
        if (abs(d) < abs(delta)) {
            delta = d;
        }
    }

    d_delta[idx] = delta;
}

int maxFixY(const Tetromino& Figrnd,  int h) {
    std::vector<Point> pointsRect1 = Figrnd.getPointsRect1();
    std::vector<Point> pointsRect2 = Figrnd.getPointsRect2();
    std::vector<Point> pointsRect3 = Figrnd.getPointsRect3();
    std::vector<Point> pointsRect4 = Figrnd.getPointsRect4();
    int numPoints = pointsRect1.size() + pointsRect2.size() + pointsRect3.size() + pointsRect4.size();

    Point* d_pointsRect1;
    Point* d_pointsRect2;
    Point* d_pointsRect3;
    Point* d_pointsRect4;
    cudaMalloc(&d_pointsRect1, pointsRect1.size() * sizeof(Point));
    cudaMalloc(&d_pointsRect2, pointsRect2.size() * sizeof(Point));
    cudaMalloc(&d_pointsRect3, pointsRect3.size() * sizeof(Point));
    cudaMalloc(&d_pointsRect4, pointsRect4.size() * sizeof(Point));
    cudaMemcpy(d_pointsRect1, pointsRect1.data(), pointsRect1.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect2, pointsRect2.data(), pointsRect2.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect3, pointsRect3.data(), pointsRect3.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect4, pointsRect4.data(), pointsRect4.size() * sizeof(Point), cudaMemcpyHostToDevice);

    int* d_delta;
    cudaMalloc(&d_delta, numPoints * sizeof(int));

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    maxFixYKernel << <numBlocks, blockSize >> > (d_pointsRect1, d_pointsRect2, d_pointsRect3, d_pointsRect4, numPoints, scale, h, d_delta);
    cudaDeviceSynchronize();

    std::vector<int> delta(numPoints);
    cudaMemcpy(delta.data(), d_delta, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    int minDelta = *std::min_element(delta.begin(), delta.end());

    cudaFree(d_pointsRect1);
    cudaFree(d_pointsRect2);
    cudaFree(d_pointsRect3);
    cudaFree(d_pointsRect4);
    cudaFree(d_delta);

    return minDelta;
}

//
//int maxFixY(const Tetromino& Figrnd, int h)
//{
//    int maxY = 0; // Инициализируем переменную для хранения максимального значения Y
//
//    // Перебираем все массивы точек в объекте Tetromino
//    for (const auto& pointsRect : { Figrnd.getPointsRect1(), Figrnd.getPointsRect2(), Figrnd.getPointsRect3(), Figrnd.getPointsRect4() })
//    {
//        // Перебираем все точки в текущем массиве точек
//        for (const auto& point : pointsRect)
//        {
//            // Обновляем значение maxY, если значение Y текущей точки больше текущего максимального значения maxY
//            if (point.y > maxY)
//            {
//                maxY = point.y;
//            }
//        }
//    }
//
//    maxY = maxY - 10 * scale;
//
//    int delta = h;
//
//    for (int i = 0; i <= h / (11 * scale); i++)
//    {
//        int d = maxY - i * 11 * scale;
//        if (std::abs(d) < std::abs(delta)) { delta = d; }
//    }
//
//    return delta;
//}




static RECT GetRECT(const std::vector<Point>& points)
{
    RECT rect;
    if (points.size() >= 4) // Проверяем, содержит ли вектор как минимум четыре точки
    {
        rect.left = points[0].x;
        rect.top = points[0].y;
        rect.right = points[2].x;
        rect.bottom = points[2].y;
    }
    else // Если вектор содержит меньше четырех точек, используем значения по умолчанию
    {
        rect.left = 0;
        rect.top = 0;
        rect.right = 0;
        rect.bottom = 0;
    }
    return rect;
}

void DrawTetrominoRects(HDC hdc, Tetromino& tetromino, COLORREF color) {
    HBRUSH hBrush = CreateSolidBrush(color); // Создаем кисть с нужным цветом
    RECT rect1 = GetRECT(tetromino.getPointsRect1());
    FillRect(hdc, &rect1, hBrush);
    RECT rect2 = GetRECT(tetromino.getPointsRect2());
    FillRect(hdc, &rect2, hBrush);
    RECT rect3 = GetRECT(tetromino.getPointsRect3());
    FillRect(hdc, &rect3, hBrush);
    RECT rect4 = GetRECT(tetromino.getPointsRect4());
    FillRect(hdc, &rect4, hBrush);

    //FillRect(hdc, &GetRECT(tetromino.getPointsRect1()), hBrush);
    //FillRect(hdc, &GetRECT(tetromino.getPointsRect2()), hBrush);
    //FillRect(hdc, &GetRECT(tetromino.getPointsRect3()), hBrush);
    //FillRect(hdc, &GetRECT(tetromino.getPointsRect4()), hBrush);
    DeleteObject(hBrush); // Освобождаем ресурсы кисти
}

__global__ void moveXYKernel(Point* d_pointsRect1, Point* d_pointsRect2, Point* d_pointsRect3, Point* d_pointsRect4, int numPoints, int deltaX, int deltaY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    d_pointsRect1[idx].x += deltaX;
    d_pointsRect1[idx].y += deltaY;

    d_pointsRect2[idx].x += deltaX;
    d_pointsRect2[idx].y += deltaY;

    d_pointsRect3[idx].x += deltaX;
    d_pointsRect3[idx].y += deltaY;

    d_pointsRect4[idx].x += deltaX;
    d_pointsRect4[idx].y += deltaY;
}

void MoveXY(Tetromino& t, int deltaX, int deltaY) {
    std::vector<Point> pointsRect1 = t.getPointsRect1();
    std::vector<Point> pointsRect2 = t.getPointsRect2();
    std::vector<Point> pointsRect3 = t.getPointsRect3();
    std::vector<Point> pointsRect4 = t.getPointsRect4();
    int numPoints = pointsRect1.size();

    Point* d_pointsRect1;
    Point* d_pointsRect2;
    Point* d_pointsRect3;
    Point* d_pointsRect4;
    cudaMalloc(&d_pointsRect1, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect2, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect3, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect4, numPoints * sizeof(Point));
    cudaMemcpy(d_pointsRect1, pointsRect1.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect2, pointsRect2.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect3, pointsRect3.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect4, pointsRect4.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    moveXYKernel << <numBlocks, blockSize >> > (d_pointsRect1, d_pointsRect2, d_pointsRect3, d_pointsRect4, numPoints, deltaX, deltaY);
    cudaDeviceSynchronize();

    cudaMemcpy(pointsRect1.data(), d_pointsRect1, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect2.data(), d_pointsRect2, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect3.data(), d_pointsRect3, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect4.data(), d_pointsRect4, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    t.setPointsRect1(pointsRect1);
    t.setPointsRect2(pointsRect2);
    t.setPointsRect3(pointsRect3);
    t.setPointsRect4(pointsRect4);

    cudaFree(d_pointsRect1);
    cudaFree(d_pointsRect2);
    cudaFree(d_pointsRect3);
    cudaFree(d_pointsRect4);
}


//
//void MoveXY(Tetromino& t, int deltaX, int deltaY)
//{
//
//    // Создаем векторы точек перед циклом
//    std::vector<Point> Rect1, Rect2, Rect3, Rect4;
//
//    // Получаем векторы точек перед циклом
//    const std::vector<Point>& pointsRect1 = t.getPointsRect1();
//    const std::vector<Point>& pointsRect2 = t.getPointsRect2();
//    const std::vector<Point>& pointsRect3 = t.getPointsRect3();
//    const std::vector<Point>& pointsRect4 = t.getPointsRect4();
//
//    for (int k = 0; k < pointsRect1.size(); k++)
//    {
//        // Масштабируем координаты каждой точки на указанный масштаб
//        Rect1.push_back(Point(pointsRect1[k].x + deltaX, pointsRect1[k].y + deltaY));
//        Rect2.push_back(Point(pointsRect2[k].x + deltaX, pointsRect2[k].y + deltaY));
//        Rect3.push_back(Point(pointsRect3[k].x + deltaX, pointsRect3[k].y + deltaY));
//        Rect4.push_back(Point(pointsRect4[k].x + deltaX, pointsRect4[k].y + deltaY));
//    }
//
//    // Устанавливаем новые векторы точек
//    t.setPointsRect1(Rect1);
//    t.setPointsRect2(Rect2);
//    t.setPointsRect3(Rect3);
//    t.setPointsRect4(Rect4);
//}

__global__ void updateRectPointsKernel(Point* pointsRect, int numPoints, int dY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    pointsRect[idx].y -= dY;
}

void updateRectPoints(Tetromino& fig, int dY) {
    std::vector<Point> pointsRect1 = fig.getPointsRect1();
    std::vector<Point> pointsRect2 = fig.getPointsRect2();
    std::vector<Point> pointsRect3 = fig.getPointsRect3();
    std::vector<Point> pointsRect4 = fig.getPointsRect4();
    int numPoints = pointsRect1.size();

    Point* d_pointsRect1;
    Point* d_pointsRect2;
    Point* d_pointsRect3;
    Point* d_pointsRect4;
    cudaMalloc(&d_pointsRect1, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect2, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect3, numPoints * sizeof(Point));
    cudaMalloc(&d_pointsRect4, numPoints * sizeof(Point));
    cudaMemcpy(d_pointsRect1, pointsRect1.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect2, pointsRect2.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect3, pointsRect3.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsRect4, pointsRect4.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    updateRectPointsKernel << <numBlocks, blockSize >> > (d_pointsRect1, numPoints, dY);
    updateRectPointsKernel << <numBlocks, blockSize >> > (d_pointsRect2, numPoints, dY);
    updateRectPointsKernel << <numBlocks, blockSize >> > (d_pointsRect3, numPoints, dY);
    updateRectPointsKernel << <numBlocks, blockSize >> > (d_pointsRect4, numPoints, dY);
    cudaDeviceSynchronize();

    cudaMemcpy(pointsRect1.data(), d_pointsRect1, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect2.data(), d_pointsRect2, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect3.data(), d_pointsRect3, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointsRect4.data(), d_pointsRect4, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    fig.setPointsRect1(pointsRect1);
    fig.setPointsRect2(pointsRect2);
    fig.setPointsRect3(pointsRect3);
    fig.setPointsRect4(pointsRect4);

    cudaFree(d_pointsRect1);
    cudaFree(d_pointsRect2);
    cudaFree(d_pointsRect3);
    cudaFree(d_pointsRect4);
}

//void updateRectPoints(Tetromino& fig, int dY) {
//    auto pointsRect1 = fig.getPointsRect1();
//    for (auto& point : pointsRect1) {
//        point.y -= dY;
//    }
//    fig.setPointsRect1(pointsRect1);
//
//    auto pointsRect2 = fig.getPointsRect2();
//    for (auto& point : pointsRect2) {
//        point.y -= dY;
//    }
//    fig.setPointsRect2(pointsRect2);
//
//    auto pointsRect3 = fig.getPointsRect3();
//    for (auto& point : pointsRect3) {
//        point.y -= dY;
//    }
//    fig.setPointsRect3(pointsRect3);
//
//    auto pointsRect4 = fig.getPointsRect4();
//    for (auto& point : pointsRect4) {
//        point.y -= dY;
//    }
//    fig.setPointsRect4(pointsRect4);
//}
__global__ void incrementYlinesKernel(int* lines, const Point* rect, int rectSize, int scale, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rectSize) return;

    int Ymin = rect[idx].y;
    int y = Ymin / (scale * 11);

    if (Ymin < height) {
        atomicAdd(&lines[y], 1);
    }
}

void incrementYlines(std::map<int, int>& lines, const std::vector<Point>& rect) {
    if (rect.empty()) return;

    // Находим минимальное значение Y в векторе точек
    auto minPointIt = std::min_element(rect.begin(), rect.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
        });

    if (minPointIt != rect.end()) {
        int Ymin = minPointIt->y;
        int y = Ymin / (scale * 11);

        if (Ymin < height) {
            lines[y]++;
        }
    }
}

void incrementYlinesCuda(std::map<int, int>& lines, const std::vector<Point>& rect) {
    if (rect.empty()) return;

    int rectSize = rect.size();
    Point* d_rect;
    int* d_lines;
    int maxLines = height / (scale * 11) + 1;

    std::vector<int> linesVec(maxLines, 0);

    cudaMalloc(&d_rect, rectSize * sizeof(Point));
    cudaMalloc(&d_lines, maxLines * sizeof(int));
    cudaMemcpy(d_rect, rect.data(), rectSize * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lines, linesVec.data(), maxLines * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (rectSize + blockSize - 1) / blockSize;
    incrementYlinesKernel << <numBlocks, blockSize >> > (d_lines, d_rect, rectSize, scale, height);
    cudaDeviceSynchronize();

    cudaMemcpy(linesVec.data(), d_lines, maxLines * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_rect);
    cudaFree(d_lines);

    for (int i = 0; i < maxLines; ++i) {
        if (linesVec[i] > 0) {
            lines[i] += linesVec[i];
        }
    }
}

//void incrementYlines(std::map<int, int>& lines, const std::vector<Point>& rect) {
//    // Находим минимальное значение Y в векторе точек
//    auto minPointIt = std::min_element(rect.begin(), rect.end(), [](const Point& a, const Point& b) {
//        return a.y < b.y;
//        });
//
//    if (minPointIt != rect.end()) {
//        int Ymin = minPointIt->y;
//        int y = Ymin / (scale * 11);
//
//        if (Ymin < height) {
//            // Увеличиваем количество заполненных клеток в соответствующей строке
//            lines[y]++;
//        }
//    }
//}
__global__ void deleteLineRectKernel(Point* rect, int rectSize, int coordY, int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rectSize) return;

    // Определяем минимальное значение Y с помощью атомарной операции
    __shared__ int minY;
    if (threadIdx.x == 0) {
        minY = rect[0].y;
        for (int i = 1; i < rectSize; i++) {
            atomicMin(&minY, rect[i].y);
        }
    }
    __syncthreads();

    if (coordY == minY) {
        rect[idx].x = INT_MAX;
        rect[idx].y = INT_MAX;
    }
    else if (coordY > minY) {
        rect[idx].y += scale * 11;
    }
}

void DeleteLineRect(std::vector<Point>& rect, int coordY) {
    if (rect.empty()) return;

    int rectSize = rect.size();
    Point* d_rect;

    cudaMalloc(&d_rect, rectSize * sizeof(Point));
    cudaMemcpy(d_rect, rect.data(), rectSize * sizeof(Point), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (rectSize + blockSize - 1) / blockSize;
    deleteLineRectKernel << <numBlocks, blockSize >> > (d_rect, rectSize, coordY, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(rect.data(), d_rect, rectSize * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaFree(d_rect);
}
//void DeleteLineRect(std::vector<Point>& rect, int coordY) {
//    // Находим минимальное значение Y в векторе точек
//    auto minYIt = std::min_element(rect.begin(), rect.end(), [](const Point& a, const Point& b) {
//        return a.y < b.y;
//        });
//
//    if (minYIt == rect.end()) {
//        return; // Если вектор пустой, выходим из функции
//    }
//
//    int minY = minYIt->y;
//
//    std::vector<Point> newRect = rect; // Создаем копию вектора
//
//    if (coordY == minY) {
//        for (auto& point : newRect) {
//            point.x = std::numeric_limits<int>::max();
//            point.y = std::numeric_limits<int>::max();
//        }
//    }
//    else if (coordY > minY) {
//        for (auto& point : newRect) {
//            point.y += scale * 11;
//        }
//    }
//
//    rect = newRect; // Заменяем исходный вектор модифицированной копией
//}

void DrawButton(HDC hdc, RECT rect, const char* text)
{
    // Рисуем кнопку
    DrawEdge(hdc, &rect, EDGE_RAISED, BF_RECT);

    // Заполняем фон кнопки белым цветом
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 255, 255));
    FillRect(hdc, &rect, hBrush);
    DeleteObject(hBrush);

    // Устанавливаем режим прозрачного текста
    SetBkMode(hdc, TRANSPARENT);

    // Преобразуем const char* в LPCWSTR
    wchar_t wtext[100];
    MultiByteToWideChar(CP_ACP, 0, text, -1, wtext, 100);

    // Выводим текст на кнопке
    DrawText(hdc, text, -1, &rect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

void GameOver(HWND hwnd)
{
    // Остановить таймер
    KillTimer(hwnd, IDT_TIMER1);

    // Создаем кнопку "New Game"
    CreateWindowEx(
        0,
        "BUTTON",  // Predefined class; Unicode assumed
        "New Game",      // Button text
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles
        100,         // x position
        200,         // y position
        150,        // Button width
        50,        // Button height
        hwnd,       // Parent window
        (HMENU)ID_NEW_GAME_BUTTON,       // Button ID
        (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE),
        NULL);      // Pointer not needed

    // Создаем кнопку "Exit"
    CreateWindowEx(
        0,
        "BUTTON",  // Predefined class; Unicode assumed
        "Exit",      // Button text
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles
        300,         // x position
        200,         // y position
        150,        // Button width
        50,        // Button height
        hwnd,       // Parent window
        (HMENU)ID_EXIT_BUTTON,       // Button ID
        (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE),
        NULL);      // Pointer not needed

    // Обновляем окно
    UpdateWindow(hwnd);
}

void DestroyButtons(HWND hwnd)
{
    // Удаляем кнопку "New Game"
    while (GetDlgItem(hwnd, ID_NEW_GAME_BUTTON)) {
        DestroyWindow(GetDlgItem(hwnd, ID_NEW_GAME_BUTTON));
    }

    // Удаляем кнопку "Exit"
    while (GetDlgItem(hwnd, ID_EXIT_BUTTON)) {
        DestroyWindow(GetDlgItem(hwnd, ID_EXIT_BUTTON));
    }

    // Обновляем окно
    InvalidateRect(hwnd, NULL, TRUE);
    UpdateWindow(hwnd);
}


void SearchLines(std::vector<Tetromino>& Figrnd_out)
{

    int Ny = height / (11 * scale) + 1;
    int Nx = width / (11 * scale);
    // Создаем словарь для хранения количества заполненных клеток в каждой строке
    std::map<int, int> lines;

    // Инициализируем словарь значениями 0
    for (int i = 0; i < Ny; ++i) {
        lines[i] = 0;
    }


    // Перебираем все элементы Figrnd_out
    for (Tetromino& fig : Figrnd_out)
    {

        incrementYlines(lines, fig.getPointsRect1());
        incrementYlines(lines, fig.getPointsRect2());
        incrementYlines(lines, fig.getPointsRect3());
        incrementYlines(lines, fig.getPointsRect4());
    }
    std::vector<int> fullLines{};
    // Проверяем, какие строки полностью заполнены
    for (auto line : lines)
    {
        // Если количество заполненных клеток в строке равно ширине поля, то строка полностью заполнена
        if (line.second == Nx)
        {
            // Добавляем номер строки в список полностью заполненных строк
            fullLines.push_back(line.first * scale * 11);
        }
    }

    if (lines[0] != 0)
    {
        StopGame = true;
    }

    int sc = 0;
    for (int coordY : fullLines)
    {
        for (Tetromino& fig : Figrnd_out) {
            std::vector<Point> points1 = fig.getPointsRect1();
            std::vector<Point> points2 = fig.getPointsRect2();
            std::vector<Point> points3 = fig.getPointsRect3();
            std::vector<Point> points4 = fig.getPointsRect4();

            DeleteLineRect(points1, coordY);
            DeleteLineRect(points2, coordY);
            DeleteLineRect(points3, coordY);
            DeleteLineRect(points4, coordY);

            fig.setPointsRect1(points1);
            fig.setPointsRect2(points2);
            fig.setPointsRect3(points3);
            fig.setPointsRect4(points4);
        }
        sc++;
    }
    if (sc == 1) { Score += 100; }
    else if (sc == 2) { Score += 300; }
    else if (sc == 3) { Score += 500; }
    else if (sc == 4) { Score += 800; }

}
__device__ int findMinXInRect(const Point* rect, int rectSize) {
    int minX = INT_MAX;
    for (int i = 0; i < rectSize; ++i) {
        if (rect[i].x < minX) {
            minX = rect[i].x;
        }
    }
    return minX;
}

__global__ void findMinXInTetrominoKernel(const Point* points, int* minX, int rectSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4) return;

    minX[idx] = findMinXInRect(points + idx * rectSize, rectSize);
}

__device__ int findMaxXInRect(const Point* rect, int rectSize) {
    int maxX = INT_MIN;
    for (int i = 0; i < rectSize; ++i) {
        if (rect[i].x > maxX) {
            maxX = rect[i].x;
        }
    }
    return maxX;
}

__global__ void findMaxXInTetrominoKernel(const Point* points, int* maxX, int rectSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4) return;

    maxX[idx] = findMaxXInRect(points + idx * rectSize, rectSize);
}

int findMinXInTetromino(const Tetromino& tetromino) {
    std::vector<std::vector<Point>> points = {
        tetromino.getPointsRect1(),
        tetromino.getPointsRect2(),
        tetromino.getPointsRect3(),
        tetromino.getPointsRect4()
    };

    int* d_minX;
    cudaMalloc(&d_minX, 4 * sizeof(int));

    Point* d_points;
    cudaMalloc(&d_points, 4 * points[0].size() * sizeof(Point));

    for (int i = 0; i < 4; ++i) {
        cudaMemcpy(d_points + i * points[i].size(), points[i].data(), points[i].size() * sizeof(Point), cudaMemcpyHostToDevice);
    }

    int blockSize = 256;
    int numBlocks = (4 + blockSize - 1) / blockSize;

    findMinXInTetrominoKernel << <numBlocks, blockSize >> > (d_points, d_minX, points[0].size());
    cudaDeviceSynchronize();

    std::vector<int> minX(4);
    cudaMemcpy(minX.data(), d_minX, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_minX);
    cudaFree(d_points);

    return *std::min_element(minX.begin(), minX.end());
}

int findMaxXInTetromino(const Tetromino& tetromino) {
    std::vector<std::vector<Point>> points = {
        tetromino.getPointsRect1(),
        tetromino.getPointsRect2(),
        tetromino.getPointsRect3(),
        tetromino.getPointsRect4()
    };

    int* d_maxX;
    cudaMalloc(&d_maxX, 4 * sizeof(int));

    Point* d_points;
    cudaMalloc(&d_points, 4 * points[0].size() * sizeof(Point));

    for (int i = 0; i < 4; ++i) {
        cudaMemcpy(d_points + i * points[i].size(), points[i].data(), points[i].size() * sizeof(Point), cudaMemcpyHostToDevice);
    }

    int blockSize = 256;
    int numBlocks = (4 + blockSize - 1) / blockSize;

    findMaxXInTetrominoKernel << <numBlocks, blockSize >> > (d_points, d_maxX, points[0].size());
    cudaDeviceSynchronize();

    std::vector<int> maxX(4);
    cudaMemcpy(maxX.data(), d_maxX, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_maxX);
    cudaFree(d_points);

    return *std::max_element(maxX.begin(), maxX.end());
}

//int findMinXInTetromino(const Tetromino& tetromino) {
//    int minX = std::numeric_limits<int>::max();
//
//    // Векторы точек в Tetromino
//    std::vector<std::vector<Point>> points = {
//        tetromino.getPointsRect1(),
//        tetromino.getPointsRect2(),
//        tetromino.getPointsRect3(),
//        tetromino.getPointsRect4()
//    };
//
//    for (const auto& rect : points) {
//        auto minIt = std::min_element(rect.begin(), rect.end(), [](const Point& a, const Point& b) {
//            return a.x < b.x;
//            });
//
//        if (minIt != rect.end() && minIt->x < minX) {
//            minX = minIt->x;
//        }
//    }
//
//    return minX;
//}
//
//int findMaxXInTetromino(const Tetromino& tetromino) {
//    int maxX = std::numeric_limits<int>::min();
//
//    // Векторы точек в Tetromino
//    std::vector<std::vector<Point>> points = {
//        tetromino.getPointsRect1(),
//        tetromino.getPointsRect2(),
//        tetromino.getPointsRect3(),
//        tetromino.getPointsRect4()
//    };
//
//    for (const auto& rect : points) {
//        auto maxIt = std::max_element(rect.begin(), rect.end(), [](const Point& a, const Point& b) {
//            return a.x < b.x;
//            });
//
//        if (maxIt != rect.end() && maxIt->x > maxX) {
//            maxX = maxIt->x;
//        }
//    }
//
//    return maxX;
//}

// Двойная буферизация
HBITMAP hbmMem = NULL;
HDC hdcMem = NULL;
HBITMAP hbmOld = NULL;
RECT clientRect;


// Функция для включения двойной буферизации
void SetDoubleBuffered(HWND hwnd, BOOL enabled) {
    if (enabled) {
        SetWindowLong(hwnd, GWL_STYLE, GetWindowLong(hwnd, GWL_STYLE) | WS_EX_COMPOSITED);
    }
}
void StartTimer(HWND hwnd)
{
    SetTimer(hwnd, IDT_TIMER1, 10, NULL);
}
void Startgame()
{
    // Обновляем окно

    Figrnd_out.clear();
    Score = 0;
    std::mt19937 gen(rd()); // Инициализация генератора случайных чисел Mersenne Twister
    std::uniform_int_distribution<> distr(0, 6); // Инициализация равномерного распределения в заданном диапазоне
    int fig = distr(gen);
    mod = sizes[fig];
    std::uniform_int_distribution<> distr2(0, mod - 1);
    current_mod = distr2(gen);
    Figrnd_group = Figs[fig];

    int deltaX = ((width / 2) / (11 * scale)) * 11 * scale - 11 * scale;
    for (int i = 0; i < mod; i++)
    {
        MoveXY(Figrnd_group[i], deltaX, 0);
    }

    Figrnd = Figrnd_group[current_mod];


    int Ydec = std::round(static_cast<int>(height) / (scale * 11));
    int Y0 = (int)Ydec * (scale * 11);
    //Y0 = 0;
    int stepX = scale * 11;
    for (int j = 0; j <= width / (stepX * 1); j += 4)
    {
        //  Figrnd_out.
        std::vector<Point> Rect1, Rect2, Rect3, Rect4;
        {
            Rect1 = { Point(j * stepX, Y0),  Point(j * stepX + scale * 10, Y0),  Point(j * stepX + scale * 10, Y0 + scale * 10),  Point(j * stepX, Y0 + scale * 10) };
            Rect2 = { Point((j + 1) * stepX, Y0),  Point((j + 1) * stepX + scale * 10, Y0),  Point((j + 1) * stepX + scale * 10, Y0 + scale * 10),  Point((j + 1) * stepX, Y0 + scale * 10) };
            Rect3 = { Point((j + 2) * stepX, Y0),  Point((j + 2) * stepX + scale * 10, Y0),  Point((j + 2) * stepX + scale * 10, Y0 + scale * 10),  Point((j + 2) * stepX, Y0 + scale * 10) };
            Rect4 = { Point((j + 3) * stepX, Y0),  Point((j + 3) * stepX + scale * 10, Y0),  Point((j + 3) * stepX + scale * 10, Y0 + scale * 10),  Point((j + 3) * stepX, Y0 + scale * 10) };
            Figrnd_out.push_back(Tetromino(Rect1, Rect2, Rect3, Rect4));
        }
    }
}



LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {


    switch (uMsg) {

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case ID_NEW_GAME_BUTTON:
            // Обработка нажатия на кнопку "New Game"
            DestroyButtons(hwnd);

            StopGame = false;
            UpdateWindow(hwnd);
            Startgame();
            // InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно
            StartTimer(hwnd); // Запуск таймера заново
            // Логика для новой игры...
            break;
        case ID_EXIT_BUTTON:
            // Обработка нажатия на кнопку "Exit"
            DestroyButtons(hwnd);
            PostQuitMessage(0);
            break;
        }
        break;


    case WM_KEYDOWN:
        switch (wParam) {
        case 'W':
        case VK_UP:
        {
            // MoveXY(Figrnd_group[i], 0, 1);
            int max = Figrnd_group.size();
            if (current_mod + 1 < max)
            {
                ++current_mod;
            }
            else { current_mod = 0; }
            Figrnd = Figrnd_group[current_mod];


            // InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно
            break;
        }
        case 'S':
        case VK_DOWN:
        {
            // MoveXY(Figrnd_group[i], 0, 1);
            int max = Figrnd_group.size();
            if (current_mod - 1 >= 0)
            {
                --current_mod;
            }
            else { current_mod = max - 1; }
            Figrnd = Figrnd_group[current_mod];


            // InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно
            break;
        }
        case 'A':
        case VK_LEFT:
        {
            int max = Figrnd_group.size();
            int min = width;
            int minx = 0;

            for (int i = 0; i < max; i++) {
                minx = findMinXInTetromino(Figrnd_group[i]);

                if (minx < min) { min = minx; }
            }
            if (min > 0) {
                for (int i = 0; i < max; i++) {
                    MoveXY(Figrnd_group[i], -scale * 11, 0);
                }
                //  InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно
            }
            break;
        }
        case 'D':
        case VK_RIGHT:
        {
            int max = Figrnd_group.size();
            int max_ = 0;
            int maxx = 0;

            for (int i = 0; i < max; i++) {
                maxx = findMaxXInTetromino(Figrnd_group[i]);

                if (maxx > max_) { max_ = maxx; }
            }
            int w = (11 * scale) * width / (11 * scale);

            for (int i = 0; i < max; i++) {
                maxx = findMaxXInTetromino(Figrnd_group[i]);
                if (maxx + (11 * scale) <= width) {
                    MoveXY(Figrnd_group[i], scale * 11, 0);
                }
            }
            //   InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно

            break;
        }
        case VK_SPACE:
        case VK_RETURN:
        {
            speed_moveY = 10;
            // InvalidateRect(hwnd, NULL, TRUE); // Перерисовываем окно
            break;
        }
        break;
        }



    case WM_CREATE:
    {
        SetDoubleBuffered(hwnd, true);
        SetTimer(hwnd, IDT_TIMER1, 10, NULL);
        GetClientRect(hwnd, &clientRect);
        hdcMem = CreateCompatibleDC(NULL);
        hbmMem = CreateCompatibleBitmap(GetDC(hwnd), clientRect.right - clientRect.left, clientRect.bottom - clientRect.top);
        hbmOld = (HBITMAP)SelectObject(hdcMem, hbmMem);
        break;
    }

    case WM_TIMER:
    {
        if (StopGame) {

            KillTimer(hwnd, IDT_TIMER1);
            break;
        }
        else {}

        // Проверяем, что сообщение пришло от нашего таймера
        if (wParam == IDT_TIMER1)
        {
            Figrnd = Figrnd_group[current_mod];

            for (int i = 0; i < mod; i++)
            {
                if (
                    Moving(Figrnd, Figrnd_out)
                    )
                {

                    MoveXY(Figrnd_group[i], 0, speed_moveY);
                }
                else
                {
                    speed_moveY = 2;
                    int dY = maxFixY(Figrnd, height);
                    // выравнивание позиций Y

                    updateRectPoints(Figrnd, dY);


                    Figrnd_out.push_back(Figrnd);

                    SearchLines(Figrnd_out);



                    std::mt19937 gen(rd()); // Инициализация генератора случайных чисел Mersenne Twister
                    std::uniform_int_distribution<> distr(0, 6); // Инициализация равномерного распределения в заданном диапазоне
                    int fig = distr(gen);
                    mod = sizes[fig];
                    std::uniform_int_distribution<> distr2(0, mod - 1);
                    current_mod = distr2(gen);
                    Figrnd_group = Figs[fig];


                    int deltaX = ((width / 2) / (11 * scale)) * 11 * scale - 11 * scale;
                    for (int i = 0; i < mod; i++)
                    {
                        MoveXY(Figrnd_group[i], deltaX, 0);
                    }

                    Figrnd = Figrnd_group[current_mod];
                }
            }


            // Вызываем функцию для перерисовки всего окна
            InvalidateRect(hwnd, NULL, FALSE); // Обновляем только нужную область
        }
        break;
    }

    case WM_DESTROY:

        if (hdcMem) {
            SelectObject(hdcMem, hbmOld);
            DeleteObject(hbmMem);
            DeleteDC(hdcMem);
        }
        PostQuitMessage(0);
        return 0;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);



        // Получаем размер клиентской области
        RECT clientRect;
        GetClientRect(hwnd, &clientRect);
        width = clientRect.right - clientRect.left;
        height = clientRect.bottom - clientRect.top;

        // Создаем контекст памяти для двойной буферизации
        HDC hdcMem = CreateCompatibleDC(hdc);
        HBITMAP hbmMem = CreateCompatibleBitmap(hdc, width, height);
        HGDIOBJ hOld = SelectObject(hdcMem, hbmMem);

        // Заполняем фон черным цветом
        HBRUSH blackBrush = (HBRUSH)GetStockObject(BLACK_BRUSH);
        FillRect(hdcMem, &clientRect, blackBrush);



        // Рисуем тетромино на контексте устройства в памяти
        COLORREF blueColor = RGB(0, 0, 255);
        DrawTetrominoRects(hdcMem, Figrnd, blueColor);
        for (int i = 0; i < Figrnd_out.size(); i++) {
            DrawTetrominoRects(hdcMem, Figrnd_out[i], blueColor);
        }

        // Создаем и выбираем перо
        HPEN hPen = CreatePen(PS_SOLID, 1, RGB(10, 10, 10));
        HPEN hOldPen = (HPEN)SelectObject(hdcMem, hPen);

        for (int i = 0; i < height / (scale * 11); i++) {
            // Задаем начальную позицию
            MoveToEx(hdcMem, i * scale * 11, 0, NULL); // Начальная точка
            // Рисуем линию до точки
            LineTo(hdcMem, i * scale * 11, height);
        }

        // Устанавливаем прозрачный фон для текста
        SetBkMode(hdcMem, TRANSPARENT);

        // Устанавливаем цвет текста в белый
        SetTextColor(hdcMem, RGB(255, 255, 255));

        // Создаем текст для отображения


        // Создаем шрифт
        HFONT hFont = CreateFont(
            26,                        // Высота шрифта
            0,                         // Средняя ширина символов
            0,                         // Угол наклона текста
            0,                         // Угол наклона базовой линии
            FW_NORMAL,                 // Толщина шрифта
            TRUE,                      // Курсив
            FALSE,                     // Подчеркнутый
            FALSE,                     // Зачеркнутый
            DEFAULT_CHARSET,           // Набор символов
            OUT_DEFAULT_PRECIS,        // Точность вывода
            CLIP_DEFAULT_PRECIS,       // Точность отсечения
            DEFAULT_QUALITY,           // Качество вывода
            FIXED_PITCH | FF_MODERN,   // Ширина и семейство шрифта
            TEXT("Monotype Corsiva")   // Имя шрифта
        );

        // Сохраняем старый шрифт, чтобы потом восстановить его
        HFONT hOldFont = (HFONT)SelectObject(hdcMem, hFont);

        std::string text = "Score " + std::to_string(Score);

        // Преобразуем std::string в std::wstring
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, &text[0], (int)text.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &text[0], (int)text.size(), &wstrTo[0], size_needed);

        TextOut(hdcMem, 10, 10, text.c_str(), wstrTo.length());



        if (StopGame) {
            GameOver(hwnd);
            StopGame = false;
        }
        else {}


        // Восстанавливаем старый шрифт и удаляем созданный шрифт
        SelectObject(hdcMem, hOldFont);
        DeleteObject(hFont);

        // Восстанавливаем старое перо и удаляем созданное перо
        SelectObject(hdcMem, hOldPen);
        DeleteObject(hPen);

        // Копируем изображение из контекста устройства в памяти на экран
        BitBlt(hdc, 0, 0, clientRect.right - clientRect.left, clientRect.bottom - clientRect.top, hdcMem, 0, 0, SRCCOPY);

        // Освобождаем ресурсы
        SelectObject(hdcMem, hOld);
        DeleteObject(hbmMem);
        DeleteDC(hdcMem);

        EndPaint(hwnd, &ps);
        return 0;
    }

                 return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int  window()
{
    LPCSTR CLASS_NAME = "Sample Window Class";

    // Получение дескриптора экземпляра
    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASS wc = {};

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    //wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH); // Установка фона окна в черный цвет
    // Регистрация оконного класса
    RegisterClass(&wc);

    // Создание окна
    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        "Sample Window",
        WS_OVERLAPPEDWINDOW,
        // Размеры и положение окна
        CW_USEDEFAULT, CW_USEDEFAULT,  // Position
        676, 1062,                      // Размеры окна (ширина x высота)
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hwnd == NULL) {
        return 0;
    }

    // Показ окна
    ShowWindow(hwnd, SW_SHOW);

    // Цикл обработки сообщений
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
        //  InvalidateRect(hwnd, NULL, TRUE);
    }
    return 0;
}


int main()
{
    Figs.push_back({ TetrominoFactory::CreateFig1(), TetrominoFactory::CreateFig1_1() });
    Figs.push_back({ TetrominoFactory::CreateFig2(), TetrominoFactory::CreateFig2_1(), TetrominoFactory::CreateFig2_2(), TetrominoFactory::CreateFig2_3() });
    Figs.push_back({ TetrominoFactory::CreateFig3(), TetrominoFactory::CreateFig3_1(), TetrominoFactory::CreateFig3_2(), TetrominoFactory::CreateFig3_3() });
    Figs.push_back({ TetrominoFactory::CreateFig4(), TetrominoFactory::CreateFig4_1(), TetrominoFactory::CreateFig4_2(), TetrominoFactory::CreateFig4_3() });
    Figs.push_back({ TetrominoFactory::CreateFig5() });
    Figs.push_back({ TetrominoFactory::CreateFig6(), TetrominoFactory::CreateFig6_1() });
    Figs.push_back({ TetrominoFactory::CreateFig7(), TetrominoFactory::CreateFig7_1() });

    Scale(scale, Figs);

    Startgame();

    window();


    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
