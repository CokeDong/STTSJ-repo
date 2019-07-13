#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <utility>
#include <thread>
#include <algorithm>
#include <set>
#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <cstdlib>
#include "STTrajectory.h"

//using namespace std;


#pragma once

// only simple fuction, no while, no switch, no recursion
float calculateDistance(float LatA, float LonA, float LatB, float LonB); // Question: 内联函数如何实现？

void split(std::string s, std::string delim, std::vector<std::string>* ret);
void split2(std::string s, std::string delim, std::vector<std::string>* ret);
void split3(std::string s, std::string delim, std::vector<std::string>* ret);

void GetSample(std::vector<size_t> &taskSet1, std::vector<size_t> &taskSet2, int sizeP, int sizeQ);
void GetSample2(std::vector<STTrajectory> &dataptr,std::vector<size_t> &taskSet1, std::vector<size_t> &taskSet2, int sizeP, int sizeQ);
void GetSample_Sorting(std::vector<STTrajectory> &dataptr, std::vector<size_t> &taskSet1, std::vector<size_t> &taskSet2, int sizeP, int sizeQ);

// this is a wrong coding!! -> stupid!
//template<class T>
//int getlength(T& arr)
//{
//	return sizeof(arr) / sizeof(arr[0]);
//}


//void OutGPUMemNeeded(size_t pmqnid, size_t pmqid, size_t pqid);