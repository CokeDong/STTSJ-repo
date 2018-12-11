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

using namespace std;


#pragma once

// only simple fuction, no while, no switch, no recursion
float calculateDistance(float LatA, float LonA, float LatB, float LonB); // Question: 内联函数如何实现？

void split(string s, string delim, vector<string>* ret);
void split2(string s, string delim, vector<string>* ret);
void split3(string s, string delim, vector<string>* ret);

void GetSample(vector<size_t> &taskSet1, vector<size_t> &taskSet2, int sizeP, int sizeQ);

// this is a wrong coding!! -> stupid!
//template<class T>
//int getlength(T& arr)
//{
//	return sizeof(arr) / sizeof(arr[0]);
//}


//void OutGPUMemNeeded(size_t pmqnid, size_t pmqid, size_t pqid);