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
double calculateDistance(double LatA, double LonA, double LatB, double LonB); // Question: 内联函数如何实现？

void split(string s, string delim, vector<string>* ret);
void split2(string s, string delim, vector<string>* ret);
void split3(string s, string delim, vector<string>* ret);