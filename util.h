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

float calculateDistance(float LatA, float LonA, float LatB, float LonB);

void split(string s, string delim, vector<string>* ret);
void split2(string s, string delim, vector<string>* ret);
void split3(string s, string delim, vector<string>* ret);