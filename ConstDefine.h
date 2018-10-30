#pragma once

// 头文件 宏定义 变量/类定义声明

// global header file system util
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

#ifdef WIN32
#include "WinTimer.h"
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#ifdef WIN32
#define sleep(x) Sleep(x*1000)
#else

#endif


// not recommended!!
// self-definition util
// 易造成循环引用 不重要
// way1: 改用 extern 方式
// way2: #include "util.h" / main中引用
// 编译不通过 : makefile没更新

using namespace std;

#define MAX_CPU_THREAD 16384
#define MAX_TRAJ 100

#define MAX_DIST 40076000 // too large!!
#define ALPHA 0.5
#define EPSILON 0.1

#define GPUOnceCnt 128
#define DUALGPU false

#define THREADNUM 256

#define THREADROW 32
#define THREADCOLUMN 8

// NY: 200
// LA:
// TWITTER:
// predefined !!!
#define MAXTRAJLEN 256


// not recommended??


#ifdef WIN32
#else
class MyTimer
{
public:
	MyTimer() {
	};
	double iStart;
	double iEnd;

	// 注意 double 类型 不然float会出错！！
	// linux 秒
	double cpuSecond() {
		struct timeval tp;
		gettimeofday(&tp, NULL);
		return ((double)tp.tv_sec + (double)tp.tv_usec*1.0e-6);
	}

	inline void start()
	{
		iStart = cpuSecond();
	}
	inline void stop()
	{
		iEnd = cpuSecond();
	}
	inline double elapse()
	{
		return iEnd - iStart;
	}
};
#endif



typedef struct Keywordtuple {
	uint32_t keywordid;
	float keywordvalue;
}Keywordtuple;


// only for invert-list
typedef struct Pointtuple {
	uint32_t pointid;
	float keywordvalue;
}Pointtuple;

typedef pair<size_t, size_t> trajPair; // very smart 定义数据结构


// 8 bytes [* 4 = 32 bytes(L2 Cache)]
typedef struct Latlon {
	float lat;
	float lon;
}Latlon;


// every task of GPU has a StatInfoTable
// 16 bytes now
typedef struct StatInfoTable {
	uint32_t latlonIdxP, latlonIdxQ; // starting id of latlon data for each traj (each task / block)
	uint32_t pointNumP, pointNumQ; // # of points in each traj

	//uint32_t textIdxP, textIdxQ; // starting position of text data for each task / block
	//uint32_t textNumP, textNumQ; // total # word in each traj


}StatInfoTable;