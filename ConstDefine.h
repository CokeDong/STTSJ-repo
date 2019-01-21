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

#ifdef WIN32
#define DIS_RESULT
#endif

// not recommended!!
// self-definition util
// 易造成循环引用 不重要
// way1: 改用 extern 方式
// way2: #include "util.h" / main中引用
// 编译不通过 : makefile没更新

using namespace std;

//#define DIS_SIM_RESULT

#define MAX_CPU_THREAD 16384
#define MAX_TRAJ 100

// 这几个宏定义相关
#define MAX_DIST 300000 // okay
#define ALPHA 0.5
#define EPSILON 0.85


#define GPUOnceCnt 128
#define DUALGPU false

#define SIZE_DATA 16


// 这几个宏定义相关：THREADNUM = THREADROW * THREADCOLUMN

// 处理 P Q num不平衡 not smart 16*16 is okay, only bound exsists imbalance!! 
// seems no impartant!
#define THREADROW 16
#define THREADCOLUMN 16

// 转置需要方阵
#define THREADROW2 16
#define THREADCOLUMN2 16 // maybe better!!

#define THREADNUM 256 // = (THREADROW*THREADCOLUMN)

// NY: 200
// LA:
// TWITTER:
// predefined !!!
// 数据集相关 must < THREADNUM, kernel is correct
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
	int keywordid;
	float keywordvalue;
}Keywordtuple;


// only for invert-list
typedef struct Pointtuple {
	int pointid;
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
	
	int latlonIdxP, latlonIdxQ; // starting id of latlon data for each traj (each task / block) in GPU, accumulated
	
	int pointNumP, pointNumQ; // # of points in each traj, excluding padding!  -->> the significance of padding may not that big!


	int keycntP, keycntQ; // # of total keywords in each traj, including padding, not-accumulated
						  // # of word in each traj, similar to int pointNumP, pointNumQ; // # of points in each traj


	// only used in kernel-V2
	int textIdxP, textIdxQ; // starting position of text data for each traj (each task / block), accumulated

	// used in kernel-V2 + kernel-V3
	size_t keywordpmqnMatrixId, keywordpmqMatrixId, keywordpqMatrixId; // starting ID in GPU for each block, accumulated
	
	// for v4
	size_t DensepqIdx; // Id ~= Idx
	
	//int padding;?
	//int keycntPnoPadding, keycntQnoPadding;
	

}StatInfoTable;



typedef struct TrajStatTable {

	// only to Traj, common attributes
	int latlonIdx;
	int pointNum;
	int keycnt;
	int textIdx;

	// for the input para. of Ccusparse<t>csrgemm for qkq ppk
	size_t csrRowPtrIdx, csrColIndIdx, csrValIdx; // only for v4
	size_t nnz;
	int row, col;

}TrajStatTable;