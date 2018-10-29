#pragma once

// ͷ�ļ� �궨�� ����/�ඨ������

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

// not recommended!!
// self-definition util
// �����ѭ������ ����Ҫ
// way1: ���� extern ��ʽ
// way2: #include "util.h" / main������
// ���벻ͨ�� : makefileû����

using namespace std;


#define MAX_TRAJ 100

#define MAX_DIST 300 // too large!!
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



#ifdef WIN32
#include "WinTimer.h"
#else
#include <sys/time.h>

class MyTimer
{
public:
	MyTimer() {
	};
	float iStart;
	float iEnd;

	// linux ��
	float cpuSecond() {
		struct timeval tp;
		gettimeofday(&tp, NULL);
		return ((float)tp.tv_sec + (float)tp.tv_usec*1.e-6);
	}

	inline void start()
	{
		iStart = cpuSecond();
	}
	inline void stop()
	{
		iEnd = cpuSecond();
	}
	inline float elapse()
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
	uint32_t pointid;
	float keywordvalue;
}Pointtuple;

typedef pair<size_t, size_t> trajPair; // very smart �������ݽṹ


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