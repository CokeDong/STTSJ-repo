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
// �����ѭ������ ����Ҫ
// way1: ���� extern ��ʽ
// way2: #include "util.h" / main������
// ���벻ͨ�� : makefileû����

using namespace std;

//#define DIS_SIM_RESULT

#define MAX_CPU_THREAD 16384
#define MAX_TRAJ 100

// �⼸���궨�����
#define MAX_DIST 300000 // okay
#define ALPHA 0.5
#define EPSILON 0.85


#define GPUOnceCnt 128
#define DUALGPU false


// �⼸���궨����أ�THREADNUM = THREADROW * THREADCOLUMN

// ���� P Q num��ƽ�� not smart 16*16 is okay, only bound exsists imbalance!! 
// seems no impartant!
#define THREADROW 32
#define THREADCOLUMN 8

// ת����Ҫ����
#define THREADROW2 16
#define THREADCOLUMN2 16 // maybe better!!

#define THREADNUM 256 // = (THREADROW*THREADCOLUMN)



// NY: 200
// LA:
// TWITTER:
// predefined !!!
// ���ݼ���� < THREADNUM
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

	// ע�� double ���� ��Ȼfloat�������
	// linux ��
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

typedef pair<size_t, size_t> trajPair; // very smart �������ݽṹ


// 8 bytes [* 4 = 32 bytes(L2 Cache)]
typedef struct Latlon {
	float lat;
	float lon;
}Latlon;


// every task of GPU has a StatInfoTable
// 16 bytes now
typedef struct StatInfoTable {
	int latlonIdxP, latlonIdxQ; // starting id of latlon data for each traj (each task / block) in GPU
	int pointNumP, pointNumQ; // # of points in each traj

	
	//int textNumP, textNumQ; // total # word in each traj
	
	int textIdxP, textIdxQ; // starting position of text data for each task / block
	size_t keywordpmqnMatrixId, keywordpmqMatrixId, keywordpqMatrixId; // starting ID in GPU for each block
	int keycntP, keycntQ; // # of total keywords in each traj
	
	//int padding;?

}StatInfoTable;