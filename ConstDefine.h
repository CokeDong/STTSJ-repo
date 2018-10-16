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
#define EPSILON 0.2



#ifdef WIN32
#include "WinTimer.h"
#else
#include <sys/time.h>

class MyTimer
{
public:
	MyTimer() {
	};
	double iStart;
	double iEnd;

	double cpuSecond() {
		struct timeval tp;
		gettimeofday(&tp, NULL);
		return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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
	double keywordvalue;
}Keywordtuple;


typedef struct Pointtuple {
	int pointid;
	double keywordvalue;
}Pointtuple;

typedef pair<size_t, size_t> trajPair; // very smart �������ݽṹ