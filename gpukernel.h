#pragma once
#include <vector>
#include "STTrajectory.h"
#include "STPoint.h"

using namespace std;


// ������ host ��������������cpp�е���

/*
void CUDAwarmUp();
void* GPUMalloc(size_t byteNum);
*/
void STSimilarityJoinCalcGPU(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	map<trajPair, float> &result);
