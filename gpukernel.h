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
	vector<float> &result);

void STSimilarityJoinCalcGPUNoZeroCopy(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	float *result);

void STSimilarityJoinCalcGPUV2(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result);

void STSimilarityJoinCalcGPUV2p1(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result);