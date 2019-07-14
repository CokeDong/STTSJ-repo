#pragma once
#include <vector>
#include "STTrajectory.h"
#include "STPoint.h"


//using namespace std;


// ������ host ��������������cpp�е���

/*
void CUDAwarmUp();
void* GPUMalloc(size_t byteNum);
*/


void STSimilarityJoinCalcGPU(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);


void STSimilarityJoinCalcGPUV2(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);


void STSimilarityJoinCalcGPUV2p1(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);



void STSimilarityJoinCalcGPUV3(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);

void STSimilarityJoinCalcGPUV31(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);



void STSimilarityJoinCalcGPUV4(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result);




void STSimilarityJoinCalcGPUVmgpu(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result,int deviceid);


void STSimilarityJoinCalcGPUVmgpuNoF(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	std::vector<float> &result, int deviceid);



void STSimilarityJoinCalcGPUNoZeroCopy(std::vector<STTrajectory> &trajSetP,
	std::vector<STTrajectory> &trajSetQ,
	float* result);


