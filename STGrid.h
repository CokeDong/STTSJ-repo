#pragma once
#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"
#include "gpukernel.h"

//using namespace std;

class STGrid {
	
public:
	// ���� grid �ĵ��� ���������ȫ�ֵ��� ���� CPU GPU ���� GAT really good? may be not, what real project looks like?
	// really in this way. must done before 7/17

	std::vector<STTrajectory> dataPtr; // С���ɣ���������/ָ�� һ�γ�ʼ����㲻��Ҫÿ�ζ���std::vector<STTrajectory> trajDB ��Ϊ���� �򻯳���
	
	

								   


	// functions started here
	void init(const std::vector<STTrajectory> &dptr);



	// ��۵��� ��Ҫ�Ż�?  ����������̫Ƶ��
	//// unit test
	//// removed to STPoint.cpp
	//void PointPointSimS(STPoint &p1, STPoint &p2);
	//void PointPointSimT(STPoint &p1, STPoint &p2);
	//void PointPointSimST(STPoint &p1, STPoint &p2);
	//void PointTrajSim(STPoint &p, STTrajectory &t);
	//// remove to STTrajectory.cpp
	//void TrajTrajSim(STTrajectory &t1, STTrajectory &t2);


	// int: 2,147,483,647 actually is not big enough
	// 49001 * 49001 = 2,401,098,001




	void joinExhaustedCPUonethread(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue

		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);

	// no filter and verification here
	// ȫ�� all thread
	void joinExhaustedCPU(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue

		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);

	// only for test
	void joinExhaustedCPUconfigurablethread(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue,

		int threadnum
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);

	void STSimilarityJoinCalcCPU(
		//float epsilon,
		//float alpha,
		const STTrajectory &T1,
		const STTrajectory &T2,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue
	);

	// const ���߳̿���
	void STSimilarityJoinCalcCPUV2(
		const STTrajectory &T1,
		const STTrajectory &T2,
		float &result // ���ô���
	);

	void STSimilarityJoinCalcCPUV3(
		const STTrajectory *T1,
		const STTrajectory *T2,
		float *result // ������ֵ���� 
	);


	void joinExhaustedGPU(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue

		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);
	void joinExhaustedGPUNZC(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue

		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);
	void joinExhaustedGPUV2(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);


	void joinExhaustedGPUV2p1(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);

	void joinExhaustedGPUV3(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);


	void joinExhaustedGPUV4(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
	);

	void joinExhaustedGPU_Final(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//std::map<trajPair, float>& result
		std::vector<trajPair>& resultpair,
		std::vector<float>& resultvalue,
		//std::vector<STTrajectory> &P,
		//std::vector<STTrajectory> &Q
		int type
	);


	// ���ڶ���ĳ�Ա�������������Ὣ�����ȱ������������������û����
	void GetTaskPair(std::vector<size_t> &taskp, std::vector<size_t> &taskq, std::vector<trajPair> &resultpair);

protected:


private:



};