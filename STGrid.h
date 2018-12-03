#pragma once
#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"
#include "gpukernel.h"

class STGrid {

public:
	// ���� grid �ĵ��� ���������ȫ�ֵ��� ���� CPU GPU ���� GAT really good? may be not, what real project looks like?
	// really in this way. must done before 7/17

	vector<STTrajectory> dataPtr; // С���ɣ���������/ָ�� һ�γ�ʼ����㲻��Ҫÿ�ζ���vector<STTrajectory> trajDB ��Ϊ���� �򻯳���
	
	

								   


	// functions started here
	void init(const vector<STTrajectory> &dptr);



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
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue

		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	// no filter and verification here
	// ȫ�� all thread
	void joinExhaustedCPU(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue

		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	// only for test
	void joinExhaustedCPUconfigurablethread(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue,

		int threadnum
		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	void STSimilarityJoinCalcCPU(
		//float epsilon,
		//float alpha,
		const STTrajectory &T1,
		const STTrajectory &T2,
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue
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
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue

		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	void joinExhaustedGPUV2(
		//float epsilon,
		//float alpha,
		int sizeP,
		int sizeQ,
		//map<trajPair, float>& result
		vector<trajPair>& resultpair,
		vector<float>& resultvalue
		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	void GetTaskPair(vector<size_t> &taskp, vector<size_t> &taskq, vector<trajPair> &resultpair);

protected:


private:



};