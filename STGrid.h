#pragma once
#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"

class STGrid {

public:
	// ���� grid �ĵ��� ���������ȫ�ֵ��� ���� CPU GPU ���� GAT really good? may be not, what real project looks like?
	// really in this way. must done before 7/17

	vector<STTrajectory> dataPtr; // С���ɣ���������/ָ�� һ�γ�ʼ����㲻��Ҫÿ�ζ���vector<STTrajectory> trajDB ��Ϊ���� �򻯳���
	vector<size_t> taskSet1, taskSet2;
	
	vector<trajPair> totaltaskCPU; // �Ƿ��̫�󣿣���

								   
	// functions started here
	void init(const vector<STTrajectory> &dptr);


	
	void STSimilarityJoinGPU();


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
		//double epsilon,
		//double alpha,
		int sizeP,
		int sizeQ,
		map<trajPair, double>& result
		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	// no filter and verification here
	void joinExhaustedCPU(
		//double epsilon,
		//double alpha,
		int sizeP,
		int sizeQ,
		map<trajPair, double>& result
		//vector<STTrajectory> &P,
		//vector<STTrajectory> &Q
	);

	void STSimilarityJoinCalcCPU(
		//double epsilon,
		//double alpha,
		const STTrajectory &T1,
		const STTrajectory &T2,
		map<trajPair, double>& result
	);

	// const ���߳̿���
	void STSimilarityJoinCalcCPUV2(
		const STTrajectory &T1,
		const STTrajectory &T2,
		double result
	);

protected:


private:



};