#pragma once
#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"
#include "gpukernel.h"

class STGrid {

public:
	// 整个 grid 的调控 在这里进行全局调用 处理 CPU GPU 类似 GAT really good? may be not, what real project looks like?
	// really in this way. must done before 7/17

	vector<STTrajectory> dataPtr; // 小技巧：引入引用/指针 一次初始化后便不需要每次都把vector<STTrajectory> trajDB 作为参数 简化程序
	
	

								   


	// functions started here
	void init(const vector<STTrajectory> &dptr);



	// 宏观调用 需要优化?  函数调用是太频繁
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
	// 全速 all thread
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

	// const 多线程考虑
	void STSimilarityJoinCalcCPUV2(
		const STTrajectory &T1,
		const STTrajectory &T2,
		float &result // 引用传递
	);

	void STSimilarityJoinCalcCPUV3(
		const STTrajectory *T1,
		const STTrajectory *T2,
		float *result // 不能用值传递 
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