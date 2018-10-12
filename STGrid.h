#pragma once
#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"

class STGrid {

public:
	// 整个 grid 的调控 在这里进行全局调用 处理 CPU GPU 类似 GAT really good? may be not, what real project looks like?
	// really in this way. must done before 7/17

	vector<STTrajectory> dataPtr; // 小技巧：引入引用/指针 一次初始化后便不需要每次都把vector<STTrajectory> trajDB 作为参数 简化程序
	vector<size_t> taskSet1, taskSet2;
	
	vector<trajPair> totaltaskCPU; // 是否会太大？？？

								   
	// functions started here
	void init(const vector<STTrajectory> &dptr);


	
	void STSimilarityJoinGPU();


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

	// const 多线程考虑
	void STSimilarityJoinCalcCPUV2(
		const STTrajectory &T1,
		const STTrajectory &T2,
		double result
	);

protected:


private:



};