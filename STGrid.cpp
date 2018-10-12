#include "STGrid.h"




void STGrid::init(const vector<STTrajectory> &dptr) {

	dataPtr = dptr; // 常量引用传递

}


void STGrid::joinExhaustedCPUonethread(
	//double epsilon,
	//double alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, double>& result) {

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	set<size_t> P;
	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	set<size_t> Q;
	for (size_t j = 0; j < sizeQ; j++) {
		taskSet2.push_back(j);
	}


	// filtering 





	// get the trajPAIR(candidate pair)
	for (size_t i = 0; i < sizeP; i++) {
		for (size_t j = 0; j < sizeQ; j++) {
			//if(i != j){ // no need
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // 这样是否不利于 多线程？？是否有性能影响 
			totaltaskCPU.push_back(tmppair); // 这里不考虑GPU内存拷贝问题
											 //}
		}
	}

	cout << "totaltaskCPU size: " << totaltaskCPU.size() << endl;


	// 多线程同时读是可以的

	// 多线程写 引入tmpresult
	double* tmpresult = new double[totaltaskCPU.size()];
	//vector<thread> thread_STSim;
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {

		// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));

		// only calculation, no judgement
		//thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV2), this, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second], tmpresult[i]));
		//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
		STSimilarityJoinCalcCPUV2(dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second], tmpresult[i]);
	
	}
	//std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));

	for (size_t i = 0; i < 20; i++) {
		cout << tmpresult[i] << endl;
	}
	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			result[totaltaskCPU[i]] = tmpresult[i];
		}
	}

	cout << "finalresult size: " << result.size() << endl;

	delete[] tmpresult;

}



void STGrid::joinExhaustedCPU(
	//double epsilon,
	//double alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, double>& result) {
	
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	set<size_t> P;
	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	set<size_t> Q;
	for (size_t j = 0; j < sizeQ; j++) {
		taskSet2.push_back(j);
	}


	// filtering 





	// get the trajPAIR(candidate pair)
	for (size_t i = 0; i < sizeP; i++) {
		for (size_t j = 0; j < sizeQ; j++) {
			//if(i != j){ // no need
				trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // 这样是否不利于 多线程？？是否有性能影响 
				totaltaskCPU.push_back(tmppair); // 这里不考虑GPU内存拷贝问题
			//}
		}
	}

	cout << "totaltaskCPU size: "<<totaltaskCPU.size() << endl;
	

	// 多线程同时读是可以的
	
	// 多线程写 引入tmpresult
	double* tmpresult = new double[totaltaskCPU.size()];
	vector<thread> thread_STSim;
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
	
		// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));
		
		// only calculation, no judgement
		thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV2), this, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second], tmpresult[i]));
		//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
	}
	std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));

	for (size_t i = 0; i < 20; i++) {
		cout << tmpresult[i] << endl;
	}
	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			result[totaltaskCPU[i]] = tmpresult[i];
		}
	}

	cout << "finalresult size: " << result.size() << endl;

	delete[] tmpresult;

}



void STGrid::STSimilarityJoinCalcCPU(
	//double epsilon,
	//double alpha,
	const STTrajectory &T1,
	const STTrajectory &T2,
	map<trajPair, double>& result
) {
	// aborted
	
}


void STGrid::STSimilarityJoinCalcCPUV2(
	const STTrajectory &T1,
	const STTrajectory &T2,
	double &result // 不能用值传递 
){
	result = T1.CalcTTSTSim(T2);
}
