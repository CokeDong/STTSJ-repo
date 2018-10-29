#include "STGrid.h"




void STGrid::init(const vector<STTrajectory> &dptr) {

	dataPtr = dptr; // 常量引用传递

}


void STGrid::joinExhaustedCPUonethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, float>& result) {



	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;

	vector<size_t> taskSet1, taskSet2;
	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	//set<size_t> Q;
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
	float* tmpresult = new float[totaltaskCPU.size()];
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
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, float>& result) {
	
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;
	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951

	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	//set<size_t> Q;
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
	float* tmpresult = new float[totaltaskCPU.size()];
	vector<thread> thread_STSim;
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
	
		// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));
		
		// only calculation, no judgement
		thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV3), this, &dataPtr[totaltaskCPU[i].first], &dataPtr[totaltaskCPU[i].second], &tmpresult[i]));
		//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
	}
	std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));
	
	/*
	for (size_t i = 0; i < 20; i++) {
		cout << tmpresult[i] << endl;
	}
	*/

	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			result[totaltaskCPU[i]] = tmpresult[i];
		}
	}

	cout << "finalresult size: " << result.size() << endl;

	timer.stop();
	printf("CPU tiime: %f s\n", timer.elapse());

	delete[] tmpresult;

}



void STGrid::joinExhaustedCPUconfigurablethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, float>& result,
	int threadnum) {

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;


	vector<size_t> taskSet1, taskSet2;
	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	//set<size_t> Q;
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
	
	// 总是无法满载！！
	float* tmpresult = new float[totaltaskCPU.size()];
	
	for (size_t j = 0; j < totaltaskCPU.size(); j += threadnum) {		
		vector<thread> thread_STSim;
		for (size_t i = 0; i < (j + threadnum > totaltaskCPU.size() ? totaltaskCPU.size()-j :threadnum); i++) {
			// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));
			// only calculation, no judgement
			thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV3), this, &dataPtr[totaltaskCPU[i + j].first], &dataPtr[totaltaskCPU[i + j].second], &tmpresult[i + j]));
			//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
		}
		std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));
		//thread_STSim.clear();//no need 作用域即可
	}
	/*
	for (size_t i = 0; i < 20; i++) {
	cout << tmpresult[i] << endl;
	}
	*/

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
	//float epsilon,
	//float alpha,
	const STTrajectory &T1,
	const STTrajectory &T2,
	map<trajPair, float>& result
) {
	// aborted
	
}


// Question1：const必要么？ 貌似应该这样 最好这样 安全
// Question2：const不适合多线程么 引用传递不能用到多线程？ 貌似是
void STGrid::STSimilarityJoinCalcCPUV2(
	const STTrajectory &T1,
	const STTrajectory &T2,
	float &result // 不能用值传递 
){
	result = T1.CalcTTSTSim(T2);
	// aborted
}

// 指针传递
// 多线程
void STGrid::STSimilarityJoinCalcCPUV3(
	const STTrajectory *T1,
	const STTrajectory *T2,
	float *result // 不能用值传递 
) {

	(*result) = (*T1).CalcTTSTSim((*T2));

}



void STGrid::joinExhaustedGPU(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	map<trajPair, float>& result
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;
	
	set<size_t> P;
	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(i);
	}
	set<size_t> Q;
	for (size_t j = 0; j < sizeQ; j++) {
		taskSet2.push_back(j);
	}


	// filtering 




	/*
	// get the trajPAIR(candidate pair)
	for (size_t i = 0; i < taskSet1.size(); i++) {
		for (size_t j = 0; j < taskSet2.size(); j++) {
			//if(i != j){ // no need
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // 这样是否不利于 多线程？？是否有性能影响 
			totaltaskGPU.push_back(tmppair); // 这里不考虑GPU内存拷贝问题
											 //}
		}
	}

	cout << "totaltaskGPU size: " << totaltaskGPU.size() << endl;
	*/
	vector<STTrajectory> trajSetP, trajSetQ;
	
	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		vector<size_t> tmptaskp;
		vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
		// Pbatch
		for (size_t k = 0; k < ( i + GPUOnceCnt > taskSet1.size() ? taskSet1.size()-i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[i + k]);
			tmptaskp.push_back(i + k);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[j + k]);
				tmptaskq.push_back(j + k);
			}

			// get trajpair(taskpair)
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			vector<float> partialResult;			
			STSimilarityJoinCalcGPU(trajSetP, trajSetQ, partialResult);
			

			// insert new result
			for (size_t k = 0; k < partialResult.size(); k++) {
				if (partialResult[k] > EPSILON) {
					result[tmptaskGPU[k]] = partialResult[k];
				}
			}

			/*
			for (map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
				//if ( (*it).second > EPSILON ) {
				if (it->second > EPSILON) {
					result.insert(*it);
				}		
			}
			*/
		}
	
	}
	
	cout << "finalresult size: " << result.size() << endl;
	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());
}


void STGrid::GetTaskPair(vector<size_t> &taskp, vector<size_t> &taskq, vector<trajPair> &resultpair) {
	for (size_t i = 0; i < taskp.size(); i++) {
		for (size_t j = 0; j < taskq.size(); j++) {
			trajPair tmppair = trajPair(taskp[i],taskq[j]);
			resultpair.push_back(tmppair);
		}
	}
}