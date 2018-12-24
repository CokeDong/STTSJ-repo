#include "STGrid.h"
#include "util.h"



void STGrid::init(const vector<STTrajectory> &dptr) {
	dataPtr = dptr; // 常量引用传递
}


void STGrid::joinExhaustedCPUonethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue) {


	MyTimer timer;
	timer.start();

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;

	vector<size_t> taskSet1, taskSet2;
	GetSample(taskSet1, taskSet2, sizeP, sizeQ);

	cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << endl;

	// filtering 








	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	GetTaskPair(taskSet1, taskSet2, totaltaskCPU);


	/*
	// get the trajPAIR(candidate pair)
	for (size_t i = 0; i < sizeP; i++) {
		for (size_t j = 0; j < sizeQ; j++) {
			//if(i != j){ // no need
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // 这样是否不利于 多线程？？是否有性能影响 
			totaltaskCPU.push_back(tmppair); // 这里不考虑GPU内存拷贝问题
											 //}
		}
	}
	*/

	
	// 舍弃 vector<STTrajectory> trajSetP, trajSetQ;
	// because STSimilarityJoinCalcCPUV3 only takes one trajectory from P and Q

	// cpy from mul-threads, not necessary maybe

	// 多线程同时读是可以的
	// 多线程写 引入tmpresult

	// only calculating here!
	float* tmpresult = new float[totaltaskCPU.size()];
	//vector<thread> thread_STSim;
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {

		// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));

		// only calculation, no judgement
		//thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV2), this, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second], tmpresult[i]));
		//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
		STSimilarityJoinCalcCPUV3(&dataPtr[totaltaskCPU[i].first], &dataPtr[totaltaskCPU[i].second], &tmpresult[i]);
	
	}
	//std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));


	/*
	for (size_t i = 0; i < 20; i++) {
		cout << tmpresult[i] << endl;
	}
	*/


	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			//result[totaltaskCPU[i]] = tmpresult[i];
			resultpair.push_back(totaltaskCPU[i]);
			resultvalue.push_back(tmpresult[i]);
		}
	}

	timer.stop();
	printf("CPU time: %f s\n", timer.elapse());
	
	// donot forget!
	delete[] tmpresult;

	cout << "finalresult size: " << resultpair.size() << endl;

	

}



void STGrid::joinExhaustedCPU(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue) {
	
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;
	GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << endl;

	// filtering 




	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	GetTaskPair(taskSet1, taskSet2, totaltaskCPU);
	/*
	// get the trajPAIR(candidate pair)
	for (size_t i = 0; i < sizeP; i++) {
		for (size_t j = 0; j < sizeQ; j++) {
			//if(i != j){ // no need
				trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // 这样是否不利于 多线程？？是否有性能影响 
				totaltaskCPU.push_back(tmppair); // 这里不考虑GPU内存拷贝问题
			//}
		}
	}
	*/
	


	// 多线程同时读是可以的
	// 多线程写 引入tmpresult

	// 多线程读
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

	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			//result[totaltaskCPU[i]] = tmpresult[i];
			resultpair.push_back(totaltaskCPU[i]);
			resultvalue.push_back(tmpresult[i]);
		}
	}

	timer.stop();
	printf("CPU time: %f s\n", timer.elapse());


	delete[] tmpresult;
	cout << "finalresult size: " << resultpair.size() << endl;
	


}



void STGrid::joinExhaustedCPUconfigurablethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue,
	int threadnum) {

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;
	MyTimer timer;
	timer.start();

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;

	vector<size_t> taskSet1, taskSet2;
	GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << endl;

	// filtering 








	vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	GetTaskPair(taskSet1, taskSet2, totaltaskCPU);


	// 多线程同时读是可以的
	// 多线程写 引入tmpresult
	
	// 总是无法满载！！
	// 原因：threadnum  不够大 ---> MAX_CPU_THREAD

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
	
	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			//result[totaltaskCPU[i]] = tmpresult[i];
			resultpair.push_back(totaltaskCPU[i]);
			resultvalue.push_back(tmpresult[i]);
		}
	}
	

	timer.stop();
	printf("CPU time: %f s\n", timer.elapse());

	delete[] tmpresult;
	cout << "finalresult size: " << resultpair.size() << endl;



}



void STGrid::STSimilarityJoinCalcCPU(
	//float epsilon,
	//float alpha,
	const STTrajectory &T1,
	const STTrajectory &T2,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
) {
	// aborted
	
}


// Question1：const必要么？ 貌似应该这样 最好这样 安全
// Question2：const不适合多线程么 引用传递不能用到多线程？ 貌似是：不可以使用常量引用
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
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;
	GetSample(taskSet1, taskSet2, sizeP, sizeQ);


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
	

	cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << endl;

	

	// have changed dic to vector
	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		vector<STTrajectory> trajSetP;
		vector<size_t> tmptaskp;
		
		// Pbatch
		for (size_t k = 0; k < ( i + GPUOnceCnt > taskSet1.size() ? taskSet1.size()-i : GPUOnceCnt); k++) {
			// debug: tiny bug here, wrong index
			//trajSetP.push_back(this->dataPtr[i + k]);
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			vector<STTrajectory> trajSetQ;
			vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			MyTimer timer2;
			timer2.start();
			vector<float> partialResult;			
			STSimilarityJoinCalcGPU(trajSetP, trajSetQ, partialResult);
			timer2.stop();
			printf("GPU time once: %f s\n", timer2.elapse());

			// insert new result
			for (size_t k = 0; k < partialResult.size(); k++) {
				if (partialResult[k] > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU[k]);
					resultvalue.push_back(partialResult[k]);
				}
			}

			
			//for (map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
			//	//if ( (*it).second > EPSILON ) {
			//	if (it->second > EPSILON) {
			//		result.insert(*it);
			//	}		
			//}
			
		}
	
	}

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	cout << "finalresult size: " << resultpair.size() << endl;

}



void STGrid::joinExhaustedGPUNZC(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;
	GetSample(taskSet1, taskSet2, sizeP, sizeQ);


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


	cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << endl;



	// have changed dic to vector
	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		vector<STTrajectory> trajSetP;
		vector<size_t> tmptaskp;

		// Pbatch
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			// debug: tiny bug here, wrong index
			//trajSetP.push_back(this->dataPtr[i + k]);
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			vector<STTrajectory> trajSetQ;
			vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			//vector<float> partialResult;
			
			MyTimer timer2;
			timer2.start();
			float* partialResult = new float[tmptaskGPU.size()];
			STSimilarityJoinCalcGPUNoZeroCopy(trajSetP, trajSetQ, partialResult);
			timer2.stop();
			printf("GPU time once: %f s\n", timer2.elapse());


			// insert new result
			for (size_t k = 0; k < tmptaskGPU.size(); k++) {
				if (partialResult[k] > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU[k]);
					resultvalue.push_back(partialResult[k]);
				}
			}
			delete[] partialResult;

			//for (size_t k = 0; k < partialResult.size(); k++) {
			//	if (partialResult[k] > EPSILON) {
			//		//result[tmptaskGPU[k]] = partialResult[k];
			//		resultpair.push_back(tmptaskGPU[k]);
			//		resultvalue.push_back(partialResult[k]);
			//	}
			//}


			//for (map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
			//	//if ( (*it).second > EPSILON ) {
			//	if (it->second > EPSILON) {
			//		result.insert(*it);
			//	}		
			//}

		}

	}

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	cout << "finalresult size: " << resultpair.size() << endl;

}



void STGrid::joinExhaustedGPUV2(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;

	GetSample(taskSet1, taskSet2, sizeP, sizeQ);


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

	cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << endl;

	//bedug: 注意作用域
	//vector<STTrajectory> trajSetP, trajSetQ;
	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		
		
		// Pbatch
		//bedug: 注意作用域
		vector<size_t> tmptaskp;
		vector<STTrajectory> trajSetP;
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}

		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			

			

			// Qbatch
			//bedug: 注意作用域
			vector<size_t> tmptaskq; 
			vector<STTrajectory> trajSetQ;			
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV2

			// get trajpair(taskpair)
			vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// only calculate the result here!!
			MyTimer timer2;
			timer2.start();
			vector<float> partialResult;
			STSimilarityJoinCalcGPUV2(trajSetP, trajSetQ, partialResult); // have big overload?
			timer2.stop();
			printf("GPU time once: %f s\n", timer2.elapse());

			// insert new result
			for (size_t k = 0; k < partialResult.size(); k++) {
				if (partialResult[k] > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU[k]);
					resultvalue.push_back(partialResult[k]);
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

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	cout << "finalresult size: " << resultpair.size() << endl;

}





void STGrid::joinExhaustedGPUV2p1(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;

	GetSample(taskSet1, taskSet2, sizeP, sizeQ);


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

	cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << endl;

	

	//vector<trajPair> resultpair;
	//vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		vector<STTrajectory> trajSetP;
		vector<size_t> tmptaskp;
		
									 // Pbatch
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			vector<STTrajectory> trajSetQ;
			vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			vector<float> partialResult;
			STSimilarityJoinCalcGPUV2p1(trajSetP, trajSetQ, partialResult);


			// insert new result
			for (size_t k = 0; k < partialResult.size(); k++) {
				if (partialResult[k] > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU[k]);
					resultvalue.push_back(partialResult[k]);
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

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	cout << "finalresult size: " << resultpair.size() << endl;

}



void STGrid::joinExhaustedGPUV3(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	vector<trajPair>& resultpair,
	vector<float>& resultvalue
	//vector<STTrajectory> &P,
	//vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	vector<size_t> taskSet1, taskSet2;

	GetSample(taskSet1, taskSet2, sizeP, sizeQ);


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

	cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << endl;

	//bedug: 注意作用域
	//vector<STTrajectory> trajSetP, trajSetQ;
	//vector<trajPair> resultpair;
	//vector<float> resultvalue;


	
	for (size_t i = 0; i < taskSet1.size(); i += taskSet1.size()) { // ONLY once


		// Pbatch
		//bedug: 注意作用域
		vector<size_t> tmptaskp;
		vector<STTrajectory> trajSetP;
		for (size_t k = 0; k < (i + taskSet1.size() > taskSet1.size() ? taskSet1.size() - i : taskSet1.size()); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}

		for (size_t j = 0; j < taskSet2.size(); j += taskSet2.size()) { // ONLY once


			// Qbatch
			//bedug: 注意作用域
			vector<size_t> tmptaskq;
			vector<STTrajectory> trajSetQ;
			for (size_t k = 0; k < (j + taskSet2.size() > taskSet2.size() ? taskSet2.size() - j : taskSet2.size()); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV3

			// get trajpair(taskpair)
			vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// only calculate the result here!!
			MyTimer timer2;
			timer2.start();
			vector<float> partialResult;
			STSimilarityJoinCalcGPUV3(trajSetP, trajSetQ, partialResult); // have big overload?
			timer2.stop();
			printf("GPU time once: %f s\n", timer2.elapse());

			// insert new result
			for (size_t k = 0; k < partialResult.size(); k++) {
				if (partialResult[k] > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU[k]);
					resultvalue.push_back(partialResult[k]);
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

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	cout << "finalresult size: " << resultpair.size() << endl;

}






// this is 类外定义
inline void STGrid::GetTaskPair(vector<size_t> &taskp, vector<size_t> &taskq, vector<trajPair> &resultpair) {
	for (size_t i = 0; i < taskp.size(); i++) {
		for (size_t j = 0; j < taskq.size(); j++) {
			trajPair tmppair = trajPair(taskp[i],taskq[j]);
			resultpair.push_back(tmppair);
		}
	}
}