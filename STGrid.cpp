#include "STGrid.h"
#include "util.h"


//using namespace std;

extern std::vector<float> cpuonethreadtimes;
extern std::vector<float> cpumthreadtimes;
extern std::vector<float> gpucoarsetimes;
extern std::vector<float> gpufinetimes;
extern std::vector<float> gpufinenoFliptimes;
extern std::vector<float> gpufinenoSortingtimes;

void STGrid::init(const std::vector<STTrajectory> &dptr) {
	dataPtr = dptr; // 常量引用传递
}


void STGrid::joinExhaustedCPUonethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue,
	int sampletype) {


	MyTimer timer;
	timer.start();

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;

	std::vector<size_t> taskSet1, taskSet2;

	if (sampletype == 0) {
		GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	}
	else
		if (sampletype == 1) {
			GetSample2(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);

		}
		else
			if (sampletype == 2) {
				GetSample_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
			}
			else
				if (sampletype == 3) {
					GetSample_Filtering_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
				}
				else
				{
					printf("No Sample Strategy!\n"); 
					assert(0);
				}
	std::cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	// filtering 








	std::vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
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

	
	// 舍弃 std::vector<STTrajectory> trajSetP, trajSetQ;
	// because STSimilarityJoinCalcCPUV3 only takes one trajectory from P and Q

	// cpy from mul-threads, not necessary maybe

	// 多线程同时读是可以的
	// 多线程写 引入tmpresult

	// only calculating here!
	float* tmpresult = new float[totaltaskCPU.size()];
	//std::vector<thread> thread_STSim;
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


	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

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

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

	

}



void STGrid::joinExhaustedCPU(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue,
	int sampletype) {
	
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;


	//GetSample(taskSet1, taskSet2, sizeP, sizeQ);

	if (sampletype == 0) {
		GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	}
	else
		if (sampletype == 1) {
			GetSample2(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);

		}
		else
			if (sampletype == 2) {
				GetSample_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
			}
			else
				if (sampletype == 3) {
					GetSample_Filtering_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
				}
				else
				{
					printf("No Sample Strategy!\n");
					assert(0);
				}


	std::cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	// filtering 




	std::vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
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
	std::vector<std::thread> thread_STSim;
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
	
		// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));
		
		// only calculation, no judgement
		thread_STSim.push_back(std::thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV3), this, &dataPtr[totaltaskCPU[i].first], &dataPtr[totaltaskCPU[i].second], &tmpresult[i]));
		//threads_FD.push_back(thread(std::mem_fn(&Grid::FDCalculateParallelHandeler), this, &queryQueue[qID], &freqVectors[qID]));
	}
	std::for_each(thread_STSim.begin(), thread_STSim.end(), std::mem_fn(&std::thread::join));
	

	/*
	for (size_t i = 0; i < 20; i++) {
		cout << tmpresult[i] << endl;
	}
	*/

	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

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
	std::cout << "finalresult size: " << resultpair.size() << std::endl;
	


}



void STGrid::joinExhaustedCPUconfigurablethread(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue,
	int threadnum,
	int sampletype) {

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;
	MyTimer timer;
	timer.start();

	// only one TrajDB - selfjoin
	// get ID only one trajDB
	//set<size_t> P;

	std::vector<size_t> taskSet1, taskSet2;

	if (sampletype == 0) {
		GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	}
	else
		if (sampletype == 1) {
			GetSample2(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);

		}
		else
			if (sampletype == 2) {
				GetSample_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
			}
			else
				if (sampletype == 3) {
					GetSample_Filtering_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
				}
				else
				if (sampletype == 4) {
					GetSample_Filtering_NoSorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
				}
					else
					{
						printf("No Sample Strategy!\n");
						assert(0);
					}
	
	//GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	std::cout << "totaltaskCPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	// filtering 








	std::vector<trajPair> totaltaskCPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
	GetTaskPair(taskSet1, taskSet2, totaltaskCPU);


	// 多线程同时读是可以的
	// 多线程写 引入tmpresult
	

	// 总是无法满载！！
	// 原因：threadnum  不够大 ---> MAX_CPU_THREAD

	float* tmpresult = new float[totaltaskCPU.size()];
	
	for (size_t j = 0; j < totaltaskCPU.size(); j += threadnum) {		
		std::vector<std::thread> thread_STSim;
		for (size_t i = 0; i < (j + threadnum > totaltaskCPU.size() ? totaltaskCPU.size()-j :threadnum); i++) {
			// thread_STSim.push_back(thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPU),this, epsilon, alpha, dataPtr[totaltaskCPU[i].first], dataPtr[totaltaskCPU[i].second],result));
			// only calculation, no judgement
			thread_STSim.push_back(std::thread(std::mem_fn(&STGrid::STSimilarityJoinCalcCPUV3), this, &dataPtr[totaltaskCPU[i + j].first], &dataPtr[totaltaskCPU[i + j].second], &tmpresult[i + j]));
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
	
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	// get final results
	for (size_t i = 0; i < totaltaskCPU.size(); i++) {
		if (tmpresult[i] > EPSILON) {
			//result[totaltaskCPU[i]] = tmpresult[i];
			resultpair.push_back(totaltaskCPU[i]);
			resultvalue.push_back(tmpresult[i]);
		}
	}
	

	timer.stop();
	float times = timer.elapse();
	printf("CPU time: %f s\n", times);

	cpumthreadtimes.push_back(times);

	delete[] tmpresult;
	std::cout << "finalresult size: " << resultpair.size() << std::endl;



}



void STGrid::STSimilarityJoinCalcCPU(
	//float epsilon,
	//float alpha,
	const STTrajectory &T1,
	const STTrajectory &T2,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
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
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;
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
	

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	

	// have changed dic to std::vector
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		std::vector<STTrajectory> trajSetP;
		std::vector<size_t> tmptaskp;
		
		// Pbatch
		for (size_t k = 0; k < ( i + GPUOnceCnt > taskSet1.size() ? taskSet1.size()-i : GPUOnceCnt); k++) {
			// debug: tiny bug here, wrong index
			//trajSetP.push_back(this->dataPtr[i + k]);
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			std::vector<STTrajectory> trajSetQ;
			std::vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			MyTimer timer2;
			timer2.start();
			std::vector<float> partialResult;			
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

			
			//for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
			//	//if ( (*it).second > EPSILON ) {
			//	if (it->second > EPSILON) {
			//		result.insert(*it);
			//	}		
			//}
			
		}
	
	}

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}



void STGrid::joinExhaustedGPUNZC(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;
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


	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;



	// have changed dic to std::vector
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		std::vector<STTrajectory> trajSetP;
		std::vector<size_t> tmptaskp;

		// Pbatch
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			// debug: tiny bug here, wrong index
			//trajSetP.push_back(this->dataPtr[i + k]);
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			std::vector<STTrajectory> trajSetQ;
			std::vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			//std::vector<float> partialResult;
			
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


			//for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
			//	//if ( (*it).second > EPSILON ) {
			//	if (it->second > EPSILON) {
			//		result.insert(*it);
			//	}		
			//}

		}

	}

	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}



void STGrid::joinExhaustedGPUV2(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;

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

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	//bedug: 注意作用域
	//std::vector<STTrajectory> trajSetP, trajSetQ;
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		
		
		// Pbatch
		//bedug: 注意作用域
		std::vector<size_t> tmptaskp;
		std::vector<STTrajectory> trajSetP;
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}

		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			

			

			// Qbatch
			//bedug: 注意作用域
			std::vector<size_t> tmptaskq; 
			std::vector<STTrajectory> trajSetQ;			
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV2

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// only calculate the result here!!
			MyTimer timer2;
			timer2.start();
			std::vector<float> partialResult;
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
			for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
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

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}





void STGrid::joinExhaustedGPUV2p1(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;

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

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	

	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		std::vector<STTrajectory> trajSetP;
		std::vector<size_t> tmptaskp;
		
									 // Pbatch
		for (size_t k = 0; k < (i + GPUOnceCnt > taskSet1.size() ? taskSet1.size() - i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			std::vector<STTrajectory> trajSetQ;
			std::vector<size_t> tmptaskq; // 注意作用域！！
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// get only the result!!
			std::vector<float> partialResult;
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
			for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
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

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}



void STGrid::joinExhaustedGPUV3(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;


	//GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	GetSample2( this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);

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

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	//bedug: 注意作用域
	//std::vector<STTrajectory> trajSetP, trajSetQ;
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;


	
	for (size_t i = 0; i < taskSet1.size(); i += taskSet1.size()) { // ONLY once


		// Pbatch
		//bedug: 注意作用域
		std::vector<size_t> tmptaskp;
		std::vector<STTrajectory> trajSetP;
		for (size_t k = 0; k < (i + taskSet1.size() > taskSet1.size() ? taskSet1.size() - i : taskSet1.size()); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}

		for (size_t j = 0; j < taskSet2.size(); j += taskSet2.size()) { // ONLY once


			// Qbatch
			//bedug: 注意作用域
			std::vector<size_t> tmptaskq;
			std::vector<STTrajectory> trajSetQ;
			for (size_t k = 0; k < (j + taskSet2.size() > taskSet2.size() ? taskSet2.size() - j : taskSet2.size()); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV3 耦合度太高

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// only calculate the result here!!
			MyTimer timer2;
			timer2.start();
			std::vector<float> partialResult;
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
			for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
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

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}




void STGrid::joinExhaustedGPUV4(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
) {
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;

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

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	//bedug: 注意作用域
	//std::vector<STTrajectory> trajSetP, trajSetQ;
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;



	for (size_t i = 0; i < taskSet1.size(); i += taskSet1.size()) { // ONLY once


																	// Pbatch
																	//bedug: 注意作用域
		std::vector<size_t> tmptaskp;
		std::vector<STTrajectory> trajSetP;
		for (size_t k = 0; k < (i + taskSet1.size() > taskSet1.size() ? taskSet1.size() - i : taskSet1.size()); k++) {
			trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
			tmptaskp.push_back(taskSet1[i + k]);
		}

		for (size_t j = 0; j < taskSet2.size(); j += taskSet2.size()) { // ONLY once


																		// Qbatch
																		//bedug: 注意作用域
			std::vector<size_t> tmptaskq;
			std::vector<STTrajectory> trajSetQ;
			for (size_t k = 0; k < (j + taskSet2.size() > taskSet2.size() ? taskSet2.size() - j : taskSet2.size()); k++) {
				//debug: a tiny error
				//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
				tmptaskq.push_back(taskSet2[j + k]);
			}

			//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV3

			// get trajpair(taskpair)
			std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
			GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

			// P Q batch-join
			// only calculate the result here!!
			MyTimer timer2;
			timer2.start();
			std::vector<float> partialResult;
			STSimilarityJoinCalcGPUV4(trajSetP, trajSetQ, partialResult); // have big overload?
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
			for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
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

	std::cout << "finalresult size: " << resultpair.size() << std::endl;

}


void STGrid::joinExhaustedGPU_Final(
	//float epsilon,
	//float alpha,
	int sizeP,
	int sizeQ,
	//std::map<trajPair, float>& result
	std::vector<trajPair>& resultpair,
	std::vector<float>& resultvalue,
	//std::vector<STTrajectory> &P,
	//std::vector<STTrajectory> &Q
	int type,
	int sampletype
) {
	
	// only one TrajDB - selfjoin
	// get ID only one trajDB
	MyTimer timer;
	timer.start();

	std::vector<size_t> taskSet1, taskSet2;

	if (sampletype == 0) {
		GetSample(taskSet1, taskSet2, sizeP, sizeQ);
	}
	else
		if (sampletype == 1) {
			GetSample2(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);

		}
		else
			if (sampletype == 2) {
				GetSample_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
			}
			else
				if (sampletype == 3) {
					GetSample_Filtering_Sorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
				}
				else
					if (sampletype == 4) {
						GetSample_Filtering_NoSorting(this->dataPtr, taskSet1, taskSet2, sizeP, sizeQ);
					}
				else
				{
					printf("Invalid Sample Strategy!\n");
					assert(0);
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

	std::cout << "totaltaskGPU size: " << taskSet1.size()*taskSet2.size() << std::endl;

	//bedug: 注意作用域
	//std::vector<STTrajectory> trajSetP, trajSetQ;
	//std::vector<trajPair> resultpair;
	//std::vector<float> resultvalue;

	bool ifduelgpu = 1;
	if (!ifduelgpu) {
		for (size_t i = 0; i < taskSet1.size(); i += taskSet1.size()) { // ONLY once


																		// Pbatch
																		//bedug: 注意作用域
			std::vector<size_t> tmptaskp;
			std::vector<STTrajectory> trajSetP;
			for (size_t k = 0; k < (i + taskSet1.size() > taskSet1.size() ? taskSet1.size() - i : taskSet1.size()); k++) {
				trajSetP.push_back(this->dataPtr[taskSet1[i + k]]);
				tmptaskp.push_back(taskSet1[i + k]);
			}

			for (size_t j = 0; j < taskSet2.size(); j += taskSet2.size()) { // ONLY once


																			// Qbatch
																			//bedug: 注意作用域
				std::vector<size_t> tmptaskq;
				std::vector<STTrajectory> trajSetQ;
				for (size_t k = 0; k < (j + taskSet2.size() > taskSet2.size() ? taskSet2.size() - j : taskSet2.size()); k++) {
					//debug: a tiny error
					//trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
					trajSetQ.push_back(this->dataPtr[taskSet2[j + k]]);
					tmptaskq.push_back(taskSet2[j + k]);
				}

				//ATTENTION: BELOW 2 operation must match!: GetTaskPair and STSimilarityJoinCalcGPUV3 耦合度太高 不是与STSimilarityJoinCalcGPUV3太高 而是与前面的tmptaskp tmptaskq 以及后面的得到partialResult要一致

				// get trajpair(taskpair)
				std::vector<trajPair> tmptaskGPU; // 是否会太大？？？ 不会：max_size=2305843009213693951
				GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);

				// P Q batch-join
				// only calculate the result here!!
				MyTimer timer2;
				timer2.start();
				std::vector<float> partialResult;

				if (type == 0) {
					STSimilarityJoinCalcGPU(trajSetP, trajSetQ, partialResult);
				}
				else
					if (type == 1) {
						STSimilarityJoinCalcGPUV2(trajSetP, trajSetQ, partialResult);
					}
					else
						if (type == 2) {
							STSimilarityJoinCalcGPUV3(trajSetP, trajSetQ, partialResult);
						}
						else
							if (type == 3) {

							}
							else
								if (type == 4) {

								}
								else
									if (type == 5) {


									}
									else {
										printf("Invalid Calc Strategy!\n");
										assert(0);
									};

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
				for (std::map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
				//if ( (*it).second > EPSILON ) {
				if (it->second > EPSILON) {
				result.insert(*it);
				}
				}
				*/
			}

		}
	}

	else {


		int devicecnt = 2;


		std::vector<std::vector<size_t>> tmptaskpp(devicecnt);
		std::vector<std::vector<STTrajectory>> trajSetPP(devicecnt);
		for (int di = 0; di < devicecnt; ++di) {
			//std::vector<size_t> tmptaskp;
			//std::vector<STTrajectory> trajSetP;
			int i = 0, j = 0;
			for (size_t k = 0; k < (i + taskSet1.size() > taskSet1.size() ? taskSet1.size() - i : taskSet1.size()); k++) {
				if (k % devicecnt == di) {
					//tmptaskpp.at(di).push_back(this->dataPtr[taskSet1[i + k]]);
					trajSetPP.at(di).push_back(this->dataPtr[taskSet1[i + k]]);
					tmptaskpp.at(di).push_back(taskSet1[i + k]);
				}
			}
		}



		std::vector<std::vector<size_t>> tmptaskqq(devicecnt);
		std::vector<std::vector<STTrajectory>> trajSetQQ(devicecnt);
		for (int di = 0; di < devicecnt; ++di) {
			//std::vector<size_t> tmptaskp;
			//std::vector<STTrajectory> trajSetP;
			int i = 0, j = 0;
			for (size_t k = 0; k < (i + taskSet2.size() > taskSet2.size() ? taskSet2.size() - i : taskSet2.size()); k++) {
				if (k % devicecnt == di) {
					//tmptaskpp.at(di).push_back(this->dataPtr[taskSet1[i + k]]);
					trajSetQQ.at(di).push_back(this->dataPtr[taskSet2[i + k]]);
					tmptaskqq.at(di).push_back(taskSet2[i + k]);
				}
			}
		}



		std::vector<std::vector<trajPair>> tmptaskGPU(devicecnt); // 是否会太大？？？ 不会：max_size=2305843009213693951

		for (int di = 0; di < devicecnt; ++di) {
			//GetTaskPair(tmptaskp, tmptaskq, tmptaskGPU);
			for (size_t i = 0; i < tmptaskpp.at(di).size(); i++) {
				for (size_t j = 0; j < tmptaskqq.at(di).size(); j++) {
					trajPair tmppair = trajPair(tmptaskpp.at(di).at(i), tmptaskqq.at(di).at(j));
					tmptaskGPU.at(di).push_back(tmppair);
				}
			}
		}
		std::vector<std::vector<float>> partialResult(devicecnt);

		MyTimer timer2;
		timer2.start();

		for (int di = 0; di < devicecnt; ++di) {
			STSimilarityJoinCalcGPUV5(trajSetPP.at(di), trajSetQQ.at(di), partialResult.at(di),di);
		}

		timer2.stop();
		printf("GPU time once: %f s\n", timer2.elapse());

		// insert new result
		for (int di = 0; di < devicecnt; ++di) {
			for (size_t k = 0; k < partialResult.at(di).size(); k++) {
				if (partialResult.at(di).at(k) > EPSILON) {
					//result[tmptaskGPU[k]] = partialResult[k];
					resultpair.push_back(tmptaskGPU.at(di)[k]);
					resultvalue.push_back(partialResult.at(di)[k]);
				}
			}
		}

	}



	timer.stop();
	printf("GPU time: %f s\n", timer.elapse());

	std::cout << "finalresult size: " << resultpair.size() << std::endl;


}


// this is 类外定义
inline void STGrid::GetTaskPair(std::vector<size_t> &taskp, std::vector<size_t> &taskq, std::vector<trajPair> &resultpair) {
	for (size_t i = 0; i < taskp.size(); i++) {
		for (size_t j = 0; j < taskq.size(); j++) {
			trajPair tmppair = trajPair(taskp[i],taskq[j]);
			resultpair.push_back(tmppair);
		}
	}
}