#include "STGrid.h"




void STGrid::init(const vector<STTrajectory> &dptr) {

	dataPtr = dptr; // �������ô���

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
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // �����Ƿ����� ���̣߳����Ƿ�������Ӱ�� 
			totaltaskCPU.push_back(tmppair); // ���ﲻ����GPU�ڴ濽������
											 //}
		}
	}

	cout << "totaltaskCPU size: " << totaltaskCPU.size() << endl;


	// ���߳�ͬʱ���ǿ��Ե�

	// ���߳�д ����tmpresult
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
				trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // �����Ƿ����� ���̣߳����Ƿ�������Ӱ�� 
				totaltaskCPU.push_back(tmppair); // ���ﲻ����GPU�ڴ濽������
			//}
		}
	}

	cout << "totaltaskCPU size: "<<totaltaskCPU.size() << endl;
	

	// ���߳�ͬʱ���ǿ��Ե�
	
	// ���߳�д ����tmpresult
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
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // �����Ƿ����� ���̣߳����Ƿ�������Ӱ�� 
			totaltaskCPU.push_back(tmppair); // ���ﲻ����GPU�ڴ濽������
											 //}
		}
	}

	cout << "totaltaskCPU size: " << totaltaskCPU.size() << endl;


	// ���߳�ͬʱ���ǿ��Ե�

	// ���߳�д ����tmpresult
	
	// �����޷����أ���
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
		//thread_STSim.clear();//no need �����򼴿�
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


// Question1��const��Ҫô�� ò��Ӧ������ ������� ��ȫ
// Question2��const���ʺ϶��߳�ô ���ô��ݲ����õ����̣߳� ò����
void STGrid::STSimilarityJoinCalcCPUV2(
	const STTrajectory &T1,
	const STTrajectory &T2,
	float &result // ������ֵ���� 
){
	result = T1.CalcTTSTSim(T2);
	// aborted
}

// ָ�봫��
// ���߳�
void STGrid::STSimilarityJoinCalcCPUV3(
	const STTrajectory *T1,
	const STTrajectory *T2,
	float *result // ������ֵ���� 
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
			trajPair tmppair = trajPair(taskSet1[i], taskSet2[j]); // �����Ƿ����� ���̣߳����Ƿ�������Ӱ�� 
			totaltaskGPU.push_back(tmppair); // ���ﲻ����GPU�ڴ濽������
											 //}
		}
	}

	cout << "totaltaskGPU size: " << totaltaskGPU.size() << endl;
	*/
	vector<STTrajectory> trajSetP, trajSetQ;
	
	for (size_t i = 0; i < taskSet1.size(); i += GPUOnceCnt) {
		// Pbatch
		for (size_t k = 0; k < ( i + GPUOnceCnt > taskSet1.size() ? taskSet1.size()-i : GPUOnceCnt); k++) {
			trajSetP.push_back(this->dataPtr[i + k]);
		}
		for (size_t j = 0; j < taskSet2.size(); j += GPUOnceCnt) {
			// Qbatch
			for (size_t k = 0; k < (j + GPUOnceCnt > taskSet2.size() ? taskSet2.size() - j : GPUOnceCnt); k++) {
				trajSetQ.push_back(this->dataPtr[j + k]);
			}
			// P Q batch-join
			map<trajPair, float> partialResult;
			
			STSimilarityJoinCalcGPU(trajSetP, trajSetQ, result);
			
			// insert new result
			for (map<trajPair, float>::iterator it = partialResult.begin(); it != partialResult.end(); it++) {
				//if ( (*it).second > EPSILON ) {
				if (it->second > EPSILON) {
					result.insert(*it);
				}		
			}
		}
	
	}
	
	cout << "finalresult size: " << result.size() << endl;

}