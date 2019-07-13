#include "ConstDefine.h"
#include "preprocess.h"
#include "STPoint.h"
#include "STTrajectory.h"
#include "STInvertedList.h"
#include "STGrid.h"
#include "test.h"
#include "util.h"


// using namespace std; // cuda cusparse�� not recommended


//extern void split2(std::string s, std::string delim, std::vector<std::string>* ret); // for test
//extern void GetKeywords(std::vector<std::string>* keywords, std::vector<std::string>* corpus);


// global really good ?? why global? not global ���ļ�ʹ�� poor coding ? how to improve really have meaning ?
// global is a poor choice!!

//// std::vector push_back ȫ���Ƚ���
//
//std::vector<STPoint> pointDB; // only points in ram
//std::vector<STTrajectory> trajDB; // still need
//
//// for filtering 
//std::vector<STInvertedList> invertedlist;

void CheckSimResult(std::vector<trajPair> paircpu, std::vector<float> valuecpu, std::vector<trajPair> pairgpu, std::vector<float> valuegpu);

int main() {

	using namespace std;

	cout << "hello world" << endl;
	bool first = false;
	Preprocess pp;
	if(first){
		// can be done by python. -> �� python Ԥ����  to do
		// fooolly extraction
		pp.VenuesExtraction("./NY/Venues.txt","./NY/VenuesExtc.txt");
		pp.TipsExtraction("./NY/Tips.txt");
	}

	//test test;
	//test.testfunc();

	// std::vector push_back ȫ���Ƚ��� all into ram!!  & or * ����
	std::vector<STPoint> pointDB; // only points in ram
	std::vector<STTrajectory> trajDB; // still need

	// for filtering 
	std::vector<STInvertedList> invertedlist;

	// in this way only because of Ԥ���� 2�� PointDB not good enough
	// slow time-consuming for large # of points and traj
	// ���ô��� ��ָ�봫�� ����

	// ����֤ �����⣡��but too slow!! -> stringstream v.s. std::string, no big difference!!
	pp.ReadPointDBLL(pointDB, "./NY/VenuesExtc.txt");
	pp.ReadPointDBKeyword(pointDB, "./NY/tfidf.txt");
	pp.ReadTrajDBPointID(trajDB, "./NY/TrajExtc.txt", pointDB);

	pp.ReadTrajDBPoint(trajDB, pointDB);

	//std::map<trajPair, float> result;
	std::vector<trajPair> resultpair;
	std::vector<float> resultvalue;

	// Ƶ������ʹ�ñ�����������Ҫ����make -> figure of scalability
	int SIZE = 16; // this is good or gloabal parameter not #define? maybe more convenient

	STGrid grid;
	grid.init(trajDB); // clever����

	
	// ���߳�
	printf("*************** 1-cpu ***************\n");
	std::vector<trajPair> resultpaircpu;
	std::vector<float> resultvaluecpu;
	// for equality, we have to padding for CPU?? -----> no need!!
	grid.joinExhaustedCPUonethread(SIZE, SIZE, resultpaircpu, resultvaluecpu);
	

	//// ���̰߳汾
	//printf("***** mul-cpu full *****\n");
	//std::vector<trajPair> resultpairmcpu;
	//std::vector<float> resultvaluemcpu;
	//grid.joinExhaustedCPU(SIZE,SIZE, resultpairmcpu, resultvaluemcpu);



	//printf("***** cpu  %d *****\n" , MAX_CPU_THREAD);
	//grid.joinExhaustedCPUconfigurablethread(SIZE, SIZE, resultpaircpu, resultvaluecpu, MAX_CPU_THREAD); // not that accurate!!
	
	

	

	// baseline���㷨 ����
	printf("*************** 1-gpu coarse(baseline) ****************\n");
	std::vector<trajPair> resultpaircoarsegpu;
	std::vector<float> resultvaluecoarsegpu;
	grid.joinExhaustedGPU(SIZE, SIZE, resultpaircoarsegpu, resultvaluecoarsegpu);
	//CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpaircoarsegpu, resultvaluecoarsegpu);
	


	//// this is worse than Zero Copy, but very very tiny
	//printf("***** 1-gpu coarse No Zero Copy*****\n");
	//std::vector<trajPair> resultpaircoarsegpunzc;
	//std::vector<float> resultvaluecoarsegpunzc;
	//grid.joinExhaustedGPUNZC(SIZE, SIZE, resultpaircoarsegpunzc, resultvaluecoarsegpunzc);
	////CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpaircoarsegpu, resultvaluecoarsegpu);
	



	// ����̫���켣 �����Դ治�� -> ����Ԥ���� �Զ����������������һ��ռ�ù����Դ� ��Ҫ̫���� ��д assert
	printf("*************** 1-gpu fine ***************\n");
	std::vector<trajPair> resultpairfinegpu;
	std::vector<float> resultvaluefinegpu;
	grid.joinExhaustedGPUV2(SIZE, SIZE, resultpairfinegpu, resultvaluefinegpu);
	//CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpairfinegpu, resultvaluefinegpu);
	

	//// not that important for performance improvement  -> �ϲ����kernel
	//printf("***** 1-gpu V2p1 fine *****\n");
	//std::vector<trajPair> resultpairfinegpu2;
	//std::vector<float> resultvaluefinegpu2;
	//grid.joinExhaustedGPUV2p1(SIZE, SIZE, resultpairfinegpu2, resultvaluefinegpu2);
	////CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpairfinegpu2, resultvaluefinegpu2);





	//// ����̫���켣 �����Դ治�� -> ����Ԥ���� �õ���cusparse�� �м�����Ҫ�Դ� ��Ҫ̫���� ��д assert
	//printf("*************** 1-gpu V3 fine cusparse nobatch ***************\n");
	//std::vector<trajPair> resultpairfinegpu4;
	//std::vector<float> resultvaluefinegpu4;
	//grid.joinExhaustedGPUV4(SIZE, SIZE, resultpairfinegpu4, resultvaluefinegpu4);
	////CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpairfinegpu4, resultvaluefinegpu4);





	// no-sorting
	// �������ⳤ�켣 ���ﹹ��batch����Ҫ
	printf("*************** 1-gpu V3 fine ***************\n");
	std::vector<trajPair> resultpairfinegpu3;
	std::vector<float> resultvaluefinegpu3;
	grid.joinExhaustedGPUV3(SIZE, SIZE, resultpairfinegpu3, resultvaluefinegpu3);
	//CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpairfinegpu3, resultvaluefinegpu3);
	


	// add no-flip





	// add sorting schedular







	//sleep(10);
	cout << "finished" << endl;
#ifdef DIS_RESULT
	getchar();
	getchar();
	getchar();
#endif
	return 0;
}


void CheckSimResult(std::vector<trajPair> paircpu, std::vector<float> valuecpu, std::vector<trajPair> pairgpu, std::vector<float> valuegpu) {
	
	// fully output: #define EPSILON -1

	for (size_t i = 0; i < paircpu.size(); i++) {
		if (i < pairgpu.size()) {
			printf("%zu\n(%zu,%zu):%.5f\n(%zu,%zu):%.5f\n\n", i, paircpu[i].first, paircpu[i].second, valuecpu[i], pairgpu[i].first, pairgpu[i].second, valuegpu[i]);
		}
		else {
			printf("%zu\n(%zu,%zu):%.5f\n(%zu,%zu):%.5f\n\n", i, paircpu[i].first, paircpu[i].second, valuecpu[i], 0, 0, 0);
		}
	}
	if (paircpu.size() < pairgpu.size()) {
		for (size_t i = paircpu.size(); i < pairgpu.size(); i++) {
			printf("%zu\n(%zu,%zu):%.5f\n(%zu,%zu):%.5f\n\n", i, 0, 0, 0, pairgpu[i].first, pairgpu[i].second, valuegpu[i]);
		}
	}
	

	//int falsecnt = 0;
	//// check pair
	//for (size_t i = 0; i < paircpu.size(); i++) {
	//		if (i < pairgpu.size()) {
	//			if (paircpu[i] != pairgpu[i]) {
	//				printf("False %zu\n(%zu,%zu):%.5f\n(%zu,%zu):%.5f\n\n", i, paircpu[i].first, paircpu[i].second, valuecpu[i], pairgpu[i].first, pairgpu[i].second, valuegpu[i]);
	//				falsecnt++;
	//			}
	//		}
	//	}
	//if (!falsecnt) printf("True!\n");

}