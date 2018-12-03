#include "ConstDefine.h"
#include "preprocess.h"
#include "STPoint.h"
#include "STTrajectory.h"
#include "STInvertedList.h"
#include "STGrid.h"
#include "test.h"
#include "util.h"

using namespace std;

//extern void split2(string s, string delim, vector<string>* ret); // for test
//extern void GetKeywords(vector<string>* keywords, vector<string>* corpus);


// global really good ?? why global? not global 跨文件使用 poor coding ? how to improve really have meaning ?
// global is a poor choice!!

//// vector push_back 全部比较慢
//
//vector<STPoint> pointDB; // only points in ram
//vector<STTrajectory> trajDB; // still need
//
//// for filtering 
//vector<STInvertedList> invertedlist;

void CheckSimResult(vector<trajPair> paircpu, vector<float> valuecpu, vector<trajPair> pairgpu, vector<float> valuegpu);

int main() {

	cout << "hello world" << endl;
	bool first = false;
	Preprocess pp;
	if(first){
		// can be done by python.
		// fooolly extraction
		pp.VenuesExtraction("./NY/Venues.txt","./NY/VenuesExtc.txt");
		pp.TipsExtraction("./NY/Tips.txt");
	}

	//test test;
	//test.testfunc();

	// vector push_back 全部比较慢 all into ram!!  & or * 传递
	vector<STPoint> pointDB; // only points in ram
	vector<STTrajectory> trajDB; // still need

	// for filtering 
	vector<STInvertedList> invertedlist;

	// in this way only because of 预处理 2次 PointDB not good enough
	// slow time-consuming for large # of points and traj
	// 引用传递 比指针传递 更简单

	// 已验证 无问题！！
	pp.ReadPointDBLL(pointDB, "./NY/VenuesExtc.txt");
	pp.ReadPointDBKeyword(pointDB, "./NY/tfidf.txt");
	pp.ReadTrajDBPointID(trajDB, "./NY/TrajExtc.txt", pointDB);

	pp.ReadTrajDBPoint(trajDB, pointDB);

	//map<trajPair, float> result;
	vector<trajPair> resultpair;
	vector<float> resultvalue;

	// 频繁调参使用变量！！
	int SIZE = SIZE_DATA;

	STGrid grid;
	grid.init(trajDB); // clever！！

	//vector<trajPair> resultpaircpu;
	//vector<float> resultvaluecpu;
	//vector<trajPair> resultpairgpu;
	//vector<float> resultvaluegpu;

	// printf("***** 1-cpu *****\n");
	//grid.joinExhaustedCPUonethread(SIZE, SIZE, resultpaircpu, resultvaluecpu);

	printf("***** mul-cpu *****\n");
	vector<trajPair> resultpairmcpu;
	vector<float> resultvaluemcpu;
	grid.joinExhaustedCPU(SIZE,SIZE, resultpairmcpu, resultvaluemcpu);
	
	//grid.joinExhaustedCPUconfigurablethread(SIZE, SIZE, resultpaircpu, resultvaluecpu, MAX_CPU_THREAD);
	
	printf("***** 1-gpu coarse *****\n");
	vector<trajPair> resultpaircoarsegpu;
	vector<float> resultvaluecoarsegpu;
	grid.joinExhaustedGPU(SIZE, SIZE, resultpaircoarsegpu, resultvaluecoarsegpu);
	CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpaircoarsegpu, resultvaluecoarsegpu);

	printf("***** 1-gpu fine *****\n");
	vector<trajPair> resultpairfinegpu;
	vector<float> resultvaluefinegpu;
	grid.joinExhaustedGPUV2(SIZE, SIZE, resultpairfinegpu, resultvaluefinegpu);
	CheckSimResult(resultpairmcpu, resultvaluemcpu, resultpairfinegpu, resultvaluefinegpu);
	

	//sleep(10);

	cout << "finished" << endl;
#ifdef DIS_RESULT
	getchar();
	getchar();
	getchar();
#endif
	return 0;
}


void CheckSimResult(vector<trajPair> paircpu, vector<float> valuecpu, vector<trajPair> pairgpu, vector<float> valuegpu) {
	
	// fully output

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