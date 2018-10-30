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


// global really good ?? why global? not global ���ļ�ʹ�� poor coding ? how to improve really have meaning ?

// global is a poor choice!!

//// vector push_back ȫ���Ƚ���
//
//vector<STPoint> pointDB; // only points in ram
//vector<STTrajectory> trajDB; // still need
//
//// for filtering 
//vector<STInvertedList> invertedlist;

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

	// vector push_back ȫ���Ƚ��� all into ram!!  & or * ����
	vector<STPoint> pointDB; // only points in ram
	vector<STTrajectory> trajDB; // still need

	// for filtering 
	vector<STInvertedList> invertedlist;

	// in this way only because of Ԥ���� 2�� PointDB not good enough
	// slow time-consuming for large # of points and traj
	// ���ô��� ��ָ�봫�� ����

	pp.ReadPointDBLL(pointDB, "./NY/VenuesExtc.txt");
	pp.ReadPointDBKeyword(pointDB, "./NY/tfidf.txt");
	pp.ReadTrajDBPointID(trajDB, "./NY/TrajExtc.txt", pointDB);
	pp.ReadTrajDBPoint(trajDB, pointDB);

	map<trajPair, float> result;

	STGrid grid;
	grid.init(trajDB); // clever����
	grid.joinExhaustedCPUonethread(32, 32, result);
	//grid.joinExhaustedCPU(128,128,result);
	
	
	//grid.joinExhaustedCPUconfigurablethread(128, 128, result, MAX_CPU_THREAD);
	
	grid.joinExhaustedGPU(128, 128, result);


	//sleep(10);

	cout << "finished" << endl;
	getchar();
	getchar();
	return 0;
}