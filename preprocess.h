#pragma once


#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"

using namespace std;

class Preprocess {

public:
	ifstream fin;
	ofstream fout;

	// √ª…∂”√
	map<string, int> trajdict;		// trajname trajid
	map<string, int> venuesdict;	// venuesname venuesid

	// map<string, int> keyworddict;	// keyword	keyewordid in DICTINARY

	// funcs need for read from outside



	// python is as well actually
	// funcs need for write to outside
	void VenuesExtraction(string fileName, string outFileName);
	void TipsExtraction(string fileName);


	// for read into main parameter
	void ReadPointDBLL(vector<STPoint> &pointdb, string fileName);
	void ReadPointDBKeyword(vector<STPoint> &pointdb, string fileName);
	void ReadTrajDBPointID(vector<STTrajectory> &trajdb, string fileName,vector<STPoint> &pointdb);
	void ReadTrajDBPoint(vector<STTrajectory> &trajdb, vector<STPoint> &pointdb);

	


	//void getInvertedIndex();



protected:

private:


};