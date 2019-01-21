#pragma once


#include "STPoint.h"
#include "STTrajectory.h"
#include "ConstDefine.h"

//using namespace std;

class Preprocess {

public:
	std::ifstream fin;
	std::ofstream fout;

	// √ª…∂”√
	std::map<std::string, int> trajdict;		// trajname trajid
	std::map<std::string, int> venuesdict;	// venuesname venuesid

	// std::map<std::string, int> keyworddict;	// keyword	keyewordid in DICTINARY

	// funcs need for read from outside



	// python is as well actually
	// funcs need for write to outside
	void VenuesExtraction(std::string fileName, std::string outFileName);
	void TipsExtraction(std::string fileName);


	// for read into main parameter
	void ReadPointDBLL(std::vector<STPoint> &pointdb, std::string fileName);
	void ReadPointDBKeyword(std::vector<STPoint> &pointdb, std::string fileName);
	void ReadTrajDBPointID(std::vector<STTrajectory> &trajdb, std::string fileName,std::vector<STPoint> &pointdb);
	void ReadTrajDBPoint(std::vector<STTrajectory> &trajdb, std::vector<STPoint> &pointdb);

	void ReadPointDBLLV2(std::vector<STPoint> &pointdb, std::string fileName);
	// next 3: not modified yet!
	void ReadPointDBKeywordV2(std::vector<STPoint> &pointdb, std::string fileName);
	void ReadTrajDBPointIDV2(std::vector<STTrajectory> &trajdb, std::string fileName, std::vector<STPoint> &pointdb);
	void ReadTrajDBPointV2(std::vector<STTrajectory> &trajdb, std::vector<STPoint> &pointdb);


	//void getInvertedIndex();



protected:

private:


};