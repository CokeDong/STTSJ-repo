#pragma once

#include "ConstDefine.h"

//#include <string>



//using namespace std;



class STPoint {

public:

	//STPoint();
	//~STPoint();
	float lat;
	float lon;
	
	// done by Cpp
	int zorder; // *** not INIT 仅仅在filter使用

	// seems better
	int stpoint_id; // this is ID , only identity of point = index of stpointDB
	
	//std::vector<std::string> keyeords; 
	//std::vector<int> keywords_id;
	std::vector<Keywordtuple> keywords; // int: ID of keywords: value of TF-IDF ,may have more than one keywords
										// turple makes things hard a little bit, we use Keywordtuple struct

	std::vector<int> belongtraj; // int: ID of traj having this point, may have more than one trajs
						
	// if help GPU? may be easy to use on CPU



	// functions related with this single point!
	
	//减少函数调用栈使用
	//float CalcPPSSim(const STPoint &p);
	//float CalcPPTSim(const STPoint &p);

	float CalcPPSTSim(const STPoint &p) const;


protected:


private:




};