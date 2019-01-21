#pragma once
#include <vector>
#include "STPoint.h"
#include "ConstDefine.h"
#include <map>
//#include <tuple>

//using namespace std;

class STInvertedList {

public:
	int keyword_id; // about this keyword_id inverted_list
	float minvalue, mxvalue;

	// which one to use ??  make a choise or both 
	std::vector<int> pointid_only;
	std::vector<Pointtuple> pointid_value; // int is ID of stpoint, float is value of this keyword in this point
									// turple is right here ! similar to python here. ? no difference ? can python process this task? i think it can !!
									// std::map is wrong !! 
									//std::map<int, float> keword_value; // std::map is container no need for a std::vector, int is for order!! not right tuple is good


protected:


private:




};