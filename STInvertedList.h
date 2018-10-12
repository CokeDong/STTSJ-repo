#pragma once
#include <vector>
#include "STPoint.h"
#include "ConstDefine.h"
#include <map>
//#include <tuple>

using namespace std;

class STInvertedList {

public:
	int keyword_id; // about this keyword_id inverted_list
	double minvalue, mxvalue;

	// which one to use ??  make a choise or both 
	vector<int> pointid_only;
	vector<Pointtuple> pointid_value; // int is ID of stpoint, double is value of this keyword in this point
									// turple is right here ! similar to python here. ? no difference ? can python process this task? i think it can !!
									// map is wrong !! 
									//map<int, double> keword_value; // map is container no need for a vector, int is for order!! not right tuple is good


protected:


private:




};