#pragma once
#include <vector>
#include "STPoint.h"
//#include <string>
#include <map>

using namespace std;

class STzorderlist {

public:
	// actually only need one list for search shared? just pointer is okay
	
	vector<vector<int>> pointdbzorder_increase; // int is the id of STpoint index, is zorder 2-d vector because one cell may have more than one STpoint
												// index ? zorder is not good not even
	// need functions to get above vector, make judgement and so on
	// or we can process in python to read from txt



protected:


private:







};