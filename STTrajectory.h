#pragma once

#include "STPoint.h"
#include "ConstDefine.h"

using namespace std;
class STTrajectory {

public:
	int sttraj_id; // global para TrajDB really necessary? = index of TrajDB yes
	int traj_length;

	//// both or choose one ? make a choise : both! good for using because not global is good, isolated satisify!! 隔离性
	vector<STPoint> traj_of_stpoint; // 浪费空间 但是省去了引入 pointDB 引用/指针
	vector<int> traj_of_stpoint_id;  // really good ? optimize? because: STPoint.STPoint_ID get the ID
									 // senquence read from STPointDB vector ? good ?



	// functions related with this single trajectory!

	void GettingSTPointOnpointID(vector<STPoint> &pointdb); // maybe pointdb actually can be released
	
	// 常量成员函数 for safety consideration 更重要是多线程考虑
	double CalcTTSTSim (const STTrajectory &stt) const;



protected:


private:



};