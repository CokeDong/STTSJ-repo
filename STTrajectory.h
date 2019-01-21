#pragma once

#include "STPoint.h"
#include "ConstDefine.h"
#include <vector>
//using namespace std;
class STTrajectory {

public:
	int sttraj_id; // global para TrajDB really necessary? = index of TrajDB yes
	int traj_length;

	//// both or choose one ? make a choise : both! good for using because not global is good, isolated satisify!! ������
	std::vector<STPoint> traj_of_stpoint; // �˷ѿռ� ����ʡȥ������ pointDB ����/ָ��
	std::vector<int> traj_of_stpoint_id;  // really good ? optimize? because: STPoint.STPoint_ID get the ID
									 // senquence read from STPointDB std::vector ? good ?



	// functions related with this single trajectory!

	void GettingSTPointOnpointID(std::vector<STPoint> &pointdb); // maybe pointdb actually can be released
	
	// ������Ա���� for safety consideration ����Ҫ�Ƕ��߳̿���
	float CalcTTSTSim (const STTrajectory &stt) const;



protected:


private:



};