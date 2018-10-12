#pragma once

#include "STPoint.h"
#include "ConstDefine.h"

using namespace std;
class STTrajectory {

public:
	int sttraj_id; // global para TrajDB really necessary? = index of TrajDB yes
	int traj_length;

	//// both or choose one ? make a choise : both! good for using because not global is good, isolated satisify!! ������
	vector<STPoint> traj_of_stpoint; // �˷ѿռ� ����ʡȥ������ pointDB ����/ָ��
	vector<int> traj_of_stpoint_id;  // really good ? optimize? because: STPoint.STPoint_ID get the ID
									 // senquence read from STPointDB vector ? good ?



	// functions related with this single trajectory!

	void GettingSTPointOnpointID(vector<STPoint> &pointdb); // maybe pointdb actually can be released
	
	// ������Ա���� for safety consideration ����Ҫ�Ƕ��߳̿���
	double CalcTTSTSim (const STTrajectory &stt) const;



protected:


private:



};