#include "preprocess.h"
#include "util.h"
#include<sstream>


//extern void split(std::string s, std::string delim, std::vector<std::string>* ret);

using namespace std;

/*
����˵����
�㣺 206416����checkin������ and 206091�������ۣ�
�켣��49027(tips���) and 49001��ȥ���չ켣 �����κε㣩 ƽ��4-5����ÿ���켣 �����ظ��㲢����
*/


// this EXTRC is foooooooooooool ֱ����python�϶�ȡһ�μ��ɣ��� 
void Preprocess::VenuesExtraction(std::string fileName,std::string outFileName) {
	fin.open(fileName);
	fout.open(outFileName);
	if ( ! fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}
	std::string linestr;
	//int linecnt = 1; // ����
	int linecnt = 0; // ����
	while  (getline(fin,linestr))
	{
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		//for (std::vector<std::string>::iterator it = ssplit.begin(); it != ssplit.end(); it++) {					
		//}
		std::string vid = *ssplit.begin();
		std::string name = vid.substr(1, vid.length() - 2);
		venuesdict[name] = linecnt++;
		std::string vlati = *(ssplit.begin() + 2);
		std::string vlong = *(ssplit.begin() + 3);
		fout << name <<'\t'<< venuesdict[name] <<'\t'<<vlati<<'\t'<<vlong<<endl;
		//fout << venuesdict[name] << '\t' << vlati << '\t' << vlong << endl;
	}
	fin.close();
	fout.close();
}


void Preprocess::TipsExtraction(std::string fileName) {
	
	
	//fin.open(fileName);
	//fout.open("./NY/TipsExtc.txt");
	//if (!fin.is_open())
	//{
	//	cout << "Error opening file";
	//	exit(1);
	//}
	//std::string linestr;
	//while (getline(fin, linestr))
	//{
	//	std::string s = linestr;
	//	std::vector<std::string> ssplit;
	//	std::string dot = "\t";
	//	split(s, dot, &ssplit);
	//	//for (std::vector<std::string>::iterator it = ssplit.begin(); it != ssplit.end(); it++) {					
	//	//}
	//	fout << ssplit.at(0) << '\t';
	//	for (int i = 1; i < ssplit.size(); i++) {
	//		std::string vid = ssplit.at(i);
	//		std::string name = vid.substr(1, vid.length() - 2);
	//		fout << name << '\t';
	//		i += 2;
	//		std::string tip = ssplit.at(i);
	//		fout << tip.substr(1, tip.length() - 2) << '\t';
	//		i += 4;
	//		int num = atoi(ssplit.at(i).c_str());
	//		i += num;
	//	}
	//	fout << endl;
	//}
	//fin.close();
	//fout.close();



	
	// only read once, ��֤ÿ���켣�еĵ�ض������� keyword !! 
	fin.open(fileName);
	fout.open("./NY/TrajExtc.txt");
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}

	std::map<std::string, std::vector<std::string>> venuescorpus; // venuesname alltips

	std::string linestr;
	int linecnt = 0;
	while (getline(fin, linestr))
	{
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		//for (std::vector<std::string>::iterator it = ssplit.begin(); it != ssplit.end(); it++) {					
		//}
		//fout << ssplit.at(0) << '\t';

		if (ssplit.size() == 1) continue; // �쳣���� ����userID �켣����Ϊ0 �������ù켣������TrajExtc

		trajdict[ssplit.at(0)] = linecnt++;
		fout << ssplit.at(0) << '\t';
		fout << trajdict[ssplit.at(0)] << '\t';
		for (int i = 1; i < ssplit.size(); i++) {
			std::string vid = ssplit.at(i);
			std::string name = vid.substr(1, vid.length() - 2);
			fout << venuesdict[name] << '\t';
			i += 2;
			if (ssplit.at(i).compare("0") == 0) i += 1; // ��β���� some �쳣
			std::string tip = ssplit.at(i);
			//fout << tip.substr(1, tip.length() - 2) << '\t';
			venuescorpus[name].push_back(tip.substr(1, tip.length() - 2)); // save results to parameter before write into file
			i += 4;
			int num = atoi(ssplit.at(i).c_str());
			i += num;
		}
		fout << endl;
	}
	fin.close();
	fout.close();



	// open twice FOOOOOOOOOOOOOL, float time consuming!!! while-loop, solve by saving results to parameter before write into file
	fin.open(fileName);
	fout.open("./NY/CorpusExtc.txt");
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}

	//std::map<std::string, std::vector<std::string>> venuescorpus; // venuesname alltips
	////std::map<std::string, std::vector<std::string>> venueskeywords;// venuesname venueskeyword

	//while (getline(fin, linestr))
	//{
	//	std::string s = linestr;
	//	std::vector<std::string> ssplit;
	//	std::string dot = "\t";
	//	split(s, dot, &ssplit);
	//	//for (std::vector<std::string>::iterator it = ssplit.begin(); it != ssplit.end(); it++) {					
	//	//}
	//	//fout << ssplit.at(0) << '\t';
	//	for (int i = 1; i < ssplit.size(); i++) {
	//		std::string vid = ssplit.at(i);
	//		std::string name = vid.substr(1, vid.length() - 2);
	//		//fout << name << '\t';
	//		i += 2;
	//		if (ssplit.at(i).compare("0") == 0) i += 1;
	//		std::string tip = ssplit.at(i);
	//		//fout << tip.substr(1, tip.length() - 2) << '\n';
	//		venuescorpus[name].push_back(tip.substr(1, tip.length() - 2));
	//		i += 4;
	//		int num = atoi(ssplit.at(i).c_str());
	//		i += num;
	//	}
	//	//fout << endl;
	//}

	// save corpus
	for (std::map<std::string, std::vector<std::string>>::iterator it = venuescorpus.begin(); it != venuescorpus.end(); it++) {
		fout << it->first << '\t' << venuesdict[it->first] << '\t'<< it->second.size()<<'\t';
		for (std::vector<std::string>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
			fout << *it2 << '\t';
		}
		fout << endl;
	}

	//GetTF(&venueskeywords[it->first], &it->second);

	fin.close();
	fout.close();

}



void Preprocess::ReadPointDBLL(std::vector<STPoint> &pointdb, std::string fileName) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}
	std::string linestr;
	float latmax=-180, lonmax=-360, latmin=180, lonmin=360;
	while (getline(fin,linestr)) {
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		STPoint stptmp;
		stptmp.stpoint_id = atoi(ssplit.at(1).c_str()); // venuesdict is okay 2
		stptmp.lat = atof(ssplit.at(2).c_str());
		stptmp.lon = atof(ssplit.at(3).c_str());
		if (stptmp.lat > latmax) latmax = stptmp.lat;
		if (stptmp.lon > lonmax) lonmax = stptmp.lon;
		if (stptmp.lat < latmin) latmin = stptmp.lat;
		if (stptmp.lon < lonmin) latmax = stptmp.lon;
		pointdb.push_back(stptmp); // ȷ��OK
	}
	
	cout << "maxdistance = " << calculateDistance(latmax, lonmax, latmin, lonmin) << endl;
	cout << "PointDBLL reading finished" << endl;
	fin.close();
}


void Preprocess::ReadPointDBKeyword(std::vector<STPoint> &pointdb, std::string fileName) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}

	std::string linestr;
	int maxkeywordcnt = -1;
	while (getline(fin, linestr)) {
		//cout << "gettting here";
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		int indexofpointdb = atoi(ssplit.at(0).c_str());
		//for (int i = 1; i < ssplit.size() - 1; i++) { // i < ssplit.size() - 1 
		//	Keywordtuple ktuple;
		//	ktuple.keywordid = atoi(ssplit.at(i).c_str());
		//	ktuple.keywordvalue = atof(ssplit.at(i + 1).c_str()); // ע���±�Խ��
		//	pointdb.at(indexofpointdb).keywords.push_back(ktuple);
		//	i++;
		//}
		//cout << ssplit.size() << endl;

		// debug: �߼����� ��λ \t\n  -1 \n ռһλ���Լ�һ ����std::vectorԽ��
		for (int i = 1; i < ssplit.size() - 1; i+=2) { //  GAP = 2 ������ BUG here: -1 \n ռһλ���Լ�һ ����std::vectorԽ��
			Keywordtuple ktuple;
			ktuple.keywordid = atoi(ssplit.at(i).c_str());
			ktuple.keywordvalue = atof(ssplit.at(i+1).c_str()); // ע���±�Խ��
			pointdb.at(indexofpointdb).keywords.push_back(ktuple);
			//i++;
		}
		// std::vector.size() �ķ���ֵ������size_type(int) !!
		if (maxkeywordcnt < (int)pointdb.at(indexofpointdb).keywords.size()) {
			maxkeywordcnt = (int)pointdb.at(indexofpointdb).keywords.size();
		}
		//cout << pointdb.at(indexofpointdb).keywords.size()<<endl; // why??  because forgetting fin.close!!!�����޷�������ȡ
	}
	cout << "max keyword cnt= " << maxkeywordcnt << endl;
	cout << "PointDBkeyword reading finished" << endl;
	fin.close();
}


void Preprocess::ReadTrajDBPointID(std::vector<STTrajectory> &trajdb, std::string fileName,std::vector<STPoint> &pointdb) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}
	std::string linestr;
	int linecnt = 0;
	int maxlen = -1;
	while (getline(fin, linestr)) {

		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		STTrajectory trajtmp;
		int trajid = atoi(ssplit.at(1).c_str());
		trajtmp.sttraj_id = trajid;// trajdict is ok 2
		//int indexofpointdb = atoi(ssplit.at(0).c_str());
		 
		// debug: �߼����� split�Զ��庯�����ǲ���ȫ ĩβ�� dot ��Ҫ -1 ��β�� \t\n!!!
		for (int i = 2; i < ssplit.size()-1; i++) { // BUG here: -1 \n ռһλ���Լ�һ ����std::vectorԽ��
			int pointID = atoi(ssplit.at(i).c_str());
			trajtmp.traj_of_stpoint_id.push_back(pointID);
			
			// here ���� STPoint::belongtraj
			pointdb.at(pointID).belongtraj.push_back(trajid);
		}
		trajtmp.traj_length = trajtmp.traj_of_stpoint_id.size();
		if (maxlen < trajtmp.traj_length) {
			maxlen = trajtmp.traj_length;
		}
		
		trajdb.push_back(trajtmp);
	}
	cout << "maxtrajetory len= " << maxlen << endl;
	cout << "maxtrajetory size= " << trajdb.size() << endl;
	cout << "TrajDBpointsID reading finished" << endl;
	fin.close();
}

void Preprocess::ReadTrajDBPoint(std::vector<STTrajectory> &trajdb, std::vector<STPoint> &pointdb) {

	for (std::vector<STTrajectory>::iterator it = trajdb.begin(); it != trajdb.end(); it++) {
		(*it).GettingSTPointOnpointID(pointdb);
	}
	cout << "TrajDBPoints reading finished" << endl;

}












void Preprocess::ReadPointDBLLV2(std::vector<STPoint> &pointdb, std::string fileName) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}

	std::string buffer;
	buffer.assign(istreambuf_iterator<char>(fin), istreambuf_iterator<char>());
	stringstream bufferstream;
	bufferstream.str(buffer);

	// cannot see any improvement!
	std::string linestr;

	float latmax = -180, lonmax = -360, latmin = 180, lonmin = 360;
	while (getline(bufferstream, linestr)) {
	//while (getline(fin, linestr)) {
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		STPoint stptmp;
		stptmp.stpoint_id = atoi(ssplit.at(1).c_str()); // venuesdict is okay 2
		stptmp.lat = atof(ssplit.at(2).c_str());
		stptmp.lon = atof(ssplit.at(3).c_str());
		if (stptmp.lat > latmax) latmax = stptmp.lat;
		if (stptmp.lon > lonmax) lonmax = stptmp.lon;
		if (stptmp.lat < latmin) latmin = stptmp.lat;
		if (stptmp.lon < lonmin) latmax = stptmp.lon;
		pointdb.push_back(stptmp); // ȷ��OK
	}

	cout << "maxdistance = " << calculateDistance(latmax, lonmax, latmin, lonmin) << endl;
	cout << "PointDBLL reading finished" << endl;
	fin.close();
}


void Preprocess::ReadPointDBKeywordV2(std::vector<STPoint> &pointdb, std::string fileName) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}

	std::string linestr;
	int maxkeywordcnt = -1;
	while (getline(fin, linestr)) {
		//cout << "gettting here";
		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		int indexofpointdb = atoi(ssplit.at(0).c_str());
		//for (int i = 1; i < ssplit.size() - 1; i++) { // i < ssplit.size() - 1 
		//	Keywordtuple ktuple;
		//	ktuple.keywordid = atoi(ssplit.at(i).c_str());
		//	ktuple.keywordvalue = atof(ssplit.at(i + 1).c_str()); // ע���±�Խ��
		//	pointdb.at(indexofpointdb).keywords.push_back(ktuple);
		//	i++;
		//}
		//cout << ssplit.size() << endl;

		// debug: �߼����� ��λ \t\n  -1 \n ռһλ���Լ�һ ����std::vectorԽ��
		for (int i = 1; i < ssplit.size() - 1; i += 2) { //  GAP = 2 ������ BUG here: -1 \n ռһλ���Լ�һ ����std::vectorԽ��
			Keywordtuple ktuple;
			ktuple.keywordid = atoi(ssplit.at(i).c_str());
			ktuple.keywordvalue = atof(ssplit.at(i + 1).c_str()); // ע���±�Խ��
			pointdb.at(indexofpointdb).keywords.push_back(ktuple);
			//i++;
		}
		// std::vector.size() �ķ���ֵ������size_type(int) !!
		if (maxkeywordcnt < (int)pointdb.at(indexofpointdb).keywords.size()) {
			maxkeywordcnt = (int)pointdb.at(indexofpointdb).keywords.size();
		}
		//cout << pointdb.at(indexofpointdb).keywords.size()<<endl; // why??  because forgetting fin.close!!!�����޷�������ȡ
	}
	cout << "max keyword cnt= " << maxkeywordcnt << endl;
	cout << "PointDBkeyword reading finished" << endl;
	fin.close();
}


void Preprocess::ReadTrajDBPointIDV2(std::vector<STTrajectory> &trajdb, std::string fileName, std::vector<STPoint> &pointdb) {

	fin.open(fileName);
	if (!fin.is_open())
	{
		cout << "Error opening file";
		exit(1);
	}
	std::string linestr;
	int linecnt = 0;
	int maxlen = -1;
	while (getline(fin, linestr)) {

		std::string s = linestr;
		std::vector<std::string> ssplit;
		std::string dot = "\t";
		split(s, dot, &ssplit);
		STTrajectory trajtmp;
		int trajid = atoi(ssplit.at(1).c_str());
		trajtmp.sttraj_id = trajid;// trajdict is ok 2
								   //int indexofpointdb = atoi(ssplit.at(0).c_str());

								   // debug: �߼����� split�Զ��庯�����ǲ���ȫ ĩβ�� dot ��Ҫ -1 ��β�� \t\n!!!
		for (int i = 2; i < ssplit.size() - 1; i++) { // BUG here: -1 \n ռһλ���Լ�һ ����std::vectorԽ��
			int pointID = atoi(ssplit.at(i).c_str());
			trajtmp.traj_of_stpoint_id.push_back(pointID);

			// here ���� STPoint::belongtraj
			pointdb.at(pointID).belongtraj.push_back(trajid);
		}
		trajtmp.traj_length = trajtmp.traj_of_stpoint_id.size();
		if (maxlen < trajtmp.traj_length) {
			maxlen = trajtmp.traj_length;
		}

		trajdb.push_back(trajtmp);
	}
	cout << "maxtrajetory len= " << maxlen << endl;
	cout << "TrajDBpointsID reading finished" << endl;
	fin.close();
}

void Preprocess::ReadTrajDBPointV2(std::vector<STTrajectory> &trajdb, std::vector<STPoint> &pointdb) {

	for (std::vector<STTrajectory>::iterator it = trajdb.begin(); it != trajdb.end(); it++) {
		(*it).GettingSTPointOnpointID(pointdb);
	}
	cout << "TrajDBPoints reading finished" << endl;

}