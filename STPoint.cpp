#include "STPoint.h"

//extern float calculateDistance(float LatA, float LonA, float LatB, float LonB); //编译不通过 诡异

// 数据成员初始化是在进入构造函数之前完成的
//STPoint::STPoint() {
//	lat = 0;
//	lon = 0;
//	stpoint_id = 0;
//
//	vector<Keywordtuple> kwtmp;
//	keywords = kwtmp;
//
//	vector<int> bjtmp;
//	belongtraj = bjtmp;
//
//}




float STPoint::CalcPPSTSim(const STPoint &p) const{
	//spacial
	float ssim = 0;
	ssim = 1 - calculateDistance(this->lat, this->lon, p.lat, p.lon) / MAX_DIST;
	//textual
	float tsim = 0;
	for (size_t i = 0; i < this->keywords.size(); i++) {
		for (size_t j = 0; j < p.keywords.size(); j++) {
			if (this->keywords[i].keywordid == p.keywords[j].keywordid) {
				tsim += this->keywords[i].keywordvalue*p.keywords[j].keywordvalue;
				break; // 单个点不会出现重复的keyword
			}
		}
	}
	return(ssim + tsim);
}