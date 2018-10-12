#include "STPoint.h"
#include "util.h"

//extern double calculateDistance(double LatA, double LonA, double LatB, double LonB); //���벻ͨ�� ����: makefileû��ӣ���

// ���ݳ�Ա��ʼ�����ڽ��빹�캯��֮ǰ��ɵ�
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




double STPoint::CalcPPSTSim(const STPoint &p) const{
	//spacial
	double ssim = 0;
	ssim = 1 - calculateDistance(this->lat, this->lon, p.lat, p.lon) / MAX_DIST;
	
	//textual
	double tsim = 0;
	for (size_t i = 0; i < this->keywords.size(); i++) {
		for (size_t j = 0; j < p.keywords.size(); j++) {
			if (this->keywords[i].keywordid == p.keywords[j].keywordid) {
				tsim += this->keywords[i].keywordvalue*p.keywords[j].keywordvalue;
				break; // �����㲻������ظ���keyword
			}
		}
	}

	
	return(ssim + tsim);
}