#include "MBB.h"

using namespace std;
// for test

//MBB::MBB()
//{
//	xmin = 0;
//	xmax = 0;
//	ymin = 0;
//	ymax = 0;
//}
//
//MBB::MBB(double val_xmin,double val_ymin,double val_xmax,double val_ymax)
//{
//	xmin = val_xmin;
//	xmax = val_xmax;
//	ymin = val_ymin;
//	ymax = val_ymax;
//}
//
//bool MBB::pInBox(double x, double y) {
//	if (x <= xmax &&x >= xmin&&y <= ymax&&y >= ymin)
//		return true;
//	else
//		return false;
//}
//
//int BoxIntersect(MBB& a1, MBB& b1) {
//	MBB a, b;
//	bool swaped = false;
//	if (a1.xmin < b1.xmin) {
//		a = a1;
//		b = b1;
//	}
//	else if (a1.xmin == b1.xmin) {
//		if (a1.xmax > b1.xmax) {
//			a = a1;
//			b = b1;
//		}
//		else
//		{
//			b = a1;
//			a = b1;
//			swaped = true;
//		}
//	}
//	else
//	{
//		b = a1;
//		a = b1;
//		swaped = true;
//	}
//	if (b.xmin >= a.xmax)
//		return 0;
//	else
//	{
//		if (b.ymax <= a.ymin)
//			return 0;
//		else if (b.ymin >= a.ymax)
//			return 0;
//		else {
//			if (a.pInBox(b.xmin, b.ymin) && a.pInBox(b.xmin, b.ymax) && a.pInBox(b.xmax, b.ymin) && a.pInBox(b.xmax, b.ymax))
//			{
//				if (!swaped)
//					return 2;
//				else
//					return 3;
//			}
//			else
//				return 1;
//		}
//	}
//}
///* return 0:���ཻ
//return 1:�ཻ��������
//return 2:a1����b1
//return 3:b1����a1
//*/
//
//int MBB::intersect(MBB& b) {
//	return (BoxIntersect(*this, b));
//}
//
//int MBB::randomGenerateMBB(MBB& generated)
//{
//	double minx, maxx, miny, maxy;
//	minx = this->xmin;
//	miny = this->ymin;
//	maxx = this->xmax;
//	maxy = this->ymax;
//
//	double x1, x2;
//	double y1, y2;
//	x1 = (rand() / double(RAND_MAX))*(maxx - minx) + minx;
//	x2 = (rand() / double(RAND_MAX))*(maxx - minx) + minx;
//	y1 = (rand() / double(RAND_MAX))*(maxy - miny) + miny;
//	y2 = (rand() / double(RAND_MAX))*(maxy - miny) + miny;
//	minx = x1 > x2 ? x2 : x1;
//	maxx = x1 > x2 ? x1 : x2;
//	miny = y1 > y2 ? y2 : y1;
//	maxy = y1 > y2 ? y1 : y2;
//	generated.xmin = minx;
//	generated.xmax = maxx;
//	generated.ymin = miny;
//	generated.ymax = maxy;
//	return 0;
//}
//
///* return 0:���ཻ
//   return 1:�ཻ��������
//   return 2:this����b
//   return 3:b����this
//*/
//
//int MBB::printMBB() {
//	
//	cout << "MBB size: " << this->GetMBBArea() << endl;
//	return 0;
//
//
//}
//
//double MBB::GetMBBArea(void) {
//	return (this->xmax - this->xmin)*(this->ymax - this->ymin)*1e6;
//}
//MBB::~MBB()
//{
//}
