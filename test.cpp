#include "test.h"

extern void split3(string s, string delim, vector<string>* ret);

void test::testfunc() {
	vector<string> ssplit;
	string dot = "\t";
	string s = "57732	";
	split3(s, dot, &ssplit);
}