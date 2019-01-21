#include "test.h"

extern void split3(std::string s, std::string delim, std::vector<std::string>* ret);

void test::testfunc() {
	std::vector<std::string> ssplit;
	std::string dot = "\t";
	std::string s = "57732	";
	split3(s, dot, &ssplit);
}