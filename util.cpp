#include "util.h"



// global function for Preprocess class; global wheels
void split(string s, string delim, vector<string>* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last); // find is ok 2
	while (index != string::npos) // not not success   string::npos is -1 actually
	{
		string stmp = s.substr(last, index - last);
		if (stmp.length() != 0) ret->push_back(stmp); // vector������push_back NULL�ַ�
		last = index + delim.size();
		index = s.find_first_of(delim, last);
		//cout << index << endl;
	}
	if (last < s.size()) // ��β��delim
	{
		string stmp = s.substr(last, index - last);
		if (stmp.length() != 0) ret->push_back(stmp);
	}
	//if (index - last > 0) // wrong judgement!!  ��β��delim index == string::npos
	//{
	//	ret->push_back(s.substr(last, index - last));
	//}
}


void split2(string s, string delim, vector<string>* ret)
{
	size_t last = 0;
	size_t index = s.find(delim, last); // find is ok 2
	while (index != string::npos) // not not success   string::npos is a big num
	{
		ret->push_back(s.substr(last, index - last));
		last = index + delim.length();
		index = s.find(delim, last);
		//cout << index << endl;
	}
	if (s.size() - last > 0) // index == string::npos ��β��delim
	{
		ret->push_back(s.substr(last, s.size() - last));
	}
}

void split3(string s, string delim, vector<string>* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last); // find is ok 2
	while (index != string::npos) // not not success   string::npos is a big num
	{
		string stmp = s.substr(last, index - last);
		if (stmp.length() != 0) {
			ret->push_back(stmp); // vector������push_back NULL�ַ�
			cout << stmp.length() << endl << stmp << endl;
		}
		last = index + delim.size();
		index = s.find_first_of(delim, last);
		//cout << index << endl;
	}
	if (last < s.size()) // ��β��delim
	{
		cout << "step2" << endl;
		string stmp = s.substr(last, index - last);
		if (stmp.length() != 0) {
			ret->push_back(stmp);
			cout << stmp.length() << endl << stmp << endl;
		}
	}
	//if (index - last > 0) // wrong judgement!!  ��β��delim index == string::npos
	//{
	//	ret->push_back(s.substr(last, index - last));
	//}
}


// inline �������� ��θ�
float calculateDistance(float LatA, float LonA, float LatB, float LonB)
{
	float Pi = 3.1415926;
	float R = 6371004; // ��λ����
	float MLatA, MLatB, MLonA, MLonB;
	MLatA = 90 - LatA;
	MLatB = 90 - LatB;
	MLonA = LonA;
	MLonB = LonB;
	float C = sin(MLatA)*sin(MLatB)*cos(MLonA - MLonB) + cos(MLatA)*cos(MLatB);
	float Distance = R*acos(C)*Pi / 180;
	return Distance;
}



void GetTF(vector<string>* keywords, vector<string>* corpus) {
	//set<string> stopwords = { " ","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","a's","aside","ask","asking","associated","at","available","away","awfully","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","came","can","cannot","cant","can't","cause","causes","certain","certainly","changes","clearly","c'mon","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","c's","currently","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","done","don't","down","downwards","during","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","hello","help","hence","her","here","hereafter","hereby","herein","here's","hereupon","hers","herself","he's","hi","him","himself","his","hither","hopefully","how","howbeit","however","i'd","ie","if","ignored","i'll","i'm","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","its","it's","itself","i've","just","keep","keeps","kept","know","known","knows","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","que","quite","qv","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","take","taken","tell","tends","th","than","thank","thanks","thanx","that","thats","that's","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","theres","there's","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","t's","twice","two","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","value","various","very","via","viz","vs","want","wants","was","wasn't","way","we","we'd","welcome","well","we'll","went","were","we're","weren't","we've","what","whatever","what's","when","whence","whenever","where","whereafter","whereas","whereby","wherein","where's","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","who's","whose","why","will","willing","wish","with","within","without","wonder","won't","would","wouldn't","yes","yet","you","you'd","you'll","your","you're","yours","yourself","yourselves","you've","zero","zt","ZT","zz","ZZ"};
	//cout << stopwords.size();

	/************ utilizing python for TF-IDF as well as InvertedIndex, no need for repeated wheel-creating ***********/

}


void GetSample(vector<size_t> &taskSet1, vector<size_t> &taskSet2, int sizeP, int sizeQ) {

	for (size_t i = 0; i < sizeP; i++) {
		taskSet1.push_back(15); // 4
	}
	//set<size_t> Q;
	for (size_t j = 0; j < sizeQ; j++) {
		taskSet2.push_back(15); // 4
	}

}

