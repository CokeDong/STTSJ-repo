#include <assert.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConstDefine.h"
#include "gpukernel.h"
//#include "util.h"

#define CUDA_CALL(x) { const cudaError_t a = (x); if (a!= cudaSuccess) { printf("\nCUDA Error: %s(err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

/*
GPU function definition.
All functions of GPU are defined here.
*/

__device__ inline float SSimGPU(float lat1, float lon1, float lat2, float lon2) {
	//float Pi = 3.1415926;
	//float R = 6371004;
	float MLatA, MLatB, MLonA, MLonB;
	MLatA = 90 - lat1;
	MLatB = 90 - lat2;
	MLonA = lon1;
	MLonB = lon2;
	float C = sin(MLatA)*sin(MLatB)*cos(MLonA - MLonB) + cos(MLatA)*cos(MLatB);
	float Distance = 6371004 *acos(C)*3.1415926 / 180;
	return (1 - Distance / MAX_DIST);
}

__device__ inline float TSimGPU(int* textDataIndexPi, int* textDataIndexQj, float* textDataValuePi, float* textDataValueQj,
	int numWordP, int numWordQ){


	// 单个thread：读取不平衡！！
	// 内存读取不符合常规操作，无法合并，同时每个thread取的个数互不相同


	// choice1: each time fetch
	// choice2: reg. store!! register. booming ?  max: 1502 min : 0 not recommended!!!


	// Q: uint32_t 和 int 比较出错？？ ：注意size()返回值是uint32_t即可 粗暴全部换成int!!!

	// may cause divergency may optimization Qj.keywordcnt=0??
	// i think is okay!!
	if (numWordP == 0 || numWordQ == 0) { return 0; }

	float tsimresult = 0;
//	if (numWordP > numWordQ) //each time fetch:  no need !! divergency cache的存在 不确定的顺序！！

	// calc tsim value
	for (size_t m = 0; m < numWordP; m++) {
		int tmpipim = textDataIndexPi[m]; // 编译器优化应该会有cache!! 不引入也一样？
		float tmpvpim = textDataValuePi[m];
		for (size_t n = 0; n < numWordQ; n++) {
			if (tmpipim == textDataIndexQj[n]) {
				tsimresult += tmpvpim*textDataValueQj[n];
				break;// 单个点不会出现重复的keyword whether 优化 不确定？？ GPU可以识别break continue 不过就是指令而已
			}
		}
	}
	return tsimresult;
}

__device__ void warpReduce(volatile float* sdata,int tid ){
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}


__global__ void computeSimGPU(float* latDataPGPU1, float* latDataQGPU1, float* lonDataPGPU1, float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
) {
	int bId = blockIdx.x;
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;


	//__shared__ size_t pmqnid, pmqid, pqid;
	//__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		//// not used in kernel-V1
		//pmqnid = task.keywordpmqnMatrixId;
		//pmqid = task.keywordpmqMatrixId;
		//pqid = task.keywordpqMatrixId;

		//keycntP = task.keycntP;
		//keycntQ = task.keycntQ;
		//textPid = task.textIdxP;
		//textQid = task.textIdxQ;
	}

	/*
	if (tId == 0) {
		task = stattableGPU[bId];

		// for cache task！
		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;
		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;

		// task.others have been processed in Host
		pmqnid = task.keywordpmqnMatrixId;
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		if (pointNumP > pointNumQ) {
			latDataPGPU = latDataPGPU1;
			latDataQGPU = latDataQGPU1;
			lonDataPGPU = lonDataPGPU1;
			lonDataQGPU = lonDataQGPU1;
			textDataPIndexGPU = textDataPIndexGPU1;
			textDataQIndexGPU = textDataQIndexGPU1;
			textDataPValueGPU = textDataPValueGPU1;
			textDataQValueGPU = textDataQValueGPU1;
			textIdxPGPU = textIdxPGPU1;
			textIdxQGPU = textIdxQGPU1;
			numWordPGPU = numWordPGPU1;
			numWordQGPU = numWordQGPU1;

			pointIdP = task.latlonIdxP;
			pointIdQ = task.latlonIdxQ;
			pointNumP = task.pointNumP;
			pointNumQ = task.pointNumQ;
			keycntP = task.keycntP;
			keycntQ = task.keycntQ;
			textPid = task.textIdxP;
			textQid = task.textIdxQ;
		}
		else {
			latDataQGPU = latDataPGPU1;
			latDataPGPU = latDataQGPU1;
			lonDataQGPU = lonDataPGPU1;
			lonDataPGPU = lonDataQGPU1;
			textDataQIndexGPU = textDataPIndexGPU1;
			textDataPIndexGPU = textDataQIndexGPU1;
			textDataQValueGPU = textDataPValueGPU1;
			textDataPValueGPU = textDataQValueGPU1;
			textIdxQGPU = textIdxPGPU1;
			textIdxPGPU = textIdxQGPU1;
			numWordQGPU = numWordPGPU1;
			numWordPGPU = numWordQGPU1;

			pointIdQ = task.latlonIdxP;
			pointIdP = task.latlonIdxQ;
			pointNumQ = task.pointNumP;
			pointNumP = task.pointNumQ;
			keycntQ = task.keycntP;
			keycntP = task.keycntQ;
			textQid = task.textIdxP;
			textPid = task.textIdxQ;
		}
	}
	*/

	__syncthreads();


	__shared__ int height, width;

	// 不妨设 numP > numQ

	// initialize maxSimRow maxSimColumn
	/*
	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {
	maxSimRow[tId + i*THREADNUM] = 0;
	maxSimColumn[tId + i*THREADNUM] = 0;
	}
	*/

	/*
	// STEP-0: GET the text-sim matrix(global memory)


	// pmqn
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
	int tmpflagi = i + tId % THREADROW;
	int pmindex, pmvalue;
	if (tmpflagi < keycntP) {
	pmindex = textDataPIndexGPU[textPid + tmpflagi];
	pmvalue = textDataPValueGPU[textPid + tmpflagi];
	}
	for (size_t j = 0; j < keycntQ; j += THREADCOLUMN) {
	int tmpflagj = j + tId / THREADROW;
	int qnindex, qnvalue;
	if (tmpflagj < keycntQ) {
	qnindex = textDataQIndexGPU[textQid + tmpflagj];
	qnvalue = textDataQValueGPU[textQid + tmpflagj];
	}
	// in such loop, can only index in this way!!
	keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
	if ((tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex == qnindex)) {
	keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
	}
	}
	}
	__syncthreads();


	// pmq
	// 16*16 方阵加速 -> 转置(~3x)
	// __shared__ int pointnumq, textidq;
	__shared__ float tmppmq[THREADROW2][THREADCOLUMN2];
	height = keycntP, width = pointNumQ;
	// two-layer loop similar to block-net
	for (size_t i = 0; i < keycntP; i += THREADROW2) {
	int tmpflagi = i + tId % THREADROW2;
	int tmpflagi2 = i + tId / THREADROW2;
	for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN2) {
	int tmpflagj = j + tId / THREADROW2;
	int tmpflagj2 = j + tId % THREADROW2;

	// similar to transpose
	// tmppmq[tId / THREADCOLUMN2][tId % THREADCOLUMN2] = 0; // 行方式
	tmppmq[tId % THREADROW2][tId / THREADROW2] = 0; // 列方式
	if ((tmpflagi < keycntP) && (tmpflagj < pointNumQ)) { // thread filtering
	int pointnumq, textidq;
	pointnumq = numWordQGPU[pointIdQ + tmpflagj];
	textidq = textIdxQGPU[pointIdQ + tmpflagj];
	for (size_t k = 0; k < pointnumq; k++) {
	// just (textidq + k) needs some effort
	tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi];
	}
	}

	__syncthreads();

	// bounding problem!
	if ((tmpflagi2 < keycntP) && (tmpflagj2 < pointNumQ)) { // thread filtering
	keypmqGPU[pmqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
	}
	}
	}


	// pq
	height = pointNumQ, width = pointNumP;
	for (size_t i = 0; i < pointNumQ; i += THREADROW2) {
	int tmpflagi = i + tId%THREADROW2;
	int tmpflagi2 = i + tId / THREADROW2;
	for (size_t j = 0; j < pointNumP; j += THREADCOLUMN2) {
	int tmpflagj = j + tId / THREADROW2;
	int tmpflagj2 = j + tId % THREADROW2;
	tmppmq[tId % THREADROW2][tId / THREADROW2] = 0;
	if ((tmpflagi < pointNumQ) && (tmpflagj < pointNumP)) {
	int pointnump, textidp;
	pointnump = numWordPGPU[pointIdP + tmpflagj];
	textidp = textIdxPGPU[pointIdP + tmpflagj];
	for (size_t k = 0; k < pointnump; k++) {
	tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pqid + (textidp + k)*height + tmpflagi];
	}
	}
	__syncthreads();
	if ((tmpflagi2 < pointNumQ) && (tmpflagj2 < pointNumP)) {
	keypqGPU[pqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
	}
	}
	}
	*/


	// STEP-1: GET the  final sim result: SimResultGPU
	// 潜在debug:
	// only correct when THREADNUM > MAXTRAJLEN;
	// initilize shared memory
	if (tId < MAXTRAJLEN) {
		maxSimRow[tId] = 0;
		maxSimColumn[tId] = 0;
	}
	__syncthreads();




	height = pointNumP, width = pointNumQ;
	// doesnot matter !!
	for (size_t i = 0; i < pointNumP; i += THREADROW) {
		// simply because of THREADROW = 32, THREADROW = 8, 32 > 8
		// here 列方式
		// not real 128 -> 32倍近似？？
		// but there is cache ??
		int tmpflagi = i + tId % THREADROW;

		float latP, latQ, lonP, lonQ;
		int textIdP, textIdQ, numWordP, numWordQ;
		if (tmpflagi < pointNumP) {
			latP = latDataPGPU[pointIdP + tmpflagi];
			lonP = lonDataPGPU[pointIdP + tmpflagi];
			textIdP = textIdxPGPU[pointIdP + tmpflagi];
			numWordP = numWordPGPU[pointIdP + tmpflagi];
			//printf("%f,%f \n", latP, lonP);
		}

		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			if (tmpflagj < pointNumQ) {
				latQ = latDataQGPU[pointIdQ + tmpflagj];
				lonQ = lonDataQGPU[pointIdQ + tmpflagj];
				textIdQ = textIdxQGPU[pointIdQ + tmpflagj];
				numWordQ = numWordQGPU[pointIdQ + tmpflagj];
			}

			tmpSim[tId] = -1;//技巧，省去下面的tID=0判断

							 // debug:  边界条件错误！！ 逻辑错误 太慢！！ nearly 2 days
							 // if (tmpflagi && pointNumQ)
			if ((tmpflagi < pointNumP) && (tmpflagj < pointNumQ)) { // bound condition

																  //// not recommended! divergency!!
																  //float tsim = 0;
																  //if (numWordP > numWordQ) {		
																  //}
																  //else {
																  //}

				float tsim = 0;

				// way1: fool
				tsim = TSimGPU(&textDataPIndexGPU[textIdP], &textDataQIndexGPU[textIdQ], &textDataPValueGPU[textIdP], &textDataQValueGPU[textIdQ], numWordP, numWordQ);

				// way2: store way -> fetch way	 fetch from global memory!! 
				//tsim = keypqGPU[pqid + tmpflagj*height + tmpflagi];


				float ssim = SSimGPU(latP, lonP, latQ, lonQ);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}
			//			else {
			//				
			//			}

			// block 同步
			// 很有必要
			__syncthreads();


			////
			//// //优化
			////if (tId == 0) {
			////	tid_row = i + THREADROW > pointNumP ? pointNumP - i : THREADROW;
			////	tid_column = j + THREADCOLUMN > pointNumP ? pointNumQ - j : THREADCOLUMN;
			////}
			////__syncthreads();
			////


			// ************--shared_mem process--************
			// very naive process 

			// get tmp-row-max: full warp active
			//tmpmaxsimRow[tId % THREADROW];
			float tmpmaxSim = -1;
			if (tId / THREADROW == 0) {
				for (size_t k = 0; k < THREADCOLUMN; k++) {
					if (tmpSim[k*THREADROW + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k*THREADROW + tId];
					}
				}
				maxSimRow[i + tId] = (maxSimRow[i + tId] > tmpmaxSim ? maxSimRow[i + tId] : tmpmaxSim);
			}
			__syncthreads(); // still need!

							 // get tmp-column-max: 1/32 warp active
							 //tmpmaxsimColumn[tId / THREADROW];
			tmpmaxSim = -1;
			if (tId%THREADROW == 0) {
				for (size_t k = 0; k < THREADROW; k++) {
					if (tmpSim[k + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k + tId];
					}
				}
				maxSimColumn[j + tId / THREADROW] = (maxSimColumn[j + tId / THREADROW] > tmpmaxSim ? maxSimColumn[j + tId / THREADROW] : tmpmaxSim);
			}
			__syncthreads(); // still need!


		}

	}


	// sum reduction


	// 潜在debug: 
	// 前提：
	//  THREADNUM > MAX-MAXTRAJLEN
	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimRow[tId] += maxSimRow[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimRow, tId);

	//	}

	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimColumn[tId] += maxSimColumn[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimColumn, tId);
    //}

	if (tId == 0) {
		SimResultGPU[bId] = maxSimRow[0] / pointNumP + maxSimColumn[0] / pointNumQ;
	}

}

/*****
computeTSimpmqn dependency:
	|_textDataPIndexGPU,textDataQIndexGPU,textDataPValueGPU,textDataQValueGPU --->|
	|	|_textPid,textQid													      |
	|_keypmqnGPU <----------------------------------------------------------------|
		|_pmqnid																 
		|_keycntP,keycntQ														  	
****/

__global__ void computeTSimpmqn(float* latDataPGPU1, float* latDataQGPU1, float* lonDataPGPU1, float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
) {

	int bId = blockIdx.x; // bId is the only index for block to determine where to fetch data, and is 0 ~ MAX_BLOCKNUM-1
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;

	//debug: 数据类型 big int !! -> int , size_t
	//__shared__ int pmqnid, pmqid, pqid;
	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		// debug: wrong silly code mistake!
		//pmqnid = task.keywordpmqMatrixId;
		//pmqid = task.keywordpmqnMatrixId;

		pmqnid = task.keywordpmqnMatrixId; // starting ID in GPU for each block, accumulated
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		
		
		textPid = task.textIdxP; // starting position of text data for each task / block, accumulated
		textQid = task.textIdxQ;
	}
	__syncthreads();


	// STEP-0: GET the text-sim matrix(global memory)
	__shared__ int height, width;

	// pmqn
	// keycntP including all the padding 
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
		int tmpflagi = i + tId % THREADROW;
		//debug: float -> int 精度问题 数据类型定义出错
		// int pmindex,pmvalue;
		int pmindex;
		float pmvalue;

		//if (tmpflagi < keycntP) {
		//	pmindex = textDataPIndexGPU[textPid + tmpflagi];
		//	//if (pmindex == -1) continue;
		//	pmvalue = textDataPValueGPU[textPid + tmpflagi];
		//}

		for (size_t j = 0; j < keycntQ; j += THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			int qnindex;
			float qnvalue;

			//if (tmpflagj < keycntQ) {
			//	qnindex = textDataQIndexGPU[textQid + tmpflagj];
			//	//if (qnindex == -1) continue;
			//	qnvalue = textDataQValueGPU[textQid + tmpflagj];
			//}

			// in such loop, can only index in this way!!
			// int -> size_t 兼容

			// debug: initialize:overlap among blocks
			//keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;

			if ((tmpflagi < keycntP) && (tmpflagj < keycntQ)) { // avoid overlapping of keypmqnGPU among blocks !!
				

				pmindex = textDataPIndexGPU[textPid + tmpflagi];
				//if (pmindex == -1) continue;
				pmvalue = textDataPValueGPU[textPid + tmpflagi];

				
				qnindex = textDataQIndexGPU[textQid + tmpflagj];
				//if (qnindex == -1) continue;
				qnvalue = textDataQValueGPU[textQid + tmpflagj];


				keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
				// debug: excluding padding here!
				if ((pmindex != -1) && (qnindex != -1) && (pmindex == qnindex)) {
					keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;	
					//printf("pmqn-> blockId:%d threadId:%d startpos:%d index:%zu value:%.5f\n", bId, tId, pmqnid, pmqnid + tmpflagj*height + tmpflagi, pmvalue*qnvalue);
					//printf("pmqn s1 -> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, pmvalue*qnvalue);
				}
				//printf("pmqn confirm -> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi]);
			}

			/*
			// debug: excluding padding here!
			if ((pmindex != -1) && (qnindex != -1) && (tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex == qnindex)) {
				keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
				//printf("pmqn-> blockId:%d threadId:%d startpos:%d index:%zu value:%.5f\n", bId, tId, pmqnid, pmqnid + tmpflagj*height + tmpflagi, pmvalue*qnvalue);
				
				//printf("pmqn-> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, pmvalue*qnvalue);
			
			}
			*/

			// block同步！ maybe not necessary because no overlap of memory write and read here, is register reused? 决定是否需要同步
			__syncthreads();

		}
	}


}

/******
computeTSimpmq dependency	
	|_keypmqnGPU ---------------------->|
	|	|_pmqnid						|
	|	|_numWordQGPU,textIdxQGPU		|
	|		|_pointIdQ					|
	|_keypmqGPU  <----------------------|
		|_pmqid
		|_keycntP,pointNumQ
*******/
__global__ void computeTSimpmq(float* latDataPGPU1, float* latDataQGPU1, float* lonDataPGPU1, float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
) {
	int bId = blockIdx.x;
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;

	
	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		// debug: wrong silly code mistake!
		//pmqnid = task.keywordpmqMatrixId;
		//pmqid = task.keywordpmqnMatrixId;
		pmqnid = task.keywordpmqnMatrixId;
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;
	}
	__syncthreads();


	// STEP-0: GET the text-sim matrix(global memory)
	__shared__ int height, width;



	// check pmqnMatrix
	//height = keycntP, width = keycntQ;
	//for (size_t i = 0; i < keycntP; i += THREADROW) {
	//	int tmpflagi = i + tId % THREADROW;
	//	for (size_t j = 0; j < keycntQ; j += THREADCOLUMN) {
	//		int tmpflagj = j + tId / THREADROW;
	//		if ((tmpflagi < keycntP) && (tmpflagj < keycntQ)) {
	//			printf("pmqn check -> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi]);
	//		}
	//	}
	//}





	// pmq
	// 16*16 方阵加速 -> 转置(~3x)
	// __shared__ int pointnumq, textidq;
	__shared__ float tmppmq[THREADROW2][THREADCOLUMN2];
	height = keycntP, width = pointNumQ;
	// two-layer loop similar to block-net
	for (size_t i = 0; i < keycntP; i += THREADROW2) {
		int tmpflagi = i + tId % THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;

			// similar to transpose
			// tmppmq[tId / THREADCOLUMN2][tId % THREADCOLUMN2] = 0; // 行方式

			// initialization of shared mem: tmppmq, must be here, or we have uninitialized tmppmq
			tmppmq[tId % THREADROW2][tId / THREADROW2] = 0; // 列方式
			if ((tmpflagi < keycntP) && (tmpflagj < pointNumQ)) { // thread filtering
				int keywordnumq, textidq;

				// ABOUT PADDING problem:
				keywordnumq = numWordQGPU[pointIdQ + tmpflagj]; // this is real # of keyword for each point without padding!
				textidq = textIdxQGPU[pointIdQ + tmpflagj]; // this is the keyword starting id for each point after padding, be careful!
											// but attention: the padding is traj-level, so the padding is always patched to the last point!! 
											// as long as the textidq and textQid accordant! as they make subtraction!
				for (size_t k = 0; k < keywordnumq; k++) {
					// just (textidq + k) needs some effort

					//if (bId == 60){  
					//	printf("************ special pmq-> k:%d blockId:%d threadId:%d value:%0.5f\n", k, bId, tId, keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi]);
					//}
					
					// debug: fecthing wrong keypmqnGPU here! data structure!
					tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq - textQid + k)*height + tmpflagi];
					//tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi];	
				
					//if (bId == 60){  
					//	printf("************ special pmq-> k:%d blockId:%d threadId:%d value:%0.5f\n", k, bId, tId, keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi]);
					//}				
				
				}
				//printf("pmq s1 -> blockId:%d threadId:%d keywordnumq:%d textidq:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, keywordnumq, textidq, tId%THREADROW2, tId / THREADROW2, tmppmq[tId % THREADROW2][tId / THREADROW2]);
			}

			// this is necessary ! because of tmppmq;
			__syncthreads();
			

			// this is not a propriate place for printf as no thread filtering
			//printf("pmq-> blockId:%d threadId:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, tId%THREADROW2, tId / THREADROW2, tmppmq[tId % THREADROW2][tId / THREADROW2]);


			// bounding problem! 
			if ((tmpflagi2 < keycntP) && (tmpflagj2 < pointNumQ)) { // thread filtering
				keypmqGPU[pmqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
				//printf("pmq s2-> blockId:%d threadId:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, tId / THREADROW2, tId % THREADROW2, tmppmq[tId / THREADROW2][tId % THREADROW2]);
			}

			// this is necessary ! because of tmppmq;
			__syncthreads();

		}
	}

}

/*******
computeTSimpq dependency
|_keypmqGPU ---------------------->|
|	|_pmqid						   |
|	|_numWordPGPU,textIdxPGPU	   |
|		|_pointIdP				   |
|_keypqGPU  <----------------------|
	|_pqid
	|_pointNumQ,pointNumP
*******/
__global__ void computeTSimpq(float* latDataPGPU1, float* latDataQGPU1, float* lonDataPGPU1, float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
) {
	int bId = blockIdx.x;
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;


	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		// debug: wrong silly code mistake!
		//pmqnid = task.keywordpmqMatrixId;
		//pmqid = task.keywordpmqnMatrixId;
		pmqnid = task.keywordpmqnMatrixId;
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;
	}
	__syncthreads();
	// STEP-0: GET the text-sim matrix(global memory)
	__shared__ int height, width;
	
	
	__shared__ float tmppq[THREADROW2][THREADCOLUMN2];
	
	// pq
	height = pointNumQ, width = pointNumP;
	for (size_t i = 0; i < pointNumQ; i += THREADROW2) {
		int tmpflagi = i + tId % THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumP; j += THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;
			tmppq[tId % THREADROW2][tId / THREADROW2] = 0;
			if ((tmpflagi < pointNumQ) && (tmpflagj < pointNumP)) {
				int pointnump, textidp;
				pointnump = numWordPGPU[pointIdP + tmpflagj];
				textidp = textIdxPGPU[pointIdP + tmpflagj];
				for (size_t k = 0; k < pointnump; k++) {
					//debug here: wrong index
					//tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pqid + (textidp + k)*height + tmpflagi];
					// debug: similar wrong index! as pmq
					tmppq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pmqid + (textidp - textPid + k)*height + tmpflagi];
					//printf("pq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId % THREADROW2][tId / THREADROW2]);
				}
			}
			__syncthreads();
			if ((tmpflagi2 < pointNumQ) && (tmpflagj2 < pointNumP)) {
				keypqGPU[pqid + tmpflagi2*width + tmpflagj2] = tmppq[tId / THREADROW2][tId % THREADROW2];
				//printf("pq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId / THREADROW2][tId % THREADROW2]);
			}
			__syncthreads();

		}
	}
}

/*****
computeSimGPUV2 dependency
	|_SimResultGPU
		|_pointNumP,pointNumQ
		|_keypqGPU
		|	|_pqid
		|_SSimGPU()
			|_latDataPGPU,lonDataPGPU,latDataQGPU,lonDataQGPU
				|_pointIdP,pointIdQ


*****/


__global__ void computeSimGPUV2(float* latDataPGPU1,float* latDataQGPU1,float* lonDataPGPU1,float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
){
	int bId = blockIdx.x;
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;


	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	
	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		// debug: wrong silly code mistake!
		//pmqnid = task.keywordpmqMatrixId;
		//pmqid = task.keywordpmqnMatrixId;
		pmqnid = task.keywordpmqnMatrixId;
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;
	}

	/*
	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		// for cache task！
		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;
		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;

		// task.others have been processed in Host
		pmqnid = task.keywordpmqMatrixId;
		pmqid = task.keywordpmqnMatrixId;
		pqid = task.keywordpqMatrixId;

		if (pointNumP > pointNumQ) {
			latDataPGPU = latDataPGPU1;
			latDataQGPU = latDataQGPU1;
			lonDataPGPU = lonDataPGPU1;
			lonDataQGPU = lonDataQGPU1;
			textDataPIndexGPU = textDataPIndexGPU1;
			textDataQIndexGPU = textDataQIndexGPU1;
			textDataPValueGPU = textDataPValueGPU1;
			textDataQValueGPU = textDataQValueGPU1;
			textIdxPGPU = textIdxPGPU1;
			textIdxQGPU = textIdxQGPU1;
			numWordPGPU = numWordPGPU1;
			numWordQGPU = numWordQGPU1;

			pointIdP = task.latlonIdxP;
			pointIdQ = task.latlonIdxQ;
			pointNumP = task.pointNumP;
			pointNumQ = task.pointNumQ;
			keycntP = task.keycntP;
			keycntQ = task.keycntQ;
			textPid = task.textIdxP;
			textQid = task.textIdxQ;
		}
		else {
			latDataQGPU = latDataPGPU1;
			latDataPGPU = latDataQGPU1;
			lonDataQGPU = lonDataPGPU1;
			lonDataPGPU = lonDataQGPU1;
			textDataQIndexGPU = textDataPIndexGPU1;
			textDataPIndexGPU = textDataQIndexGPU1;
			textDataQValueGPU = textDataPValueGPU1;
			textDataPValueGPU = textDataQValueGPU1;
			textIdxQGPU = textIdxPGPU1;
			textIdxPGPU = textIdxQGPU1;
			numWordQGPU = numWordPGPU1;
			numWordPGPU = numWordQGPU1;

			pointIdQ = task.latlonIdxP;
			pointIdP = task.latlonIdxQ;
			pointNumQ = task.pointNumP;
			pointNumP = task.pointNumQ;
			keycntQ = task.keycntP;
			keycntP = task.keycntQ;
			textQid = task.textIdxP;
			textPid = task.textIdxQ;
		}
	}
	*/


	__syncthreads();


	// 不妨设 numP > numQ

	// initialize maxSimRow maxSimColumn
	/*
	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {
	maxSimRow[tId + i*THREADNUM] = 0;
	maxSimColumn[tId + i*THREADNUM] = 0;
	}
	*/

	// STEP-0: GET the text-sim matrix(global memory)
	__shared__ int height, width;
	

	/*

	// pmqn
	// keycntP including all the padding 
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
		int tmpflagi = i + tId % THREADROW;
		//debug: float -> int 精度问题 数据类型定义出错
		// int pmindex,pmvalue;
		int pmindex;
		float pmvalue;
		if (tmpflagi < keycntP) {
			pmindex = textDataPIndexGPU[textPid + tmpflagi];
			//if (pmindex == -1) continue;
			pmvalue = textDataPValueGPU[textPid + tmpflagi];
		}
		for (size_t j = 0; j < keycntQ; j+=THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			int qnindex;
			float qnvalue;
			if (tmpflagj < keycntQ) {
				qnindex = textDataQIndexGPU[textQid + tmpflagj];
				//if (qnindex == -1) continue;
				qnvalue = textDataQValueGPU[textQid + tmpflagj];
			}		
			// in such loop, can only index in this way!!
			// int -> size_t 兼容
			keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
			// debug: excluding padding here!
			if ((pmindex != -1) && (qnindex != -1) &&(tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex == qnindex)) {
				keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
				//printf("pmqn-> blockId:%d threadId:%d value:%.5f\n", bId, tId, pmvalue*qnvalue);
			}

			// block同步！ maybe not necessary because no shared memory here, is register reused? 决定是否需要同步
			__syncthreads();
		}
	}

	
	__syncthreads(); // inpropriate, this have to be sync of different blocks, as ALL the global memory have to be used later.


	// pmq
	// 16*16 方阵加速 -> 转置(~3x)
	// __shared__ int pointnumq, textidq;
	__shared__ float tmppmq[THREADROW2][THREADCOLUMN2];
	height = keycntP, width = pointNumQ;
	// two-layer loop similar to block-net
	for (size_t i = 0; i < keycntP; i += THREADROW2) {	
		int tmpflagi = i + tId % THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;

			// similar to transpose
			// tmppmq[tId / THREADCOLUMN2][tId % THREADCOLUMN2] = 0; // 行方式
			tmppmq[tId % THREADROW2][tId / THREADROW2] = 0; // 列方式
			if ((tmpflagi < keycntP) && (tmpflagj < pointNumQ)) { // thread filtering
				int pointnumq, textidq;
				pointnumq = numWordQGPU[pointIdQ + tmpflagj];
				textidq = textIdxQGPU[pointIdQ + tmpflagj];
				for (size_t k = 0; k < pointnumq; k++) {
					// just (textidq + k) needs some effort
					tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi];
				}
				//printf("pmq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId % THREADROW2][tId / THREADROW2]);
			}

			__syncthreads();

			// bounding problem! 
			if ((tmpflagi2 < keycntP) && (tmpflagj2 < pointNumQ)) { // thread filtering
				keypmqGPU[pmqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
				//printf("pmq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId / THREADROW2][tId % THREADROW2]);
			}

			__syncthreads();
		}
	}
	__syncthreads();
	// inpropriate, this have to be sync of different blocks, as ALL the global memory have to be used later.

	// pq
	height = pointNumQ, width = pointNumP;
	for (size_t i = 0; i < pointNumQ; i+= THREADROW2) {
		int tmpflagi = i + tId%THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumP; j+= THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;
			tmppmq[tId % THREADROW2][tId / THREADROW2] = 0;
			if ((tmpflagi < pointNumQ) && (tmpflagj < pointNumP)) {
				int pointnump, textidp;
				pointnump = numWordPGPU[pointIdP + tmpflagj];
				textidp = textIdxPGPU[pointIdP + tmpflagj];
				for (size_t k = 0; k < pointnump; k++) {
					tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pqid + (textidp + k)*height + tmpflagi];
				}
			}
			__syncthreads();
			if ((tmpflagi2 < pointNumQ) && (tmpflagj2 < pointNumP)) {
				keypqGPU[pqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
			}

			__syncthreads();
		}
	}

	__syncthreads();

	*/


	// STEP-1: GET the  final sim result: SimResultGPU

	// only correct when THREADNUM > MAXTRAJLEN;
	// initilize shared memory
	if (tId < MAXTRAJLEN) {
		maxSimRow[tId] = 0;
		maxSimColumn[tId] = 0;
	}
	__syncthreads();


	
	

	height = pointNumP, width = pointNumQ;
	// doesnot matter !!
	for (size_t i = 0; i < pointNumP; i += THREADROW) {
		// simply because of THREADROW = 32, THREADROW = 8, 32 > 8
		// here 列方式
		// not real 128 -> 32倍近似？？
		// but there is cache ??	
		int tmpflagi = i + tId % THREADROW;
		float latP, latQ, lonP, lonQ;
		//int textIdP, textIdQ, numWordP, numWordQ;
		if(tmpflagi < pointNumP){
			latP = latDataPGPU[pointIdP + tmpflagi];
			lonP = lonDataPGPU[pointIdP + tmpflagi];
			//textIdP = textIdxPGPU[pointIdP + tmpflagi];
			//numWordP = numWordPGPU[pointIdP + tmpflagi];
			//printf("%f,%f \n", latP, lonP);
		}

		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			if (tmpflagj < pointNumQ) {
				latQ = latDataQGPU[pointIdQ + tmpflagj];
				lonQ = lonDataQGPU[pointIdQ + tmpflagj];
				//textIdQ = textIdxQGPU[pointIdQ + tmpflagj];
				//numWordQ = numWordQGPU[pointIdQ + tmpflagj];
			}

			tmpSim[tId] = -1;//技巧，省去下面的tID=0判断

			// debug:  边界条件错误！！ 逻辑错误 太慢！！ nearly 2 days
			// if (tmpflagi && pointNumQ)
			if ((tmpflagi< pointNumP) && (tmpflagj< pointNumQ)) { // bound condition

				//// not recommended! divergency!!
				//float tsim = 0;
				//if (numWordP > numWordQ) {		
				//}
				//else {
				//}

				float tsim = 0;

				// way1: fool
				//tsim = TSimGPU(&textDataPIndexGPU[textIdP], &textDataQIndexGPU[textIdQ], &textDataPValueGPU[textIdP], &textDataQValueGPU[textIdQ], numWordP, numWordQ);
				
				// way2: store way 决定-> fetch way	是否合并访问 fetch from global memory!! 
				tsim = keypqGPU[pqid + tmpflagj*height + tmpflagi];
			

				float ssim = SSimGPU(latP, lonP, latQ, lonQ);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}
//			else {
//				
//			}
			
			// block 同步
			// 很有必要
			__syncthreads();


			////
			//// //优化
			////if (tId == 0) {
			////	tid_row = i + THREADROW > pointNumP ? pointNumP - i : THREADROW;
			////	tid_column = j + THREADCOLUMN > pointNumP ? pointNumQ - j : THREADCOLUMN;
			////}
			////__syncthreads();
			////


			// ************--shared_mem process--************
			// very naive process 

			// get tmp-row-max: full warp active
			//tmpmaxsimRow[tId % THREADROW];
			float tmpmaxSim = -1;
			if (tId / THREADROW == 0) {
				for (size_t k = 0; k < THREADCOLUMN; k++) {
					if (tmpSim[k*THREADROW + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k*THREADROW + tId];
					}
				}
				maxSimRow[i + tId] = (maxSimRow[i + tId] > tmpmaxSim ? maxSimRow[i + tId] : tmpmaxSim);
			}
			__syncthreads(); // still need!

			// get tmp-column-max: 1/32 warp active
			//tmpmaxsimColumn[tId / THREADROW];
			tmpmaxSim = -1;
			if (tId%THREADROW == 0) {
				for (size_t k = 0; k < THREADROW; k++) {
					if (tmpSim[k + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k + tId];
					}
				}
				maxSimColumn[j + tId / THREADROW] = (maxSimColumn[j + tId / THREADROW] > tmpmaxSim ? maxSimColumn[j + tId / THREADROW] : tmpmaxSim);
			}
			__syncthreads(); // still need!


		}

	}


	
	// sum reduction

//	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {

	// 潜在debug: 
	// 前提：
	//  THREADNUM > MAX-MAXTRAJLEN
	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimRow[tId] += maxSimRow[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimRow, tId);

//	}

	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimColumn[tId] += maxSimColumn[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimColumn, tId);


	if (tId == 0) {
		SimResultGPU[bId] = maxSimRow[0] / pointNumP + maxSimColumn[0] / pointNumQ;
	}
	
}

__global__ void computeSimGPUV2p1(float* latDataPGPU1, float* latDataQGPU1, float* lonDataPGPU1, float* lonDataQGPU1,
	int* textDataPIndexGPU1, int* textDataQIndexGPU1, float* textDataPValueGPU1, float* textDataQValueGPU1,
	int* textIdxPGPU1, int* textIdxQGPU1, int* numWordPGPU1, int* numWordQGPU1,
	StatInfoTable* stattableGPU, float* keypmqnGPU, float* keypmqGPU, float* keypqGPU, float* SimResultGPU
) {
	int bId = blockIdx.x;
	int tId = threadIdx.x;

	// 1-D 没采用2-D 可自定义存储方式
	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	//__shared__ int tid_row;
	//__shared__ int tid_column;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;


	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int keycntP, keycntQ, textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];

		latDataPGPU = latDataPGPU1;
		latDataQGPU = latDataQGPU1;
		lonDataPGPU = lonDataPGPU1;
		lonDataQGPU = lonDataQGPU1;
		textDataPIndexGPU = textDataPIndexGPU1;
		textDataQIndexGPU = textDataQIndexGPU1;
		textDataPValueGPU = textDataPValueGPU1;
		textDataQValueGPU = textDataQValueGPU1;
		textIdxPGPU = textIdxPGPU1;
		textIdxQGPU = textIdxQGPU1;
		numWordPGPU = numWordPGPU1;
		numWordQGPU = numWordQGPU1;


		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;

		// debug: wrong silly code mistake!
		//pmqnid = task.keywordpmqMatrixId;
		//pmqid = task.keywordpmqnMatrixId;
		pmqnid = task.keywordpmqnMatrixId;
		pmqid = task.keywordpmqMatrixId;
		pqid = task.keywordpqMatrixId;

		keycntP = task.keycntP;
		keycntQ = task.keycntQ;
		textPid = task.textIdxP;
		textQid = task.textIdxQ;
	}

	/*
	//fetch task info
	if (tId == 0) {
	task = stattableGPU[bId];

	// for cache task！
	pointIdP = task.latlonIdxP;
	pointIdQ = task.latlonIdxQ;
	pointNumP = task.pointNumP;
	pointNumQ = task.pointNumQ;
	keycntP = task.keycntP;
	keycntQ = task.keycntQ;
	textPid = task.textIdxP;
	textQid = task.textIdxQ;

	// task.others have been processed in Host
	pmqnid = task.keywordpmqMatrixId;
	pmqid = task.keywordpmqnMatrixId;
	pqid = task.keywordpqMatrixId;

	if (pointNumP > pointNumQ) {
	latDataPGPU = latDataPGPU1;
	latDataQGPU = latDataQGPU1;
	lonDataPGPU = lonDataPGPU1;
	lonDataQGPU = lonDataQGPU1;
	textDataPIndexGPU = textDataPIndexGPU1;
	textDataQIndexGPU = textDataQIndexGPU1;
	textDataPValueGPU = textDataPValueGPU1;
	textDataQValueGPU = textDataQValueGPU1;
	textIdxPGPU = textIdxPGPU1;
	textIdxQGPU = textIdxQGPU1;
	numWordPGPU = numWordPGPU1;
	numWordQGPU = numWordQGPU1;

	pointIdP = task.latlonIdxP;
	pointIdQ = task.latlonIdxQ;
	pointNumP = task.pointNumP;
	pointNumQ = task.pointNumQ;
	keycntP = task.keycntP;
	keycntQ = task.keycntQ;
	textPid = task.textIdxP;
	textQid = task.textIdxQ;
	}
	else {
	latDataQGPU = latDataPGPU1;
	latDataPGPU = latDataQGPU1;
	lonDataQGPU = lonDataPGPU1;
	lonDataPGPU = lonDataQGPU1;
	textDataQIndexGPU = textDataPIndexGPU1;
	textDataPIndexGPU = textDataQIndexGPU1;
	textDataQValueGPU = textDataPValueGPU1;
	textDataPValueGPU = textDataQValueGPU1;
	textIdxQGPU = textIdxPGPU1;
	textIdxPGPU = textIdxQGPU1;
	numWordQGPU = numWordPGPU1;
	numWordPGPU = numWordQGPU1;

	pointIdQ = task.latlonIdxP;
	pointIdP = task.latlonIdxQ;
	pointNumQ = task.pointNumP;
	pointNumP = task.pointNumQ;
	keycntQ = task.keycntP;
	keycntP = task.keycntQ;
	textQid = task.textIdxP;
	textPid = task.textIdxQ;
	}
	}
	*/


	__syncthreads();


	// 不妨设 numP > numQ

	// initialize maxSimRow maxSimColumn
	/*
	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {
	maxSimRow[tId + i*THREADNUM] = 0;
	maxSimColumn[tId + i*THREADNUM] = 0;
	}
	*/

	// STEP-0: GET the text-sim matrix(global memory)
	__shared__ int height, width;


	/*

	// pmqn
	// keycntP including all the padding
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
	int tmpflagi = i + tId % THREADROW;
	//debug: float -> int 精度问题 数据类型定义出错
	// int pmindex,pmvalue;
	int pmindex;
	float pmvalue;
	if (tmpflagi < keycntP) {
	pmindex = textDataPIndexGPU[textPid + tmpflagi];
	//if (pmindex == -1) continue;
	pmvalue = textDataPValueGPU[textPid + tmpflagi];
	}
	for (size_t j = 0; j < keycntQ; j+=THREADCOLUMN) {
	int tmpflagj = j + tId / THREADROW;
	int qnindex;
	float qnvalue;
	if (tmpflagj < keycntQ) {
	qnindex = textDataQIndexGPU[textQid + tmpflagj];
	//if (qnindex == -1) continue;
	qnvalue = textDataQValueGPU[textQid + tmpflagj];
	}
	// in such loop, can only index in this way!!
	// int -> size_t 兼容
	keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
	// debug: excluding padding here!
	if ((pmindex != -1) && (qnindex != -1) &&(tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex == qnindex)) {
	keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
	//printf("pmqn-> blockId:%d threadId:%d value:%.5f\n", bId, tId, pmvalue*qnvalue);
	}

	// block同步！ maybe not necessary because no shared memory here, is register reused? 决定是否需要同步
	__syncthreads();
	}
	}


	__syncthreads(); // inpropriate, this have to be sync of different blocks, as ALL the global memory have to be used later.


	// pmq
	// 16*16 方阵加速 -> 转置(~3x)
	// __shared__ int pointnumq, textidq;
	__shared__ float tmppmq[THREADROW2][THREADCOLUMN2];
	height = keycntP, width = pointNumQ;
	// two-layer loop similar to block-net
	for (size_t i = 0; i < keycntP; i += THREADROW2) {
	int tmpflagi = i + tId % THREADROW2;
	int tmpflagi2 = i + tId / THREADROW2;
	for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN2) {
	int tmpflagj = j + tId / THREADROW2;
	int tmpflagj2 = j + tId % THREADROW2;

	// similar to transpose
	// tmppmq[tId / THREADCOLUMN2][tId % THREADCOLUMN2] = 0; // 行方式
	tmppmq[tId % THREADROW2][tId / THREADROW2] = 0; // 列方式
	if ((tmpflagi < keycntP) && (tmpflagj < pointNumQ)) { // thread filtering
	int pointnumq, textidq;
	pointnumq = numWordQGPU[pointIdQ + tmpflagj];
	textidq = textIdxQGPU[pointIdQ + tmpflagj];
	for (size_t k = 0; k < pointnumq; k++) {
	// just (textidq + k) needs some effort
	tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi];
	}
	//printf("pmq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId % THREADROW2][tId / THREADROW2]);
	}

	__syncthreads();

	// bounding problem!
	if ((tmpflagi2 < keycntP) && (tmpflagj2 < pointNumQ)) { // thread filtering
	keypmqGPU[pmqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
	//printf("pmq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId / THREADROW2][tId % THREADROW2]);
	}

	__syncthreads();
	}
	}
	__syncthreads();
	// inpropriate, this have to be sync of different blocks, as ALL the global memory have to be used later.

	// pq
	height = pointNumQ, width = pointNumP;
	for (size_t i = 0; i < pointNumQ; i+= THREADROW2) {
	int tmpflagi = i + tId%THREADROW2;
	int tmpflagi2 = i + tId / THREADROW2;
	for (size_t j = 0; j < pointNumP; j+= THREADCOLUMN2) {
	int tmpflagj = j + tId / THREADROW2;
	int tmpflagj2 = j + tId % THREADROW2;
	tmppmq[tId % THREADROW2][tId / THREADROW2] = 0;
	if ((tmpflagi < pointNumQ) && (tmpflagj < pointNumP)) {
	int pointnump, textidp;
	pointnump = numWordPGPU[pointIdP + tmpflagj];
	textidp = textIdxPGPU[pointIdP + tmpflagj];
	for (size_t k = 0; k < pointnump; k++) {
	tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pqid + (textidp + k)*height + tmpflagi];
	}
	}
	__syncthreads();
	if ((tmpflagi2 < pointNumQ) && (tmpflagj2 < pointNumP)) {
	keypqGPU[pqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
	}

	__syncthreads();
	}
	}

	__syncthreads();

	*/

	// pmqn
	// keycntP including all the padding 
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
	int tmpflagi = i + tId % THREADROW;
	//debug: float -> int 精度问题 数据类型定义出错
	// int pmindex,pmvalue;
	int pmindex;
	float pmvalue;

	//if (tmpflagi < keycntP) {
	//	pmindex = textDataPIndexGPU[textPid + tmpflagi];
	//	//if (pmindex == -1) continue;
	//	pmvalue = textDataPValueGPU[textPid + tmpflagi];
	//}

	for (size_t j = 0; j < keycntQ; j += THREADCOLUMN) {
		int tmpflagj = j + tId / THREADROW;
		int qnindex;
		float qnvalue;

		//if (tmpflagj < keycntQ) {
		//	qnindex = textDataQIndexGPU[textQid + tmpflagj];
		//	//if (qnindex == -1) continue;
		//	qnvalue = textDataQValueGPU[textQid + tmpflagj];
		//}

		// in such loop, can only index in this way!!
		// int -> size_t 兼容

		// debug: initialize:overlap among blocks
		//keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;

		if ((tmpflagi < keycntP) && (tmpflagj < keycntQ)) { // avoid overlapping of keypmqnGPU among blocks !!


			pmindex = textDataPIndexGPU[textPid + tmpflagi];
			//if (pmindex == -1) continue;
			pmvalue = textDataPValueGPU[textPid + tmpflagi];


			qnindex = textDataQIndexGPU[textQid + tmpflagj];
			//if (qnindex == -1) continue;
			qnvalue = textDataQValueGPU[textQid + tmpflagj];


			keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
			// debug: excluding padding here!
			if ((pmindex != -1) && (qnindex != -1) && (pmindex == qnindex)) {
				keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
				//printf("pmqn-> blockId:%d threadId:%d startpos:%d index:%zu value:%.5f\n", bId, tId, pmqnid, pmqnid + tmpflagj*height + tmpflagi, pmvalue*qnvalue);
				//printf("pmqn s1 -> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, pmvalue*qnvalue);
			}
			//printf("pmqn confirm -> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi]);
		}

		/*
		// debug: excluding padding here!
		if ((pmindex != -1) && (qnindex != -1) && (tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex == qnindex)) {
		keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = pmvalue*qnvalue;
		//printf("pmqn-> blockId:%d threadId:%d startpos:%d index:%zu value:%.5f\n", bId, tId, pmqnid, pmqnid + tmpflagj*height + tmpflagi, pmvalue*qnvalue);

		//printf("pmqn-> blockId:%d threadId:%d startpos:%d value:%.5f\n", bId, tId, pmqnid, pmvalue*qnvalue);

		}
		*/

		// block同步！ maybe not necessary because no overlap of memory write and read here, is register reused? 决定是否需要同步
		__syncthreads();

	}
	}


	// pmq
	// 16*16 方阵加速 -> 转置(~3x)
	// __shared__ int pointnumq, textidq;
	__shared__ float tmppmq[THREADROW2][THREADCOLUMN2]; 
	height = keycntP, width = pointNumQ;
	// two-layer loop similar to block-net
	for (size_t i = 0; i < keycntP; i += THREADROW2) {
		int tmpflagi = i + tId % THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;

			// similar to transpose
			// tmppmq[tId / THREADCOLUMN2][tId % THREADCOLUMN2] = 0; // 行方式

			// initialization of shared mem: tmppmq, must be here, or we have uninitialized tmppmq
			tmppmq[tId % THREADROW2][tId / THREADROW2] = 0; // 列方式
			if ((tmpflagi < keycntP) && (tmpflagj < pointNumQ)) { // thread filtering
				int keywordnumq, textidq;
				keywordnumq = numWordQGPU[pointIdQ + tmpflagj]; // this is real # of keyword for each point without padding!
				textidq = textIdxQGPU[pointIdQ + tmpflagj]; // this is the keyword starting id for each point after padding, be careful!
				for (size_t k = 0; k < keywordnumq; k++) {
					// just (textidq + k) needs some effort

					//if (bId == 60){  
					//	printf("************ special pmq-> k:%d blockId:%d threadId:%d value:%0.5f\n", k, bId, tId, keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi]);
					//}

					// debug: fecthing wrong keypmqnGPU here! data structure!
					tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq - textQid + k)*height + tmpflagi];
					//tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi];	

					//if (bId == 60){  
					//	printf("************ special pmq-> k:%d blockId:%d threadId:%d value:%0.5f\n", k, bId, tId, keypmqnGPU[pmqnid + (textidq + k)*height + tmpflagi]);
					//}				

				}
				//printf("pmq s1 -> blockId:%d threadId:%d keywordnumq:%d textidq:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, keywordnumq, textidq, tId%THREADROW2, tId / THREADROW2, tmppmq[tId % THREADROW2][tId / THREADROW2]);
			}

			// this is necessary ! because of tmppmq;
			__syncthreads();


			// this is not a propriate place for printf as no thread filtering
			//printf("pmq-> blockId:%d threadId:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, tId%THREADROW2, tId / THREADROW2, tmppmq[tId % THREADROW2][tId / THREADROW2]);


			// bounding problem! 
			if ((tmpflagi2 < keycntP) && (tmpflagj2 < pointNumQ)) { // thread filtering
				keypmqGPU[pmqid + tmpflagi2*width + tmpflagj2] = tmppmq[tId / THREADROW2][tId % THREADROW2];
				//printf("pmq s2-> blockId:%d threadId:%d xindex:%d yindex:%d value:%.5f\n", bId, tId, tId / THREADROW2, tId % THREADROW2, tmppmq[tId / THREADROW2][tId % THREADROW2]);
			}

			// this is necessary ! because of tmppmq;
			__syncthreads();

		}
	}


	// pq
	height = pointNumQ, width = pointNumP;
	__shared__ float tmppq[THREADROW2][THREADCOLUMN2];// in this way, no need for __syncthreads(); but maybe not that good ?
	for (size_t i = 0; i < pointNumQ; i += THREADROW2) {
		int tmpflagi = i + tId % THREADROW2;
		int tmpflagi2 = i + tId / THREADROW2;
		for (size_t j = 0; j < pointNumP; j += THREADCOLUMN2) {
			int tmpflagj = j + tId / THREADROW2;
			int tmpflagj2 = j + tId % THREADROW2;
			tmppq[tId % THREADROW2][tId / THREADROW2] = 0;
			if ((tmpflagi < pointNumQ) && (tmpflagj < pointNumP)) {
				int pointnump, textidp;
				pointnump = numWordPGPU[pointIdP + tmpflagj];
				textidp = textIdxPGPU[pointIdP + tmpflagj];
				for (size_t k = 0; k < pointnump; k++) {
					//debug here: wrong index
					//tmppmq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pqid + (textidp + k)*height + tmpflagi];
					// debug: similar wrong index! as pmq
					tmppq[tId % THREADROW2][tId / THREADROW2] += keypmqGPU[pmqid + (textidp - textPid + k)*height + tmpflagi];
					//printf("pq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId % THREADROW2][tId / THREADROW2]);
				}
			}
			__syncthreads();
			if ((tmpflagi2 < pointNumQ) && (tmpflagj2 < pointNumP)) {
				keypqGPU[pqid + tmpflagi2*width + tmpflagj2] = tmppq[tId / THREADROW2][tId % THREADROW2];
				//printf("pq-> blockId:%d threadId:%d value:%.5f\n", bId, tId, tmppmq[tId / THREADROW2][tId % THREADROW2]);
			}
			__syncthreads();

		}
	}





	// STEP-1: GET the  final sim result: SimResultGPU

	// only correct when THREADNUM > MAXTRAJLEN;
	// initilize shared memory
	if (tId < MAXTRAJLEN) {
		maxSimRow[tId] = 0;
		maxSimColumn[tId] = 0;
	}
	
	__syncthreads(); // we still have to syncthreads





	height = pointNumP, width = pointNumQ;
	// doesnot matter !!
	for (size_t i = 0; i < pointNumP; i += THREADROW) {
		// simply because of THREADROW = 32, THREADROW = 8, 32 > 8
		// here 列方式
		// not real 128 -> 32倍近似？？
		// but there is cache ??	
		int tmpflagi = i + tId % THREADROW;
		float latP, latQ, lonP, lonQ;
		//int textIdP, textIdQ, numWordP, numWordQ;
		if (tmpflagi < pointNumP) {
			latP = latDataPGPU[pointIdP + tmpflagi];
			lonP = lonDataPGPU[pointIdP + tmpflagi];
			//textIdP = textIdxPGPU[pointIdP + tmpflagi];
			//numWordP = numWordPGPU[pointIdP + tmpflagi];
			//printf("%f,%f \n", latP, lonP);
		}

		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			if (tmpflagj < pointNumQ) {
				latQ = latDataQGPU[pointIdQ + tmpflagj];
				lonQ = lonDataQGPU[pointIdQ + tmpflagj];
				//textIdQ = textIdxQGPU[pointIdQ + tmpflagj];
				//numWordQ = numWordQGPU[pointIdQ + tmpflagj];
			}

			tmpSim[tId] = -1;//技巧，省去下面的tID=0判断

							 // debug:  边界条件错误！！ 逻辑错误 太慢！！ nearly 2 days
							 // if (tmpflagi && pointNumQ)
			if ((tmpflagi< pointNumP) && (tmpflagj< pointNumQ)) { // bound condition

																  //// not recommended! divergency!!
																  //float tsim = 0;
																  //if (numWordP > numWordQ) {		
																  //}
																  //else {
																  //}

				float tsim = 0;

				// way1: fool
				//tsim = TSimGPU(&textDataPIndexGPU[textIdP], &textDataQIndexGPU[textIdQ], &textDataPValueGPU[textIdP], &textDataQValueGPU[textIdQ], numWordP, numWordQ);

				// way2: store way 决定-> fetch way	是否合并访问 fetch from global memory!! 
				tsim = keypqGPU[pqid + tmpflagj*height + tmpflagi];


				float ssim = SSimGPU(latP, lonP, latQ, lonQ);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}
			//			else {
			//				
			//			}

			// block 同步
			// 很有必要
			__syncthreads();


			////
			//// //优化
			////if (tId == 0) {
			////	tid_row = i + THREADROW > pointNumP ? pointNumP - i : THREADROW;
			////	tid_column = j + THREADCOLUMN > pointNumP ? pointNumQ - j : THREADCOLUMN;
			////}
			////__syncthreads();
			////


			// ************--shared_mem process--************
			// very naive process 

			// get tmp-row-max: full warp active
			//tmpmaxsimRow[tId % THREADROW];
			float tmpmaxSim = -1;
			if (tId / THREADROW == 0) {
				for (size_t k = 0; k < THREADCOLUMN; k++) {
					if (tmpSim[k*THREADROW + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k*THREADROW + tId];
					}
				}
				maxSimRow[i + tId] = (maxSimRow[i + tId] > tmpmaxSim ? maxSimRow[i + tId] : tmpmaxSim);
			}
			__syncthreads(); // still need!

							 // get tmp-column-max: 1/32 warp active
							 //tmpmaxsimColumn[tId / THREADROW];
			tmpmaxSim = -1;
			if (tId%THREADROW == 0) {
				for (size_t k = 0; k < THREADROW; k++) {
					if (tmpSim[k + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k + tId];
					}
				}
				maxSimColumn[j + tId / THREADROW] = (maxSimColumn[j + tId / THREADROW] > tmpmaxSim ? maxSimColumn[j + tId / THREADROW] : tmpmaxSim);
			}
			__syncthreads(); // still need!


		}

	}



	// sum reduction

	//	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {

	// 潜在debug: 
	// 前提：
	//  THREADNUM > MAX-MAXTRAJLEN
	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimRow[tId] += maxSimRow[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimRow, tId);

	//	}

	//for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
	for (size_t activethread = MAXTRAJLEN / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimColumn[tId] += maxSimColumn[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimColumn, tId);


	if (tId == 0) {
		SimResultGPU[bId] = maxSimRow[0] / pointNumP + maxSimColumn[0] / pointNumQ;
	}

}



// this is double-time-consuming ABORTED!! WARP consider not the most important, still have effect!! 
// heavy task: TSim calculation!
__global__ void computeSimGPUZhang(float* latDataPGPU, float* latDataQGPU, float* lonDataPGPU, float* lonDataQGPU,
	int* textDataPIndexGPU, int* textDataQIndexGPU, float* textDataPValueGPU, float* textDataQValueGPU,
	int* textIdxPGPU, int* textIdxQGPU, int* numWordPGPU, int* numWordQGPU,
	StatInfoTable* stattableGPU, float* SimResultGPU
) {

	int bId = blockIdx.x;
	int tId = threadIdx.x;

	__shared__ float tmpSim[THREADNUM];
	__shared__ float maxSim[MAXTRAJLEN]; // one more is because easy to reduce maximum
	__shared__ float maxSim1;


	__shared__ StatInfoTable task;
	__shared__ int pointIdP, pointNumP, pointIdQ, pointNumQ;

	if (tId == 0) {
		task = stattableGPU[bId];

		// for cache task！
		pointIdP = task.latlonIdxP;
		pointIdQ = task.latlonIdxQ;
		pointNumP = task.pointNumP;
		pointNumQ = task.pointNumQ;
	}
	__syncthreads();


	if (tId < MAXTRAJLEN) {
		maxSim[tId] = 0;
	}
	__syncthreads();

	float latP, latQ, lonP, lonQ;
	int textIdP, textIdQ, numWordP, numWordQ;

	for (size_t i = 0; i < pointNumP; i += THREADROW) {
		int tmpflagi = i + tId / 8; 
		if (tmpflagi < pointNumP) {
			latP = latDataPGPU[pointIdP + tmpflagi];
			lonP = lonDataPGPU[pointIdP + tmpflagi];
			textIdP = textIdxPGPU[pointIdP + tmpflagi];
			numWordP = numWordPGPU[pointIdP + tmpflagi];
			//printf("%f,%f \n", latP, lonP);
		}

		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN) {
			int tmpflagj = j + tId % 8;
			if (tmpflagj < pointNumQ) {
				latQ = latDataQGPU[pointIdQ + tmpflagj];
				lonQ = lonDataQGPU[pointIdQ + tmpflagj];
				textIdQ = textIdxQGPU[pointIdQ + tmpflagj];
				numWordQ = numWordQGPU[pointIdQ + tmpflagj];
			}

			tmpSim[tId] = -1;//技巧，省去下面的tID=0判断

			if ((tmpflagi< pointNumP) && (tmpflagj< pointNumQ)) { // bound condition

				float tsim = TSimGPU(&textDataPIndexGPU[textIdP], &textDataQIndexGPU[textIdQ], &textDataPValueGPU[textIdP], &textDataQValueGPU[textIdQ], numWordP, numWordQ);

				// fetch from global memory!!
				// warp 合并
				float ssim = SSimGPU(latP, lonP, latQ, lonQ);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}

			__syncthreads();



			float tmpmaxSim = -1;
			if (tId / THREADROW == 0) {
				for (size_t k = 0; k < THREADCOLUMN; k++) {
					if (tmpSim[k*8 + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k*THREADROW + tId];
					}
				}
				maxSim[i + tId] = (maxSim[i + tId] > tmpmaxSim ? maxSim[i + tId] : tmpmaxSim);
			}

			// tmpSim 可能会产生 warp 间 bank conflict，更加关注warp 内 bank conflict,故可以去掉
			__syncthreads(); // still need!


		}

	}



	// sum reduction

	
	for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSim[tId] += maxSim[tId + activethread];
			__syncthreads();
		}
	}
	if (tId < 32) warpReduce(maxSim, tId);
	maxSim1 = maxSim[0];
	__syncthreads();


	for (size_t i = 0; i < pointNumQ; i += THREADROW) {
		int tmpflagi = i + tId / 8;
		if (tmpflagi < pointNumQ) {
			latQ = latDataPGPU[pointIdQ + tmpflagi];
			lonQ = lonDataPGPU[pointIdQ + tmpflagi];
			textIdQ = textIdxPGPU[pointIdQ + tmpflagi];
			numWordQ = numWordPGPU[pointIdQ + tmpflagi];
			//printf("%f,%f \n", latP, lonP);
		}

		for (size_t j = 0; j < pointNumP; j += THREADCOLUMN) {
			int tmpflagj = j + tId % 8;
			if (tmpflagj < pointNumP) {
				latP = latDataQGPU[pointIdP + tmpflagj];
				lonP = lonDataQGPU[pointIdP + tmpflagj];
				textIdP = textIdxQGPU[pointIdP + tmpflagj];
				numWordP = numWordQGPU[pointIdP + tmpflagj];
			}

			tmpSim[tId] = -1;//技巧，省去下面的tID=0判断

			if ((tmpflagi< pointNumQ) && (tmpflagj< pointNumP)) { // bound condition

				float tsim = TSimGPU(&textDataQIndexGPU[textIdP], &textDataPIndexGPU[textIdQ], &textDataQValueGPU[textIdP], &textDataPValueGPU[textIdQ], numWordQ, numWordP);

				// fetch from global memory!!
				// warp 合并
				float ssim = SSimGPU(latQ, lonQ, latP, lonP);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}

			__syncthreads();

			float tmpmaxSim = -1;
			if (tId / THREADROW == 0) {
				for (size_t k = 0; k < THREADCOLUMN; k++) {
					if (tmpSim[k * 8 + tId] > tmpmaxSim) {
						tmpmaxSim = tmpSim[k*THREADROW + tId];
					}
				}
				maxSim[i + tId] = (maxSim[i + tId] > tmpmaxSim ? maxSim[i + tId] : tmpmaxSim);
			}

			__syncthreads(); // still need!


		}

	}



	// sum reduction

	for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSim[tId] += maxSim[tId + activethread];
			__syncthreads();
		}
	}
	if (tId < 32) warpReduce(maxSim, tId);
	__syncthreads();


	if (tId == 0) {
		SimResultGPU[bId] = maxSim1 / pointNumP + maxSim[0] / pointNumQ;
	}

}


























/*
CPU function definition.
All functions of CPU are defined here.
*/

inline void CUDAwarmUp() {
	CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost)); // zero-copy mem
	CUDA_CALL(cudaSetDevice(0)); // GPU-0
	if(DUALGPU) CUDA_CALL(cudaSetDevice(1)); // GPU-1
}

inline void CUDAwarmUp2() {
	// no zero-copy mem! here
	//CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost)); // zero-copy mem
	CUDA_CALL(cudaSetDevice(0)); // GPU-0
	if (DUALGPU) CUDA_CALL(cudaSetDevice(1)); // GPU-1
}

inline void* GPUMalloc(size_t byteNum) {
	void *addr;
	CUDA_CALL(cudaMalloc((void**)&addr, byteNum));
	return addr;
}

// ******* failure ********
// all pointer transimit, just for erase redundant codes
// The application may have hit an error when dereferencing Unified Memory from the host. Please rerun the application under cuda-gdb or Nsight Eclipse Edition to catch host side errors.
void STSimPQCommonCPUPreProcess(void* latDataPGPU, void* latDataQGPU, void* lonDataPGPU, void* lonDataQGPU,
	void* textDataPIndexGPU, void* textDataQIndexGPU, void* textDataPValueGPU, void* textDataQValueGPU,
	void* textIdxPGPU, void* textIdxQGPU, void* numWordPGPU, void* numWordQGPU,
	void* stattableGPU, void* keypmqnMatrixGPU, void* keypmqMatrixGPU, void* keypqMatrixGPU, //float* SimResultGPU,
	StatInfoTable* stattableCPU, void* gpuAddrPSet, void* gpuAddrQSet, void* gpuAddrStat, //float *SimResult,
	//cudaEvent_t* memcpy_to_start, cudaEvent_t* kernel_start, cudaEvent_t* kernel_stop,
	vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	cudaStream_t* stream, // can or not? maybe this way is not allowed!
	MyTimer* timer
) {
	CUDAwarmUp();

	
	//void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	//void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL

	// here only for quick occupying GPU 
	gpuAddrPSet = GPUMalloc((size_t)20 * 1024 * 1024);
	gpuAddrQSet = GPUMalloc((size_t)20 * 1024 * 1024);
	gpuAddrStat = GPUMalloc((size_t)10 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	//cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;

	//CUDA_CALL(cudaEventCreate(memcpy_to_start));
	//CUDA_CALL(cudaEventCreate(kernel_start));
	//CUDA_CALL(cudaEventCreate(kernel_stop));

	//cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(stream));


	//MyTimer timer;
	(*timer).start();

	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data

	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

	//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
	// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
	//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

										  // for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;

	// 需要手动free!!
	stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };



	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {

		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			// debug here!: logic mistakes cause wrong results! 18/12/5
			stattableCPU[i*dataSizeQ + j].textIdxP = textPId; // we only have to store the index for text trajectory grain, not point grain!

		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);

			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size()); // store the index for text point grain
			textIdxPCPU.push_back(textPId); // store the index for text point grain

			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);

	}

	//CUDA_CALL(cudaEventRecord((*memcpy_to_start), (*stream)));

	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	latDataPGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	lonDataPGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textIdxPGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	numWordPGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textDataPIndexGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textDataPValueGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());



	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);
			latlonQId++;
			// need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);

				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;
				keywordcnt++;
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);

		//for (size_t j = 0; j < dataSizeP; j++) {
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	latDataQGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textIdxQGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	numWordQGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textDataQIndexGPU = (int*)pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, (*stream)));
	textDataQValueGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());



	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			// not symmetric Matrix processing
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			if (stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ) {
				pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			}
			else {
				pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			}

			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			// each block
			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];

		}
	}


	pnow = gpuAddrStat;
	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, (*stream)));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = (StatInfoTable*)pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);
	keypmqnMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqnid);
	keypmqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqid);
	keypqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pqid);


	// debug: big int -> size_t
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** total status size *****%f GB\n", (pmqnid + pmqid + pqid)*4.0 / 1024 / 1024 / 1024);

	
	// zero-copy 内存 
	// 需要手动free!!

	// cannot be here, or SEG ERROR
	////float *SimResult, *SimResultGPU;
	//CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	//CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));

	(*timer).stop();
	printf("CPU  processing time: %f s\n", (*timer).elapse());

}


void STSimPQCommonCPUFinalProcess() {

	// no need

}


void STSimilarityJoinCalcGPU(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result) {


	/*
	void* gpuAddrPSet, *gpuAddrQSet, *gpuAddrStat;
	//cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	cudaStream_t stream;
	MyTimer *timer;
	StatInfoTable* stattableCPU;

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	float *SimResult;
	float *SimResultGPU;


	//// CPU processing
	//STSimPQCommonCPUPreProcess(latDataPGPU, latDataQGPU, lonDataPGPU, lonDataQGPU,
	//	textDataPIndexGPU, textDataQIndexGPU, textDataPValueGPU, textDataQValueGPU,
	//	textIdxPGPU, textIdxQGPU, numWordPGPU, numWordQGPU,
	//	stattableGPU, keypmqnMatrixGPU, keypmqMatrixGPU, keypqMatrixGPU, //SimResultGPU,
	//	stattableCPU, gpuAddrPSet, gpuAddrQSet, gpuAddrStat, //SimResult,
	//	//&memcpy_to_start, &kernel_start, &kernel_stop,
	//	trajSetP,
	//	trajSetQ,
	//	&stream,
	//	timer);


	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();
	//cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	//CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	//CUDA_CALL(cudaEventCreate(&kernel_start));
	//CUDA_CALL(cudaEventCreate(&kernel_stop));


	//float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));


	*/

	MyTimer timer;
	timer.start();

	CUDAwarmUp();
	
	//void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	//void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL

	// here only for quick occupying GPU 
	void* gpuAddrPSet = GPUMalloc((size_t)20 * 1024 * 1024);
	void* gpuAddrQSet = GPUMalloc((size_t)20 * 1024 * 1024); 
	void* gpuAddrStat = GPUMalloc((size_t)1 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));
	
	


	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

	//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
													// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
	//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable
	
	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

	// for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;
	
	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };


	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		
		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			// debug here!: logic mistakes cause wrong results! 18/12/5
			stattableCPU[i*dataSizeQ + j].textIdxP = textPId; // we only have to store the index for text trajectory grain

		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);

			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size()); // store the index for text point grain
			textIdxPCPU.push_back(textPId); // store the index for text point grain

			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32-remainder)/4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);
		

		//pointcntTrajP.push_back()
	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);
			latlonQId++;
			// need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				
				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;
				keywordcnt++;
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);
		//for (size_t j = 0; j < dataSizeP; j++) {
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());


	
	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			//// not symmetric Matrix processing
			//stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			//if(stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ){
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}
			
			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			// each block
			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];

		}
	}

	pnow = gpuAddrStat;
	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);
	keypmqnMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqnid);
	keypmqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqid);
	keypqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pqid);

	// debug: big int -> size_t
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** total status size *****%f GB\n", (pmqnid + pmqid + pqid)*4.0 / 1024 / 1024 / 1024);
	
	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));

	timer.stop();
	printf("CPU  processing time: %f s\n", timer.elapse());
	



	// running kernel
	
	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaStreamSynchronize(stream));



	CUDA_CALL(cudaEventRecord(kernel_start, stream));
	computeSimGPU << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	CUDA_CALL(cudaEventRecord(kernel_stop, stream));


	CUDA_CALL(cudaStreamSynchronize(stream));
	//CUDA_CALL(cudaDeviceSynchronize());
	
	timer.start();// here is appropriate!

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);

	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}
	timer.stop();
	printf("resultback time: (calculated by timer)%f s\n", timer.elapse()); // very quick!! but nzc is not slow as well!!
	timer.start();

	// free CPU memory
	free(stattableCPU);

	// free GPU memory
	CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrPSet));
	CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
    CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse());

	//return;

	
}


void STSimilarityJoinCalcGPUNoZeroCopy(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	float* result) {


	/*
	void* gpuAddrPSet, *gpuAddrQSet, *gpuAddrStat;
	//cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	cudaStream_t stream;
	MyTimer *timer;
	StatInfoTable* stattableCPU;

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	float *SimResult;
	float *SimResultGPU;


	//// CPU processing
	//STSimPQCommonCPUPreProcess(latDataPGPU, latDataQGPU, lonDataPGPU, lonDataQGPU,
	//	textDataPIndexGPU, textDataQIndexGPU, textDataPValueGPU, textDataQValueGPU,
	//	textIdxPGPU, textIdxQGPU, numWordPGPU, numWordQGPU,
	//	stattableGPU, keypmqnMatrixGPU, keypmqMatrixGPU, keypqMatrixGPU, //SimResultGPU,
	//	stattableCPU, gpuAddrPSet, gpuAddrQSet, gpuAddrStat, //SimResult,
	//	//&memcpy_to_start, &kernel_start, &kernel_stop,
	//	trajSetP,
	//	trajSetQ,
	//	&stream,
	//	timer);


	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();
	//cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	//CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	//CUDA_CALL(cudaEventCreate(&kernel_start));
	//CUDA_CALL(cudaEventCreate(&kernel_stop));


	//float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));
	*/
	MyTimer timer;
	timer.start(); // here: 0.X s

	CUDAwarmUp2();

	//CUDA_CALL(cudaSetDevice(0)); // GPU-0
	//if (DUALGPU) CUDA_CALL(cudaSetDevice(1)); // GPU-1


	//void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	//void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	//void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);


	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL

	//timer.start(); // here: 0.X s

	// here only for quick occupying GPU

	
	// cudaMalloc()如果是第一个常规runtime函数的话（cudaGetDeviceCount / cudaDeviceReset / cudaSetDevice这些特殊的不算），的确会引入一定的初始化时间
	// Warm up
	// GPU sample: not calculating this!!
	void* gpuAddrPSet, *gpuAddrQSet, *gpuAddrStat;
	CUDA_CALL(cudaMalloc((void**)&gpuAddrPSet, (size_t)20 * 1024 * 1024));
	CUDA_CALL(cudaMalloc((void**)&gpuAddrQSet, (size_t)20 * 1024 * 1024));
	CUDA_CALL(cudaMalloc((void**)&gpuAddrStat, (size_t)1 * 1024 * 1024 * 1024));
	
	//void* gpuAddrPSet = GPUMalloc((size_t)20 * 1024 * 1024);
	//void* gpuAddrQSet = GPUMalloc((size_t)20 * 1024 * 1024);
	//void* gpuAddrStat = GPUMalloc((size_t)1 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																   //void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);
	//timer.start(); // here: 0.000X s

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));


	
	//timer.start(); // here: 0.000X s, here is right !!

	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

											//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
											// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
											//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

										  // for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;

	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };


	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {

		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			// debug here!: logic mistakes cause wrong results! 18/12/5
			stattableCPU[i*dataSizeQ + j].textIdxP = textPId; // we only have to store the index for text trajectory grain

		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);

			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size()); // store the index for text point grain
			textIdxPCPU.push_back(textPId); // store the index for text point grain

			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);


		//pointcntTrajP.push_back()
	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);
			latlonQId++;
			// need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);

				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;
				keywordcnt++;
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);
		//for (size_t j = 0; j < dataSizeP; j++) {
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());



	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			//// not symmetric Matrix processing
			//stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			//if(stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ){
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}

			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			// each block
			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];

		}
	}

	pnow = gpuAddrStat;
	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);

	// must before keypmqnMatrixGPU keypmqMatrixGPU keypqMatrixGPU
	// this is very small memory, actually
	float *SimResultGPU;
	SimResultGPU = (float*)pnow;
	CUDA_CALL(cudaMemset(pnow, 0, sizeof(float) * dataSizeP * dataSizeQ)); // may unnecessary!
	pnow = (void*)((float*)pnow + dataSizeP * dataSizeQ);


	// may 飞掉 没关系，不用即可
	keypmqnMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqnid);
	keypmqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqid);
	keypqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pqid);


	// debug: big int -> size_t
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** total status size *****%f GB\n", (pmqnid + pmqid + pqid)*4.0 / 1024 / 1024 / 1024);


	/*
	// zero-copy 内存
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));
	*/


	timer.stop();
	printf("CPU  processing time: %f s\n", timer.elapse());



	//timer.start(); // including kernel time, so here is wrong!

	// running kernel

	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaStreamSynchronize(stream));



	CUDA_CALL(cudaEventRecord(kernel_start, stream));
	computeSimGPU << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	CUDA_CALL(cudaEventRecord(kernel_stop, stream));

	

	CUDA_CALL(cudaStreamSynchronize(stream)); // this is necessary, as SimResultGPU(of course and formal asynccpy) will return control to CPU at once! must be calculated done by kernel computeSimGPU, have proved!
	//CUDA_CALL(cudaDeviceSynchronize());


	//timer.start(); // here: 0.X s must after cudaStreamSynchronize, but not propriate here !

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);

	cudaEvent_t resultback;
	CUDA_CALL(cudaEventCreate(&resultback));
	float resultback_time = 0.0;

	// GPU->CPU 传回结果
	//float *SimResult = new float[dataSizeP * dataSizeQ];
	CUDA_CALL(cudaMemcpyAsync(result, SimResultGPU, sizeof(float) * dataSizeP * dataSizeQ, cudaMemcpyDeviceToHost, stream));
	CUDA_CALL(cudaEventRecord(resultback, stream));

	
	CUDA_CALL(cudaStreamSynchronize(stream)); //  this is necessary, we just have to look at later codes! cudaMemcpyAsync will return control to CPU at once! result is used!!!

	
	timer.start(); // here: 0.X s must after cudaStreamSynchronize

	
	CUDA_CALL(cudaEventElapsedTime(&resultback_time, kernel_stop, resultback));
	printf("resultback time: %.5f s\n", resultback_time / 1000.0);
	
	//CUDA_CALL(cudaDeviceSynchronize());



	//for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
	//	result[i] = SimResult[i];
	//}


	// free CPU memory
	free(stattableCPU);
	//delete[] SimResult;


	// free GPU memory
	//CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrPSet));
	CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management -> 0.X s overload !!
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaEventDestroy(resultback));

	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse());

	//return;


}




void STSimilarityJoinCalcGPUV2(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result) {


	MyTimer timer;
	timer.start();

	CUDAwarmUp();
	/*
	void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	*/

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL

	// here only for quick occupying GPU 
	void* gpuAddrPSet = GPUMalloc((size_t)20 * 1024 * 1024);
	void* gpuAddrQSet = GPUMalloc((size_t)20 * 1024 * 1024);
	void* gpuAddrStat = GPUMalloc((size_t)1 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));




	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

											//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
											// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
											//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

										  // for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;

	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {

		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			stattableCPU[i*dataSizeQ + j].textIdxP = textPId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);
			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size());
			textIdxPCPU.push_back(textPId);
			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);// keycnt including padding , if want not including padding, to do
		
		//for (size_t j = 0; j < dataSizeQ; j++) {
		//	stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
		//}

		//pointcntTrajP.push_back()
	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId; 
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);

			latlonQId++; // grain of each point, accumulated

			// need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);

				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;// grain of each point, accumulated
				keywordcnt++; // grain of each trajectory, not-accumulated
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;// grain of each point 
				keywordcnt++; // grain of each trajectory
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);


		//for (size_t j = 0; j < dataSizeP; j++) {
		//	// debug: this is wrong data structure!
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());



	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			// not symmetric Matrix processing  -> aborted first programming! to be easier
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			///*
			// maybe this is not wrong, but may cause high coupling with kernel!
			//if (stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ) {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}
			//*/

			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];

		}
	}

	pnow = gpuAddrStat;
	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);
	keypmqnMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqnid);
	keypmqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqid);
	keypqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pqid);

	// debug: big int -> size_t
	//OutGPUMemNeeded(pmqnid, pmqid,pqid);
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** total status size *****%f GB\n", (pmqnid + pmqid + pqid)*4.0 / 1024 / 1024 / 1024);

	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));

	timer.stop();
	printf("CPU  processing time: %f s\n", timer.elapse()); // data pre-processing on CPU

	// running kernel
	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaStreamSynchronize(stream));


	CUDA_CALL(cudaEventRecord(kernel_start, stream));


	// multi-kernel, but no need, because different block have no overlap between global memory! for keypmqnMatrixGPU keypmqMatrixGPU keypqMatrixGPU


	computeTSimpmqn << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);

	// debug: 非默认stream, this is necessary ? or not at all? ： NO , no overlap between global memory
	//CUDA_CALL(cudaStreamSynchronize(stream));


	computeTSimpmq << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	//CUDA_CALL(cudaStreamSynchronize(stream));


	computeTSimpq << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	//CUDA_CALL(cudaStreamSynchronize(stream));

	// above three can be merged!

	computeSimGPUV2 << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);


	CUDA_CALL(cudaEventRecord(kernel_stop, stream));

	//CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaStreamSynchronize(stream)); // be here is good,and necessary
	
	timer.start();


	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);


	// here has about 2s latency
	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}

	timer.stop();
	printf("resultback time: (calculated by timer)%f s\n", timer.elapse()); // very quick!! but nzc is not slow as well!!
	timer.start();

	// free CPU memory
	free(stattableCPU);

	// free GPU memory
	// debug: cudaFree doesn't erase anything!! it simply returns memory to a pool to be re-allocated
	// cudaMalloc doesn't guarantee the value of memory that has been allocated (to 0)
	// You need to Initialize memory (both global and shared) that your program uses, in order to have consistent results!!
	// The same is true for malloc and free, by the way
	CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrPSet));
	CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse()); // cuda-managing time
	//return;
}




void STSimilarityJoinCalcGPUV2p1(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result) {

	CUDAwarmUp();
	/*
	void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	*/

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL

	// here only for quick occupying GPU 
	void* gpuAddrPSet = GPUMalloc((size_t)20 * 1024 * 1024);
	void* gpuAddrQSet = GPUMalloc((size_t)20 * 1024 * 1024);
	void* gpuAddrStat = GPUMalloc((size_t)1 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	// 非默认stream中的数据传输使用函数cudaMemcpyAsync()? 不对 默认的也可以用
	//cudaStream_t stream = NULL; 
	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));



	MyTimer timer;
	timer.start();

	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

											//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
											// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
											//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

										  // for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;

	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {

		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			stattableCPU[i*dataSizeQ + j].textIdxP = textPId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);
			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size());
			textIdxPCPU.push_back(textPId);
			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);// keycnt including padding , if want not including padding, to do

										  //for (size_t j = 0; j < dataSizeQ; j++) {
										  //	stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
										  //}

										  //pointcntTrajP.push_back()
	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);

			latlonQId++; // grain of each point, accumulated

						 // need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);

				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;// grain of each point, accumulated
				keywordcnt++; // grain of each trajectory, not-accumulated
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;// grain of each point 
				keywordcnt++; // grain of each trajectory
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);


		//for (size_t j = 0; j < dataSizeP; j++) {
		//	// debug: this is wrong data structure!
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());



	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			// not symmetric Matrix processing  -> aborted first programming! to be easier
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			///*
			// maybe this is not wrong, but may cause high coupling with kernel!
			//if (stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ) {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}
			//*/

			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];

		}
	}

	pnow = gpuAddrStat;
	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);
	keypmqnMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqnid);
	keypmqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pmqid);
	keypqMatrixGPU = (float*)pnow;
	pnow = (void*)((float*)pnow + pqid);

	// debug: big int -> size_t
	//OutGPUMemNeeded(pmqnid, pmqid,pqid);
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** total status size *****%f GB\n", (pmqnid + pmqid + pqid)*4.0 / 1024 / 1024 / 1024);

	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));

	timer.stop();
	printf("CPU  processing time: %f s\n", timer.elapse());

	// running kernel
	//CUDA_CALL(cudaDeviceSynchronize());
	//CUDA_CALL(cudaStreamSynchronize(stream));


	CUDA_CALL(cudaEventRecord(kernel_start, 0));

	/*
	// no need, because different block have no overlap between global memory! for keypmqnMatrixGPU keypmqMatrixGPU keypqMatrixGPU
	computeTSimpmqn << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);

	// debug: 非默认stream, this is necessary ? or not at all? ： NO 
	//CUDA_CALL(cudaStreamSynchronize(stream));


	computeTSimpmq << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	//CUDA_CALL(cudaStreamSynchronize(stream));


	computeTSimpq << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	//CUDA_CALL(cudaStreamSynchronize(stream));

	// above three can be merged!
	*/

	computeSimGPUV2p1 << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);


	CUDA_CALL(cudaEventRecord(kernel_stop, stream));

	//CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaStreamSynchronize(stream)); // be here is good
	//CUDA_CALL(cudaEventSynchronize(kernel_stop));
	
	timer.start();

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);

	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}

	timer.stop();
	printf("resultback time: (calculated by timer)%f s\n", timer.elapse()); // very quick!! but nzc is not slow as well!!
	timer.start();


	// free CPU memory
	free(stattableCPU);

	// free GPU memory
	// debug: cudaFree doesn't erase anything!! it simply returns memory to a pool to be re-allocated
	// cudaMalloc doesn't guarantee the value of memory that has been allocated (to 0)
	// You need to Initialize memory (both global and shared) that your program uses, in order to have consistent results!!
	// The same is true for malloc and free, by the way
	CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrPSet));
	CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	if(stream != NULL) CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse());
	//return;
}






void STSimilarityJoinCalcGPUV3(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result) {


	MyTimer timer;
	timer.start();

	CUDAwarmUp();
	/*
	void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	*/

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL
	int gpuData = 500;
	//int gpuQSet = 20;
	int gpuStat = 6; // please be BIG ! 
	// here only for quick occupying GPU 
	void* gpuAddrData = GPUMalloc((size_t)gpuData * 1024 * 1024);
	//void* gpuAddrQSet = GPUMalloc((size_t)gpuQSet * 1024 * 1024);
	void* gpuAddrStat = GPUMalloc((size_t)gpuStat * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																   //void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));




	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

											//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
											// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
											//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

										  // for status info.
	vector<int> keycntTrajP, keycntTrajQ;
	//vector<int> pointcntTrajP, pointcntTrajQ;

	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {

		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			stattableCPU[i*dataSizeQ + j].textIdxP = textPId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);
			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size());
			textIdxPCPU.push_back(textPId);
			latlonPId++;
			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {
				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}
		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;
			}
		}

		keycntTrajP.push_back(keywordcnt);// keycnt including padding , if want not including padding, to do

										  //for (size_t j = 0; j < dataSizeQ; j++) {
										  //	stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
										  //}

										  //pointcntTrajP.push_back()
	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrData;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}

		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);

			latlonQId++; // grain of each point, accumulated

						 // need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);

				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;// grain of each point, accumulated
				keywordcnt++; // grain of each trajectory, not-accumulated
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;// grain of each point 
				keywordcnt++; // grain of each trajectory
			}
		}

		// status info. here
		keycntTrajQ.push_back(keywordcnt);


		//for (size_t j = 0; j < dataSizeP; j++) {
		//	// debug: this is wrong data structure!
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	//pnow = gpuAddrQSet;

	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());

	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));



	// pre-order for status info.
	size_t pmqnid = 0, pmqid = 0, pqid = 0;
	
	//size_t statussum = 0; // no need

	vector<int> stattableoffset; // store the pointer offset for stattableCPU for each round
	stattableoffset.push_back(0);
	vector<size_t> pmqnidtable, pmqidtable, pqidtable; // store the total pmqn pmq pq for each round

	//pmqnidtable.push_back(0); // starting id !
	//pmqidtable.push_back(0);
	//pqidtable.push_back(0);

	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {
			
			// we must pre-judge first! and first one must fit  gpuStat !!, hidden bug here -> e.g. 32*32 2GB have one: 2.07GB wrong access wrong! 
			size_t prejudgesum = (pmqnid  + keycntTrajP[i] * keycntTrajQ[j] +
			pmqid + stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i] +
			pqid + stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ);

			//statussum = pmqnid + pmqid + pqid; // not += , // not propriate!
			if (prejudgesum*4.0 / 1024 / 1024 / 1024 > gpuStat*1.0) {
				pmqnidtable.push_back(pmqnid);
				pmqidtable.push_back(pmqid);
				pqidtable.push_back(pqid);
				pmqnid = 0, pmqid = 0, pqid = 0; // starting a new round
				stattableoffset.push_back(i*dataSizeQ + j);
			}

			//size_t pmqnpre = pmqnid; // no need, same for afterwards
			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			// not symmetric Matrix processing  -> aborted first programming! to be easier
			//size_t pmqpre = pmqid;
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			///*
			// maybe this is not wrong, but may cause high coupling with kernel!
			//if (stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ) {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}
			//*/
			//size_t pqpre = pqid;
			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;

			// this is okay, can be better in v4
			stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];
			

			//size_t sumpre = statussum;
			//statussum = pmqnid + pmqid + pqid; // not += 

			//if (statussum*4.0 / 1024 / 1024 / 1024 > gpuStat*1.0) {
			//	
			//	pmqnidtable.push_back(pmqnpre);
			//	pmqidtable.push_back(pmqpre);
			//	pqidtable.push_back(pqpre);

			//	stattableoffset.push_back(i*dataSizeQ + j);

			//	pmqnid = pmqnid - pmqnpre;
			//	pmqid = pmqid - pmqpre;
			//	pqid = pqid - pqpre;

			//	statussum = (pmqnid - pmqnpre) + (pmqid - pmqpre) + (pqid - pqpre);
			//}

		}
	}

	// donnot forget this! final result for pmqnid pmqid pqid
	pmqnidtable.push_back(pmqnid);
	pmqidtable.push_back(pmqid);
	pqidtable.push_back(pqid);


	// stattable very important
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP*dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP*dataSizeQ);


	
	// have PROVED right,不足之处： statussizeonce 导致 block 数目不可控， 不平衡严重影响GPU性能!!  
	// -> CUSP! is  useful here! 最大限度提高 block数目? not that obvious,only one-gemm for a grid! not that good! -> whether take advatage of dynamic parallelism?
	for (size_t i = 0; i < stattableoffset.size(); i++) {

		pnow = gpuAddrStat;

		// stattable cpy: one block only once!! fetch i+1, be careful!
		int statussizeonce = (i == stattableoffset.size() - 1) ? dataSizeP * dataSizeQ - stattableoffset[i] : stattableoffset[i + 1] - stattableoffset[i];
//		printf("************ statussizeonce: %d \n************", statussizeonce);

		// no cpy here!
		//// stattable very important
		//CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU + stattableoffset[i], sizeof(StatInfoTable)* statussizeonce, cudaMemcpyHostToDevice, stream));
		////CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
		//stattableGPU = pnow;
		//pnow = (void*)((StatInfoTable*)pnow + statussizeonce);

		keypmqnMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pmqnidtable[i]);
		keypmqMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pmqidtable[i]);
		keypqMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pqidtable[i]);
	

		// debug: big int -> size_t
		//OutGPUMemNeeded(pmqnid, pmqid,pqid);
//		printf("***** size_t ***** %zu %zu %zu\n", pmqnidtable[i], pmqidtable[i], pqidtable[i]);
		//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
		//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
//		printf("***** total status size *****%f GB\n", (pmqnidtable[i] + pmqidtable[i] + pqidtable[i])*4.0 / 1024 / 1024 / 1024);

		// running kernel
		//CUDA_CALL(cudaDeviceSynchronize());
		//CUDA_CALL(cudaStreamSynchronize(stream));


		// ABOVE low cost! and cnted because of CUDA_CALL(cudaStreamSynchronize(stream));
		if (i == 0) {
			timer.stop();
			printf("CPU  processing time: %f s\n", timer.elapse()); // data pre-processing on CPU
			timer.start();
			CUDA_CALL(cudaEventRecord(kernel_start, stream));
		}

		// multi-kernel, but no need, because different block have no overlap between global memory! for keypmqnMatrixGPU keypmqMatrixGPU keypqMatrixGPU

		computeTSimpmqn << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);


		// debug: 非默认stream, this is necessary ? or not at all? ： NO , no overlap between global memory
		//CUDA_CALL(cudaStreamSynchronize(stream));

		computeTSimpmq << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);
		//CUDA_CALL(cudaStreamSynchronize(stream));


		computeTSimpq << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);
		//CUDA_CALL(cudaStreamSynchronize(stream));

		// above three can be merged!

		computeSimGPUV2 << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);

		// why must here?
		if (i == stattableoffset.size() - 1) {
			CUDA_CALL(cudaEventRecord(kernel_stop, stream));
		}

		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaStreamSynchronize(stream)); // be here is good,and necessary! really necessary to ensure correctness!

	}

	// out of FOR loop
	// here is wrong !! why
	//CUDA_CALL(cudaEventRecord(kernel_stop, stream));

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);


	// here has about 2s latency
	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}

	timer.stop();
	printf("resultback time: (calculated by timer)%f s\n", timer.elapse()); // very quick!! but nzc is not slow as well!!
	timer.start();

	// free CPU memory
	free(stattableCPU);

	// free GPU memory
	// debug: cudaFree doesn't erase anything!! it simply returns memory to a pool to be re-allocated
	// cudaMalloc doesn't guarantee the value of memory that has been allocated (to 0)
	// You need to Initialize memory (both global and shared) that your program uses, in order to have consistent results!!
	// The same is true for malloc and free, by the way
	CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrData));
	//CUDA_CALL(cudaFree(gpuAddrPSet));
	//CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse()); // cuda-managing time
																  //return;
}






void STSimilarityJoinCalcGPUV4(vector<STTrajectory> &trajSetP,
	vector<STTrajectory> &trajSetQ,
	vector<float> &result) {


	MyTimer timer;
	timer.start();

	CUDAwarmUp();
	/*
	void* gpuAddrTextualIndex = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrTextualValue = GPUMalloc((size_t)400 * 1024 * 1024); // MB
	void* gpuAddrSpacialLat = GPUMalloc((size_t)200 * 1024 * 1024);
	void* gpuAddrSpacialLon = GPUMalloc((size_t)200 * 1024 * 1024);
	*/

	// GPUmem-alloc
	// 需要手动free!!
	// CUDA_CALL
	int gpuData = 500;
	//int gpuQSet = 20;
	int gpuStat = 6; // please be BIG ! 
					 // here only for quick occupying GPU 
	void* gpuAddrData = GPUMalloc((size_t)gpuData * 1024 * 1024);
	//void* gpuAddrQSet = GPUMalloc((size_t)gpuQSet * 1024 * 1024);
	void* gpuAddrStat = GPUMalloc((size_t)gpuStat * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																		 //void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

	cudaStream_t stream;
	CUDA_CALL(cudaStreamCreate(&stream));




	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();



	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

	//vector<int> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
	// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
	//vector<int> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable

	vector<int> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<int> numWordPCPU, numWordQCPU; // keyword num in each point

	vector<int> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<float> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	
	// for status info.  -> can be merged into StatInfoTable* stattableCPU !!
	vector<int> keycntTrajP, keycntTrajQ; // keycnt each traj

	//vector<int> pointcntTrajP, pointcntTrajQ;


	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };


	// sparse matrix data
	vector<int> qkqcsrRowPtr;
	vector<int> qkqcsrColInd;
	vector<float> qkqcsrVal;

	vector<int> ppkcsrRowPtr;
	vector<int> ppkcsrColInd;
	vector<float> ppkcsrVal;



	TrajStatTable* trajPStattable = (TrajStatTable*)malloc(sizeof(TrajStatTable)*dataSizeP);
	if (trajPStattable == NULL) { printf("malloc failed!");  assert(0); };

	TrajStatTable* trajQStattable = (TrajStatTable*)malloc(sizeof(TrajStatTable)*dataSizeQ);
	if (trajQStattable == NULL) { printf("malloc failed!");  assert(0); };



	// for moving gpu pointer!  -------> maybe better to define: TrajStatTable! as above, for better code organization
	// can be aborted! as we have trajPStattable trajQStattable now!
	vector<size_t> qkqcsrRowPtrIdx;
	vector<size_t> qkqcsrColIndIdx;
	vector<size_t> qkqcsrValIdx;

	vector<size_t> ppkcsrRowPtrIdx;
	vector<size_t> ppkcsrColIndIdx;
	vector<size_t> ppkcsrValIdx;

	









	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	//void *keycntGPU;
	void *keypmqnMatrixGPU, *keypmqMatrixGPU, *keypqMatrixGPU;

	// for v4
	void *qkqcsrRowPtrGPU, *qkqcsrColIndGPU, *qkqcsrValueGPU;
	void *ppkcsrRowPtrGPU, *ppkcsrColIndGPU, *ppkcsrValueGPU;




	// P != Q
	// process P
	int latlonPId = 0, textPId = 0;

	size_t ppointcntaccumulated = 0, pkeywordcntaccumulated = 0;

	//// not here!
	//ppkcsrRowPtrIdx.push_back(ppointcntaccumulated);
	//ppkcsrColIndIdx.push_back(pkeywordcntaccumulated);
	//ppkcsrValIdx.push_back(pkeywordcntaccumulated);

	for (size_t i = 0; i < trajSetP.size(); i++) {
		

		
		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (int)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (int)trajSetP[i].traj_of_stpoint.size();

			stattableCPU[i*dataSizeQ + j].textIdxP = textPId;
		}


		int ppointcnt = 0;
		ppointcnt = trajSetP[i].traj_of_stpoint.size();
		


		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetP[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetP[i].traj_of_stpoint[j].lat;
			p.lon = trajSetP[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataPCPU.push_back(p.lat);
			lonDataPCPU.push_back(p.lon);
			numWordPCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.size());
			textIdxPCPU.push_back(textPId);
			latlonPId++;

			ppkcsrRowPtr.push_back(keywordcnt);

			for (size_t k = 0; k < trajSetP[i].traj_of_stpoint[j].keywords.size(); k++) {

				ppkcsrColInd.push_back(keywordcnt);
				ppkcsrVal.push_back(1.0);


				textDataPIndexCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetP[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textPId++;
				keywordcnt++;

				
			}
		}

		


		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetP[i].traj_of_stpoint.size() % 32; // bytes
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataPCPU.push_back(p.lat);
				lonDataPCPU.push_back(p.lon);
				numWordPCPU.push_back(-1);
				textIdxPCPU.push_back(-1);
				latlonPId++;
			}
		}


		// donnot forget this! and before the padding
		ppkcsrRowPtr.push_back(keywordcnt);

		// before the padding
		size_t nnz = keywordcnt;
	

		// debug: 逻辑错误！！ --> 自定义补齐 padding
		//remainder = 4 * textPId % 32; -> // 32 bytes对齐
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {

				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
				keywordcnt++;

			}
		}
		


		// size = |P set|(trajSetP.size())
		ppkcsrRowPtrIdx.push_back(ppointcntaccumulated);
		ppkcsrColIndIdx.push_back(pkeywordcntaccumulated);
		ppkcsrValIdx.push_back(pkeywordcntaccumulated);


		trajPStattable[i].csrRowPtrIdx = ppointcntaccumulated;
		trajPStattable[i].csrColIndIdx = pkeywordcntaccumulated;
		trajPStattable[i].csrValIdx = pkeywordcntaccumulated;
		trajPStattable[i].nnz = nnz;
		trajPStattable[i].row = ppointcnt;
		trajPStattable[i].col = keywordcnt;// including padding!


		// update the Idx correspondingly
		ppointcntaccumulated += (ppointcnt + 1); // csr -> we have to plus 1
		pkeywordcntaccumulated += nnz;


		


		keycntTrajP.push_back(keywordcnt);// keycnt including padding , if want not including padding, to do
		//// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].keycntP = keywordcnt;
		}
		// ---------> moved to, actually is the same effect!
		/*for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {
			...
			}
		}
		*/

		// but we have to accumulate that!
		// pkeywordcnt = keywordcnt
		// ppointcnt = ppointcnt





		//for (size_t j = 0; j < dataSizeQ; j++) {
		//	stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
		//}

		//pointcntTrajP.push_back()

	}

	CUDA_CALL(cudaEventRecord(memcpy_to_start, stream));
	// Copy data of P to GPU
	void *pnow = gpuAddrData;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxPCPU[0], sizeof(int)*textIdxPCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxPGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(int)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((int*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(int)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(float)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataPValueCPU.size());


	// process Q

	int qpointcntaccumulated = 0, qkeywordcntaccumulated = 0;

	int latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (int)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (int)trajSetQ[i].traj_of_stpoint.size();

			stattableCPU[j*dataSizeQ + i].textIdxQ = textQId;
		}


		int qpointcnt = 0;
		qpointcnt = trajSetQ[i].traj_of_stpoint.size();


		int keywordcnt = 0;
		for (size_t j = 0; j < trajSetQ[i].traj_of_stpoint.size(); j++) {
			Latlon p;
			p.lat = trajSetQ[i].traj_of_stpoint[j].lat;
			p.lon = trajSetQ[i].traj_of_stpoint[j].lon;
			//latlonDataPCPU.push_back(p);
			latDataQCPU.push_back(p.lat);
			lonDataQCPU.push_back(p.lon);
			numWordQCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.size());
			textIdxQCPU.push_back(textQId);

			latlonQId++; // grain of each point, accumulated

						 // need to define parameter to clean code!!
			for (size_t k = 0; k < trajSetQ[i].traj_of_stpoint[j].keywords.size(); k++) {

				//textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				//textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				
				qkqcsrRowPtr.push_back(keywordcnt);
				qkqcsrColInd.push_back(k);
				qkqcsrVal.push_back(1.0);


				// tiny bug!! mem error!!
				textDataQIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataQValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;// grain of each point, accumulated
				keywordcnt++; // grain of each trajectory, not-accumulated
			}
		}

		// for L2 cache(32 byte) alignment
		int remainder = 4 * trajSetQ[i].traj_of_stpoint.size() % 32;
		Latlon p; p.lat = 180.0; p.lon = 360.0;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				latDataQCPU.push_back(p.lat);
				lonDataQCPU.push_back(p.lon);
				numWordQCPU.push_back(-1);
				textIdxQCPU.push_back(-1);
				latlonQId++;
			}
		}


		// donnot forget this! and before the padding
		qkqcsrRowPtr.push_back(keywordcnt);
		
		int nnz = keywordcnt;

		// ATTENTION!!---> keywordcnt
		remainder = 4 * keywordcnt % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;// grain of each point 
				keywordcnt++; // grain of each trajectory
			}
		}

		// size = |Q set|(trajSetQ.size())
		qkqcsrRowPtrIdx.push_back(qpointcntaccumulated);
		qkqcsrColIndIdx.push_back(qkeywordcntaccumulated);
		qkqcsrValIdx.push_back(qkeywordcntaccumulated);


		trajQStattable[i].csrRowPtrIdx = qpointcntaccumulated; // just cpy from p-processing, not that good tho.
		trajQStattable[i].csrColIndIdx = qkeywordcntaccumulated;
		trajQStattable[i].csrValIdx = qkeywordcntaccumulated;
		trajQStattable[i].nnz = nnz;
		trajQStattable[i].row = keywordcnt;
		trajQStattable[i].col = qpointcnt;// including padding!


		// update the Idx correspondingly
		qpointcntaccumulated += (nnz + 1); // csr -> we have to plus 1
		qkeywordcntaccumulated += nnz;



		// status info. here
		keycntTrajQ.push_back(keywordcnt);
		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].keycntQ = keywordcnt;
		}


		//for (size_t j = 0; j < dataSizeP; j++) {
		//	// debug: this is wrong data structure!
		//	stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		//}
	}



	// Copy data of Q to GPU
	//pnow = gpuAddrQSet;

	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	// debug: wrong code!!! 符号错误造成逻辑错误 cpy原因
	//lonDataPGPU = pnow;
	lonDataQGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textIdxQCPU[0], sizeof(int)*textIdxQCPU.size(), cudaMemcpyHostToDevice, stream));
	textIdxQGPU = pnow;
	pnow = (void*)((int*)pnow + textIdxQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(int)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordQGPU = pnow;
	pnow = (void*)((int*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(int)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQIndexGPU = pnow;
	pnow = (void*)((int*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(float)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataQValueGPU = pnow;
	pnow = (void*)((float*)pnow + textDataQValueCPU.size());

	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));



	// maybe not that needed!
	//int* qkqcsrRowPtrGPU = new int[];
	//float **pqDenseGPU = new float *[dataSizeP]; // we use new, not malloc; similarly, donot forget delete[] to avoid leak
	//for (size_t i = 0; i < dataSizeP; i++) {
	//	pqDenseGPU[i] = new float[dataSizeQ];
	//}



	//pnow = gpuAddrStat;













	// pre-order for status info.
	size_t pmqnid = 0, pmqid = 0, pqid = 0;

	//size_t statussum = 0; // no need

	vector<int> stattableoffset; // store the pointer offset for stattableCPU for each round
	stattableoffset.push_back(0);
	vector<size_t> pmqnidtable, pmqidtable, pqidtable; // store the total pmqn pmq pq for each round

	//pmqnidtable.push_back(0); // starting id !
	//pmqidtable.push_back(0);
	//pqidtable.push_back(0);


	// updating stattableCPU.keywordpmqnMatrixId keywordpmqMatrixId keywordpqMatrixId
	for (size_t i = 0; i < trajSetP.size(); i++) {
		for (size_t j = 0; j < trajSetQ.size(); j++) {

			// we must pre-judge first! and first one must fit  gpuStat !!, hidden bug here -> e.g. 32*32 2GB have one: 2.07GB wrong access wrong! 
			size_t prejudgesum = (pmqnid + keycntTrajP[i] * keycntTrajQ[j] +
				pmqid + stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i] +
				pqid + stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ);

			//statussum = pmqnid + pmqid + pqid; // not += , // not propriate!
			if (prejudgesum*4.0 / 1024 / 1024 / 1024 > gpuStat*1.0) {

				pmqnidtable.push_back(pmqnid);
				pmqidtable.push_back(pmqid);
				pqidtable.push_back(pqid);

				pmqnid = 0, pmqid = 0, pqid = 0; // starting a new round
				stattableoffset.push_back(i*dataSizeQ + j);
			}

			//size_t pmqnpre = pmqnid; // no need, same for afterwards
			stattableCPU[i*dataSizeQ + j].keywordpmqnMatrixId = pmqnid;
			pmqnid += keycntTrajP[i] * keycntTrajQ[j];

			// not symmetric Matrix processing  -> aborted first programming! to be easier
			//size_t pmqpre = pmqid;
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];

			///*
			// maybe this is not wrong, but may cause high coupling with kernel!
			//if (stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ) {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			//}
			//else {
			//	pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			//}
			//*/
			//size_t pqpre = pqid;
			stattableCPU[i*dataSizeQ + j].keywordpqMatrixId = pqid;
			pqid += stattableCPU[i*dataSizeQ + j].pointNumP*stattableCPU[i*dataSizeQ + j].pointNumQ;


			////this is okay, no need for change, just change for check, ---------> moved to ABOVE
			// stattableCPU[i*dataSizeQ + j].keycntP = keycntTrajP[i];
			// stattableCPU[i*dataSizeQ + j].keycntQ = keycntTrajQ[j];


			//size_t sumpre = statussum;
			//statussum = pmqnid + pmqid + pqid; // not += 

			//if (statussum*4.0 / 1024 / 1024 / 1024 > gpuStat*1.0) {
			//	
			//	pmqnidtable.push_back(pmqnpre);
			//	pmqidtable.push_back(pmqpre);
			//	pqidtable.push_back(pqpre);

			//	stattableoffset.push_back(i*dataSizeQ + j);

			//	pmqnid = pmqnid - pmqnpre;
			//	pmqid = pmqid - pmqpre;
			//	pqid = pqid - pqpre;

			//	statussum = (pmqnid - pmqnpre) + (pmqid - pmqpre) + (pqid - pqpre);
			//}

		}
	}
	// donnot forget this! final result for pmqnid pmqid pqid
	pmqnidtable.push_back(pmqnid);
	pmqidtable.push_back(pmqid);
	pqidtable.push_back(pqid);


	// stattable very important
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP*dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP*dataSizeQ);



	// have PROVED right,不足之处： statussizeonce 导致 block 数目不可控， 不平衡严重影响GPU性能!!  
	// -> CUSP! is  useful here! 最大限度提高 block数目? not that obvious,only one-gemm for a grid! not that good! -> whether take advatage of dynamic parallelism?
	for (size_t i = 0; i < stattableoffset.size(); i++) {

		// each ROUND, we will move the pointer to gpuAddrStat !!
		pnow = gpuAddrStat;

		// stattable cpy: one block only once!! fetch i+1, be careful!
		int statussizeonce = (i == stattableoffset.size() - 1) ? dataSizeP * dataSizeQ - stattableoffset[i] : stattableoffset[i + 1] - stattableoffset[i];
		//		printf("************ statussizeonce: %d \n************", statussizeonce);

		// no cpy here!

		//// stattable very important
		//CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU + stattableoffset[i], sizeof(StatInfoTable)* statussizeonce, cudaMemcpyHostToDevice, stream));
		////CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
		//stattableGPU = pnow;
		//pnow = (void*)((StatInfoTable*)pnow + statussizeonce);

		keypmqnMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pmqnidtable[i]);
		keypmqMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pmqidtable[i]);
		keypqMatrixGPU = (float*)pnow;
		pnow = (void*)((float*)pnow + pqidtable[i]);


		// debug: big int -> size_t
		//OutGPUMemNeeded(pmqnid, pmqid,pqid);
		//		printf("***** size_t ***** %zu %zu %zu\n", pmqnidtable[i], pmqidtable[i], pqidtable[i]);
		//printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
		//printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
		//		printf("***** total status size *****%f GB\n", (pmqnidtable[i] + pmqidtable[i] + pqidtable[i])*4.0 / 1024 / 1024 / 1024);

		// running kernel
		//CUDA_CALL(cudaDeviceSynchronize());
		//CUDA_CALL(cudaStreamSynchronize(stream));


		// ABOVE low cost! and cnted because of CUDA_CALL(cudaStreamSynchronize(stream));
		if (i == 0) {
			timer.stop();
			printf("CPU  processing time: %f s\n", timer.elapse()); // data pre-processing on CPU
			timer.start();
			CUDA_CALL(cudaEventRecord(kernel_start, stream));
		}

		// multi-kernel, but no need, because different block have no overlap between global memory! for keypmqnMatrixGPU keypmqMatrixGPU keypqMatrixGPU

		computeTSimpmqn << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);


		// debug: 非默认stream, this is necessary ? or not at all? ： NO , no overlap between global memory
		//CUDA_CALL(cudaStreamSynchronize(stream));

		computeTSimpmq << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);
		//CUDA_CALL(cudaStreamSynchronize(stream));


		computeTSimpq << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);
		//CUDA_CALL(cudaStreamSynchronize(stream));

		// above three can be merged!

		computeSimGPUV2 << < statussizeonce, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
			(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
			(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
			(StatInfoTable*)stattableGPU + stattableoffset[i], (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU + stattableoffset[i]
			);

		// why must here?
		if (i == stattableoffset.size() - 1) {
			CUDA_CALL(cudaEventRecord(kernel_stop, stream));
		}

		//CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaStreamSynchronize(stream)); // be here is good,and necessary! really necessary to ensure correctness!

	}

	// out of FOR loop
	// here is wrong !! why
	//CUDA_CALL(cudaEventRecord(kernel_stop, stream));

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);


	// here has about 2s latency
	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}

	timer.stop();
	printf("resultback time: (calculated by timer)%f s\n", timer.elapse()); // very quick!! but nzc is not slow as well!!
	timer.start();

	// free CPU memory
	free(stattableCPU);

	// free GPU memory
	// debug: cudaFree doesn't erase anything!! it simply returns memory to a pool to be re-allocated
	// cudaMalloc doesn't guarantee the value of memory that has been allocated (to 0)
	// You need to Initialize memory (both global and shared) that your program uses, in order to have consistent results!!
	// The same is true for malloc and free, by the way
	CUDA_CALL(cudaFreeHost(SimResult));
	CUDA_CALL(cudaFree(gpuAddrData));
	//CUDA_CALL(cudaFree(gpuAddrPSet));
	//CUDA_CALL(cudaFree(gpuAddrQSet));
	CUDA_CALL(cudaFree(gpuAddrStat));

	// GPU stream management
	CUDA_CALL(cudaEventDestroy(memcpy_to_start));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());

	timer.stop();
	printf("CPU  after-processing time: %f s\n", timer.elapse()); // cuda-managing time
																  //return;
}











/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/