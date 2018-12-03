#include <assert.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConstDefine.h"
#include "gpukernel.h"


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
	__shared__ int pointIdP, pointNumP, keycntP, pointIdQ, pointNumQ, keycntQ;

	__shared__ int pmqnid, pmqid, pqid;
	__shared__ int textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

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
			if ((tmpflagi< pointNumP) && (tmpflagj< pointNumQ)) { // bound condition

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
	__shared__ int pointIdP, pointNumP, keycntP, pointIdQ, pointNumQ, keycntQ;

	__shared__ size_t pmqnid, pmqid, pqid;
	__shared__ int textPid, textQid;


	// seems not important!

	// merely for P-Q exchanging
	__shared__ float *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU, *textDataPValueGPU, *textDataQValueGPU;
	__shared__ int *textDataPIndexGPU, *textDataQIndexGPU, *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;

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
	
	// pmqn
	height = keycntP, width = keycntQ;
	for (size_t i = 0; i < keycntP; i += THREADROW) {
		int tmpflagi = i + tId % THREADROW;
		int pmindex, pmvalue;
		if (tmpflagi < keycntP) {
			pmindex = textDataPIndexGPU[textPid + tmpflagi];
			pmvalue = textDataPValueGPU[textPid + tmpflagi];
		}
		for (size_t j = 0; j < keycntQ; j+=THREADCOLUMN) {
			int tmpflagj = j + tId / THREADROW;
			int qnindex, qnvalue;
			if (tmpflagj < keycntQ) {
				qnindex = textDataQIndexGPU[textQid + tmpflagj];
				qnvalue = textDataQValueGPU[textQid + tmpflagj];
			}		
			// in such loop, can only index in this way!!
			// int -> size_t 兼容
			keypmqnGPU[pmqnid + tmpflagj*height + tmpflagi] = 0;
			if ((tmpflagi < keycntP) && (tmpflagj < keycntQ) && (pmindex== qnindex)) {
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
	__syncthreads();

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
		}
	}

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
		int textIdP, textIdQ, numWordP, numWordQ;
		if(tmpflagi < pointNumP){
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
				
				// way2: store way -> fetch way	 fetch from global memory!! 
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

void CUDAwarmUp() {
	CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	CUDA_CALL(cudaSetDevice(0)); // GPU-0
	if(DUALGPU) CUDA_CALL(cudaSetDevice(1)); // GPU-1
}

void* GPUMalloc(size_t byteNum) {
	void *addr;
	CUDA_CALL(cudaMalloc((void**)&addr, byteNum));
	return addr;
}


void STSimilarityJoinCalcGPU(vector<STTrajectory> &trajSetP,
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
	void* gpuAddrStat = GPUMalloc((size_t)2 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

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
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
		}

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
		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		}
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

			// not symmetric Matrix processing
			stattableCPU[i*dataSizeQ + j].keywordpmqMatrixId = pmqid;
			if(stattableCPU[i*dataSizeQ + j].pointNumP > stattableCPU[i*dataSizeQ + j].pointNumQ){
				pmqid += stattableCPU[i*dataSizeQ + j].pointNumQ*keycntTrajP[i];
			}
			else {
				pmqid += stattableCPU[i*dataSizeQ + j].pointNumP*keycntTrajQ[j];
			}
			
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
	printf("***** size_t ***** %zu %zu %zu\n", pmqnid, pmqid, pqid);
	printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
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

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);

	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}


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

	//return;
}


void STSimilarityJoinCalcGPUV2(vector<STTrajectory> &trajSetP,
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
	void* gpuAddrStat = GPUMalloc((size_t)2 * 1024 * 1024 * 1024); // 10GB need too much space for stats info.


																	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaEvent_t memcpy_to_start, kernel_start, kernel_stop;
	CUDA_CALL(cudaEventCreate(&memcpy_to_start));
	CUDA_CALL(cudaEventCreate(&kernel_start));
	CUDA_CALL(cudaEventCreate(&kernel_stop));

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

		keycntTrajP.push_back(keywordcnt);
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].textIdxP = keywordcnt;
		}

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
		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].textIdxQ = keywordcnt;
		}
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
	printf("***** avg. wordcnt ***** %f\n", sqrt(pmqnid*1.0 / (SIZE_DATA*SIZE_DATA)));
	printf("***** avg. pointcnt ***** %f\n", sqrt(pqid*1.0 / (SIZE_DATA*SIZE_DATA)));
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
	computeSimGPUV2 << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(int*)textDataPIndexGPU, (int*)textDataQIndexGPU, (float*)textDataPValueGPU, (float*)textDataQValueGPU,
		(int*)textIdxPGPU, (int*)textIdxQGPU, (int*)numWordPGPU, (int*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)keypmqnMatrixGPU, (float*)keypmqMatrixGPU, (float*)keypqMatrixGPU, (float*)SimResultGPU
		);
	CUDA_CALL(cudaEventRecord(kernel_stop, stream));
	

	CUDA_CALL(cudaStreamSynchronize(stream));
	//CUDA_CALL(cudaDeviceSynchronize());

	float memcpy_time = 0.0, kernel_time = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&memcpy_time, memcpy_to_start, kernel_start));
	CUDA_CALL(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));

	printf("memcpy time: %.5f s\n", memcpy_time / 1000.0);
	printf("kernel time: %.5f s\n", kernel_time / 1000.0);

	// rediculous
	for (size_t i = 0; i < dataSizeP*dataSizeQ; i++) {
		result.push_back(SimResult[i]);
	}


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