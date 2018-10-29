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

__device__ inline float TSimGPU(){

}

__device__ void warpReduce(volatile float* sdata,int tid ){
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void computeSimGPU(float* latDataPGPU,float* latDataQGPU,float* lonDataPGPU,float* lonDataQGPU,
	uint32_t* textDataPIndexGPU, uint32_t* textDataQIndexGPU, uint32_t* textDataPValueGPU, uint32_t* textDataQValueGPU,
	uint32_t* textIdxPGPU, uint32_t* textIdxQGPU, uint32_t* numWordPGPU, uint32_t* numWordQGPU,
	StatInfoTable* stattableGPU,float* SimResultGPU
	) {
	int bId = blockIdx.x;
	int tId = threadIdx.x;
	
	__shared__ StatInfoTable task;
	__shared__ uint32_t pointIdP, pointNumP, pointIdQ, pointNumQ;

	__shared__ float tmpSim[THREADNUM];

	__shared__ float maxSimRow[MAXTRAJLEN];
	__shared__ float maxSimColumn[MAXTRAJLEN];

	__shared__ uint32_t tid_row;
	__shared__ uint32_t tid_column;

	//fetch task info
	if (tId == 0) {
		task = stattableGPU[bId];		
		pointIdP = task.latlonIdxP;
		pointNumP = task.pointNumP;
		pointIdQ = task.latlonIdxQ;
		pointNumQ = task.pointNumQ;
	}
	__syncthreads();


	// numP > numQ

	// initialize maxSimRow maxSimColumn
	/*
	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {
		maxSimRow[tId + i*THREADNUM] = 0;
		maxSimColumn[tId + i*THREADNUM] = 0;
	}
	*/

	maxSimRow[tId] = 0;
	maxSimColumn[tId] = 0;
	__syncthreads();

	float latP, latQ, lonP, lonQ;

	for (size_t i = 0; i < pointNumP; i += THREADROW) {
		latP = latDataPGPU[pointIdP + i + tId%THREADROW];
		lonP = lonDataPGPU[pointIdP + i + tId%THREADROW];
		printf("%f,%f \n", latP, lonP);

/*
		for (size_t j = 0; j < pointNumQ; j += THREADCOLUMN) {

			latQ = latDataQGPU[pointIdQ + j + tId / THREADROW];
			lonQ = lonDataQGPU[pointIdQ + j + tId / THREADROW];

			if ((i + tId % THREADROW < pointNumP) && (j + tId / THREADROW < pointNumQ)) { // bound condition
				float tsim = 0;
				float ssim = SSimGPU(latQ, lonQ, latP, lonP);
				tmpSim[tId] = ALPHA * ssim + (1 - ALPHA) * tsim;
			}
			else {
				tmpSim[tId] = -1;//技巧，省去下面的tID=0判断
			}
			
			// block 同步
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

			// naive process 

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
*/
	}


	/*
	// sum reduction
//	for (size_t i = 0; i < ((MAXTRAJLEN - 1) / THREADNUM) + 1; i++) {

	for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimRow[tId] += maxSimRow[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimRow, tId);

//	}

	for (size_t activethread = THREADNUM / 2; activethread > 32; activethread >>= 1) {
		if (tId < activethread) {
			maxSimColumn[tId] += maxSimColumn[tId + activethread];
			__syncthreads();
		}
	}

	if (tId < 32) warpReduce(maxSimColumn, tId);


	if (tId == 0) {
		SimResultGPU[bId] = maxSimRow[0] / pointNumP + maxSimColumn[0] / pointNumQ;
	}
	*/

	return;
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
	void* gpuAddrPSet = GPUMalloc((size_t)1000 * 1024 * 1024);
	void* gpuAddrQSet = GPUMalloc((size_t)1500 * 1024 * 1024);

	//void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	

	size_t dataSizeP = trajSetP.size(), dataSizeQ = trajSetQ.size();

	// build cpu data
	//vector<Latlon> latlonDataPCPU, latlonDataQCPU; // latlon array
	vector<float> latDataPCPU, latDataQCPU; // lat array
	vector<float> lonDataPCPU, lonDataQCPU; // lon array

	//vector<uint32_t> latlonIdxPCPU, latlonIdxQCPU; // way1: starting id of latlon data for each traj (each task / block) 
													// way2: void* gpuStatInfo = GPUMalloc((size_t)200 * 1024 * 1024); -> StatInfoTable
	//vector<uint32_t> latlonPointNumPCPU, latlonPointNumQCPU; // # of points in each traj -> StatInfoTable
	
	vector<uint32_t> textDataPIndexCPU, textDataQIndexCPU; // keyword Index array
	vector<uint32_t> textDataPValueCPU, textDataQValueCPU; // keyword Value array
	vector<uint32_t> textIdxPCPU, textIdxQCPU; // starting id of text data for each point
	vector<uint32_t> numWordPCPU, numWordQCPU; // keyword num in each point

	
	// 需要手动free!!
	StatInfoTable* stattableCPU = (StatInfoTable*)malloc(sizeof(StatInfoTable)* dataSizeP * dataSizeQ);
	if (stattableCPU == NULL) { printf("malloc failed!");  assert(0); };

	void *latDataPGPU, *latDataQGPU, *lonDataPGPU, *lonDataQGPU;
	void *textDataPIndexGPU, *textDataQIndexGPU, *textDataPValueGPU, *textDataQValueGPU;
	void *textIdxPGPU, *textIdxQGPU, *numWordPGPU, *numWordQGPU;
	void *stattableGPU;

	// P != Q
	// process P
	uint32_t latlonPId = 0, textPId = 0;
	for (size_t i = 0; i < trajSetP.size(); i++) {
		
		// 统计表
		for (size_t j = 0; j < dataSizeQ; j++) {
			stattableCPU[i*dataSizeQ + j].latlonIdxP = (uint32_t)latlonPId;
			stattableCPU[i*dataSizeQ + j].pointNumP = (uint32_t)trajSetP[i].traj_of_stpoint.size();
		}

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
		remainder = 4 * textPId % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataPIndexCPU.push_back(-1);
				textDataPValueCPU.push_back(-1);
				textPId++;
			}
		}
	}

	// Copy data of P to GPU
	void *pnow = gpuAddrPSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataPCPU[0], sizeof(float)*latDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataPGPU = pnow;
	pnow = (void*)((float*)pnow + latDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataPCPU[0], sizeof(float)*lonDataPCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(uint32_t)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordPCPU[0], sizeof(uint32_t)*numWordPCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + numWordPCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPIndexCPU[0], sizeof(uint32_t)*textDataPIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataPIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataPValueCPU[0], sizeof(uint32_t)*textDataPValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataPValueCPU.size());



	// process Q

	uint32_t latlonQId = 0, textQId = 0;
	for (size_t i = 0; i < trajSetQ.size(); i++) {

		for (size_t j = 0; j < dataSizeP; j++) {
			stattableCPU[j*dataSizeQ + i].latlonIdxQ = (uint32_t)latlonQId;
			stattableCPU[j*dataSizeQ + i].pointNumQ = (uint32_t)trajSetQ[i].traj_of_stpoint.size();
		}

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
				textDataPIndexCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordid);
				textDataPValueCPU.push_back(trajSetQ[i].traj_of_stpoint[j].keywords.at(k).keywordvalue);
				textQId++;
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
		remainder = 4 * textQId % 32;
		if (remainder) {
			for (size_t k = 0; k < (32 - remainder) / 4; k++) {
				textDataQIndexCPU.push_back(-1);
				textDataQValueCPU.push_back(-1);
				textQId++;
			}
		}
	}

	// Copy data of Q to GPU
	pnow = gpuAddrQSet;
	CUDA_CALL(cudaMemcpyAsync(pnow, &latDataQCPU[0], sizeof(float)*latDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	latDataQGPU = pnow;
	pnow = (void*)((float*)pnow + latDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &lonDataQCPU[0], sizeof(float)*lonDataQCPU.size(), cudaMemcpyHostToDevice, stream));
	lonDataPGPU = pnow;
	pnow = (void*)((float*)pnow + lonDataQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(uint32_t)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &numWordQCPU[0], sizeof(uint32_t)*numWordQCPU.size(), cudaMemcpyHostToDevice, stream));
	numWordPGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + numWordQCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQIndexCPU[0], sizeof(uint32_t)*textDataQIndexCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPIndexGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataQIndexCPU.size());
	CUDA_CALL(cudaMemcpyAsync(pnow, &textDataQValueCPU[0], sizeof(uint32_t)*textDataQValueCPU.size(), cudaMemcpyHostToDevice, stream));
	textDataPValueGPU = pnow;
	pnow = (void*)((uint32_t*)pnow + textDataQValueCPU.size());

	// stattable cpy: one block only once!!
	CUDA_CALL(cudaMemcpyAsync(pnow, stattableCPU, sizeof(StatInfoTable)* dataSizeP * dataSizeQ, cudaMemcpyHostToDevice, stream));
	//CUDA_CALL(cudaMemcpyAsync(pnow, &stattableCPU[0], sizeof(StatInfoTable)*stattableCPU.size(), cudaMemcpyHostToDevice, stream));
	stattableGPU = pnow;
	pnow = (void*)((StatInfoTable*)pnow + dataSizeP * dataSizeQ);

	
	// zero-copy 内存 
	// 需要手动free!!
	float *SimResult, *SimResultGPU;
	CUDA_CALL(cudaHostAlloc((void**)&SimResult, dataSizeP*dataSizeQ * sizeof(float), cudaHostAllocMapped));
	CUDA_CALL(cudaHostGetDevicePointer((void**)&SimResultGPU, SimResult, 0));


	// running kernel
	cudaDeviceSynchronize();

	computeSimGPU << < dataSizeP*dataSizeQ, THREADNUM, 0, stream >> > ((float*)latDataPGPU, (float*)latDataQGPU, (float*)lonDataPGPU, (float*)lonDataQGPU,
		(uint32_t*)textDataPIndexGPU, (uint32_t*)textDataQIndexGPU, (uint32_t*)textDataPValueGPU, (uint32_t*)textDataQValueGPU,
		(uint32_t*)textIdxPGPU, (uint32_t*)textIdxQGPU, (uint32_t*)numWordPGPU, (uint32_t*)numWordQGPU,
		(StatInfoTable*)stattableGPU, (float*)SimResultGPU
		);

	cudaDeviceSynchronize();

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

	// GPU stream management
	CUDA_CALL(cudaStreamDestroy(stream));
	CUDA_CALL(cudaDeviceReset());
	return;
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