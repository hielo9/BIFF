/*
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
//RMK: Need random numbers
#include <curand_kernel.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
//#include <boost/array.hpp>
//#include <boost/numeric/odeint/integrate/integrate.hpp>
//using namespace boost::numeric::odeint;
//#include "thrust/device_vector.h"

//using namespace std;

//change this to float if your device does not support double computation
//typedef double value_type;
//[ thrust_phase_chain_system
//change this to host_vector< ... > if you want to run on CPU
//typedef thrust::device_vector< value_type > state_type;
//typedef thrust::device_vector< size_t > index_vector_type;
//typedef thrust::host_vector< value_type > state_type;
//typedef thrust::host_vector< size_t > index_vector_type;


#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;
//const size_t nbins = params.numBins;

//#include <boost/array.hpp>
//#include <boost/numeric/odeint.hpp>
//using namespace boost::numeric::odeint;
//typedef boost::array< double , 20 > bin_type;
//typedef vector< value_type > bin_type;
//typedef thrust::device_vector< value_type > bin_type;

//__global__ uint dLessParticles;
//__global__ uint dNumParticles;

// simulation parameters in constant memory
//__constant__ CellParams cparams;

struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float3 posData = thrust::get<0>(t);
        volatile float3 momData = thrust::get<1>(t);
        volatile float massData = thrust::get<2>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 mom = make_float3(momData.x, momData.y, momData.z);
        float mass = make_float(mass);
        float delta;

        // Do we really need to check for the mass to be positive before we do this?
//        if (posData.w>=0.0) {

        mom.y = 0.0;

        pos += mom/mass * deltaTime;

        // RMK: Boundary handling

        if (pos.x > params.worldSize.x+params.worldOrigin.x)
        {
        	// This is way original nVidia stuff; I might need to think about the finite size of these again some day
            //pos.x = 1.0f - params.particleRadius;
        	delta = pos.x - (params.worldSize.x+params.worldOrigin.x);
        	pos.x = params.worldSize.x+params.worldOrigin.x - delta;
            //vel.x *= params.boundaryDamping;
        }

        //if (pos.x < -1.0f + params.particleRadius)
        //Assume R[0]==0 because we're doing cylindrical coordinates'
        if (pos.x < params.worldOrigin.x)
        {
            //pos.x = -1.0f + params.particleRadius;
        	//pos.x = params.R[0] - 2*pos.x;
        	delta = params.worldOrigin.x - pos.x;
        	pos.x = params.worldOrigin.x + delta; //params.R[0];
            //vel.x *= params.boundaryDamping;
        }
// RMK: Shouldn't have to worry about y-axis for cylindrical coordinates
/*
        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }
*/
        if (pos.z > params.worldSize.z+params.worldOrigin.z)
        {
            //pos.z = 1.0f - params.particleRadius;
            //pos.z = 2*pos.z - params.Z[params.gridSize.z];
        	delta = pos.z - (params.worldSize.z+params.worldOrigin.z);
        	pos.z = params.worldSize.z+params.worldOrigin.z - delta;
            //vel.z *= params.boundaryDamping;
        }
        if (pos.z > params.worldSize.z+params.worldOrigin.z) {
        	printf("What the heck, man?\n");
        }

        //if (pos.z < -1.0f + params.particleRadius)
        if (pos.z < params.worldOrigin.z)
        {
            //pos.z = -1.0f + params.particleRadius;
        	//pos.z = 2*pos.z - params.Z[0];
        	delta = params.worldOrigin.x - pos.z;
        	pos.z = params.worldOrigin.x + delta;
            //vel.z *= params.boundaryDamping;
        }

        if (pos.z < params.worldOrigin.z || pos.z>params.worldSize.z || pos.x < params.worldOrigin.x || pos.x>params.worldSize.x) {
        	printf("mom: %e %e %e\n",mom.x,mom.y,mom.z);
        	printf("pos: %f %f %f, mass: %f\n",pos.x,pos.y,pos.z,mass);
        	printf("disp:%e %e %e\n",mom.x/mass*deltaTime,mom.y/mass*deltaTime,mom.z/mass*deltaTime);
        	//printf("Pop it out and hope it's an isolated incident....\n");
        	//posData.w = -1.0;
        }

/*
        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }
*/
        }

        //pos.y = 0.0;
        // store new position and velocity
        //thrust::get<0>(t) = make_float4(pos, posData.w);
        //thrust::get<1>(t) = make_float4(vel, velData.w);
        thrust::get<0>(t) = pos;
        thrust::get<1>(t) = mom;
        thrust::get<2>(t) = mass;
    }
};

// No more velocity grids
//__device__ int3 calcVelGridPos(float3 p)

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    float eps = 0.00;
    //int i;

    gridPos.x = floor((p.x - params.worldOrigin.x+eps) / params.cellSize.x);
    gridPos.y = 0; //floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z+eps) / params.cellSize.z);
    if (gridPos.x>params.numCells.x-1) {
    	gridPos.x = params.numCells.x-1;
    	//printf("HEY NOW; can't do this: %d %d %15.12f %15.12f\n",gridPos.x,gridPos.z,p.x,p.z);
    	printf("HEY NOW; can't do this: %d %d %15.12f %15.12f %d %d %15.12f %15.12f %e\n",gridPos.x,gridPos.z,p.x,p.z,params.numCells.x,params.numCells.z,params.worldSize.x,params.worldSize.z);
    	//printf("based on %15.12f %15.12f\n",params.worldOrigin.z,params.worldSize.z);
    }
    if (gridPos.z>params.numCells.z-1) {
    	gridPos.z = params.numCells.z-1;
    	printf("HEY NOW; can't do this: %d %d %15.12f %15.12f %d %d %15.12f %15.12f %e\n",gridPos.x,gridPos.z,p.x,p.z,params.numCells.x,params.numCells.z,params.worldSize.x,params.worldSize.z);
    	//printf("based on %15.12f %15.12f\n",params.worldOrigin.z,params.worldSize.z);
    }
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
// BASICALLY, we're just turning a location in a 3-D array into a 1-D array, flattening it
__device__ uint calcNewGridHash(int3 gridPos)
{
	// RMK: Better hope that gridPos !> gridSize-1
    //gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    //gridPos.y = gridPos.y & (params.gridSize.y-1);
    //gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.numCells.y), params.numCells.x) + __umul24(gridPos.y, params.numCells.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcNewHashMaxD(uint   *gridParticleHash,  // output
                     uint   *gridParticleIndex, // output
                     float3 *pos,               // input: positions
                     uint    maxNumParticles)
{
	//printf("Made it this far (calcHashD)\n");
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint hash;

    if (index >= maxNumParticles) return;

    volatile float3 p = pos[index];

    // get address in grid
    //int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    //uint hash = calcGridHash(gridPos);
//    if (p.w >= 0.0) {
    	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    	//int3 gridPos = calcGridPosW(make_float3(p.x, p.y, p.z),p.w);
    	hash = calcNewGridHash(gridPos);
    	//if (hash==4225) {
    	//	printf("how did we get 4225? %f %f %f %d %d %d\n",p.x,p.y,p.z,gridPos.x,gridPos.y,gridPos.z);
    	//}
//    } else {
//    	hash = params.null_grid_value;
//    }
	//if (hash==4225) {
	//	printf("this is how we get 4225? %f %f %f %f\n",p.x,p.y,p.z,p.w);
	//}

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartMaxD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float3 *sortedPos,        // output: sorted positions
                                  float3 *sortedMom,        // output: sorted momenta
                                  float3 *sortedAngMom,     // output: sorted angular momenta
                                  float  *sortedMass,       // output: sorted mass
                                  float  *sortedScalar,     // output: sorted scalar (RMK)
                                  float  *sortedDiamBins,
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float3 *oldPos,           // input: sorted position array
                                  float3 *oldMom,           // input: sorted momentum array
                                  float3 *oldAngMom,        // input: sorted angular momentum array
                                  float  *oldMass,          // input: sorted mass array (RMK)
                                  float  *oldScalar,        // input: sorted scalar array (RMK)
                                  float  *oldDiamBins,
                                  uint    maxNumParticles,
								  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    //if (index==169032) {
    //    printf("inspect! %d %d %d %d %d\n",cellEnd[1],index,blockIdx.x,blockDim.x,threadIdx.x);
    //}

    // handle case when no. of particles not multiple of block size
    if (index < maxNumParticles)
//    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }
    //if (index==169032) {
    //    printf("inspect-! %d %d %d %d %d %d\n",cellEnd[1],index,blockIdx.x,blockDim.x,threadIdx.x,hash);
    //}

    __syncthreads();

//    if (index < maxNumParticles)
    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;

            //if (hash == 1  || hash == 2) {
            //	printf("dangit %d, %d %d %d\n",index,cellStart[hash],cellEnd[hash],hash);
            //	printf("dangit %d, %d %d %d\n",index,cellStart[sharedHash[threadIdx.x]],cellEnd[sharedHash[threadIdx.x]],sharedHash[threadIdx.x]);
            //}

        }

        //if (index == maxNumParticles - 1)
        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

    }
    //if (index < maxNumParticles)
    if (index < maxNumParticles)
    {

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float3 pos    = FETCH(oldPos,    sortedIndex);       // macro does either global read or texture fetch
        float3 mom    = FETCH(oldMom,    sortedIndex);       // see particles_kernel.cuh
        float3 angmom = FETCH(oldAngMom, sortedIndex);
        float  mass   = FETCH(oldMass,   sortedIndex);
        float  scalar = FETCH(oldScalar, sortedIndex);

        sortedPos   [index] = pos;
        sortedMom   [index] = mom;
        sortedAngMom[index] = angmom;
        sortedMass  [index] = mass;
        sortedScalar[index] = scalar;
        for (int b=0;b<params.numBins;b++) {
        	sortedDiamBins[index*params.numBins+b] = oldDiamBins[sortedIndex*params.numBins+b];
        }
//        if (hash==1 || sharedHash[threadIdx.x]==1) {
//            printf("inspect %d %d %d %d %d\n",cellEnd[1],index,blockIdx.x,blockDim.x,threadIdx.x);
//        }
        //if (index==169032) {
        //    printf("inspect!! %d %d %d %d %d %d %d\n",cellEnd[1],index,hash,sharedHash[threadIdx.x],blockIdx.x,blockDim.x,threadIdx.x);
        //}
    }
}

__device__
//float dInterp(float u1,float u2,float u3,float u4,float w1,float w2,float w3,float w4,float denom) {
float dInterp(float u1,float u2) {
	//printf("interp: %f, %f, %f, %f, %f, %f, %f, %f %f\n",u1,u2,u3,u4,w1,w2,w3,w4,denom);
	float mult = 1.0*(u1-u2);
	//return 0.2*(u1-u2);
	float thresh = 0.2;
	if (mult<-thresh) {
		mult = -thresh;
	} else if (mult>thresh) {
		mult = thresh;
	}
	return mult;
}

// RMK: interpolate quantities from cell data (using adjacent cells, probably
__device__
float interp(float u1,float u2,float u3,float u4,float w1,float w2,float w3,float w4,float denom) {
	//printf("interp: %f, %f, %f, %f, %f, %f, %f, %f %f\n",u1,u2,u3,u4,w1,w2,w3,w4,denom);
	return (w1*u1 +	w2*u2 +	w3*u3 +	w4*u4) / denom;
}

__device__
float random_gauss(float mean, float stddev, float U1, float U2) {
	// RMK: This should be altered to provide better seeding of random numbers...
	//curandState localState;
	// Use Box-Muller transform
	//float U1 = ((float)rand()+1)/((float)RAND_MAX+1);
	//float U2 = ((float)rand()+1)/((float)RAND_MAX+1);
	//float U1 = curand_uniform(&localState);
	//float U2 = curand_uniform(&localState);
	//printf("Us %f %f\n",U1,U2);

	//rand1 = sqrt(-2.0*log(U1))*cos(2.0*pi*U2);
	//printf("U1,2: %f %f\n",U1,U2);
	float rand2 = sqrt(-2.0*log(U1))*sin(2.0*CUDART_PI_F*U2);

	return rand2*stddev+mean;
}


__global__
void advectD(float3 *newVel,               // output: new velocity
             float3 *oldPos,               // input: sorted positions
           //   float4 *oldVel,               // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
              float *randomArray,
//			float *Uz,
//			float *Ur,
//			float *nut,
//			float *gradZNut,
//			float *gradRNut,
//			float *dUz,
//			float *dUr)
            )

{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

 	int i1, i2; //, ii;
	int k1, k2; //, jj;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    if (pos.x>params.cellSize.x*(0.5+gridPos.x)) {
		i1 = gridPos.x;  i2 = i1+1;
	} else {
		i2 = gridPos.x;  i1 = i2-1;
	}
	if (pos.z>params.cellSize.z*(0.5+gridPos.z)) {
		k1 = gridPos.z;  k2 = k1+1;
	} else {
		k2 = gridPos.z;  k1 = k2-1;
	}
	if (i1<0) {
		i1 = 0;  i2 = i1+1;
	} else if (i2>params.numCells.x-1) {
		i2 = params.numCells.x-1;  i1 = i2-1;
	}
	if (k1<0) {
		k1 = 0;  k2 = k1+1;
	} else if (k2>params.numCells.z-1) {
		k2 = params.numCells.z-1;  k1 = k2-1;
	}
	float dur1 = FETCH(dUr,i1*params.numCells.z+gridPos.z);
	float dur2 = FETCH(dUr,i2*params.numCells.z+gridPos.z);
	float duz1 = FETCH(dUz,gridPos.x*params.numCells.z+k1);
	float duz2 = FETCH(dUz,gridPos.x*params.numCells.z+k2);

	int pdfi1 = i1;  int pdfi2 = i2;
	int pdfk1 = k1;  int pdfk2 = k2;

	// get address in velocity grid
    int3 velGridPos = calcVelGridPos(pos);

    // instead we will use node values:

    i1 = velGridPos.x;  i2 = i1+1;
    k1 = velGridPos.z;  k2 = k1+1;

	float w1r, w2r;
	float w1z, w2z;
	float denomr, denomz;

	w1r = pos.x-params.R[i1];
	w2r = params.R[i2]-pos.x;
	denomr = params.R[i2]-params.R[i1];
	w1z = pos.z-params.Z[k1];
	w2z = params.Z[k2]-pos.z;
	denomz = params.Z[k2]-params.Z[k1];

	float rand1 = FETCH(randomArray,index*3);
    float rand2 = FETCH(randomArray,index*3+1);
    float rand3 = FETCH(randomArray,index*3+2);

    float3 vel = make_float3(0.0f);

    // write new velocity back to original unsorted location

    //newVel[index] = make_float4(vel, 0.0f);
    newVel[index] = vel;
}

__global__
void mixCurlD(float  *newScalar,            // output: new scalar
		      float  *oldScalar,            // input: sorted scalar
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;
}

__global__
void balanceCellsD(            // calculate and return velocities
		// float4 *newPos,
		// float *newScalar,
        float3 *sortedPos,         // accept the positions (sorted)
        float3 *sortedMom,         // accept the momenta (sorted)
        float3 *sortedAngMom,      // accept the angular momenta (sorted)
        float3 *sortedMass,        // accept the mass (sorted)
		float  *sortedScalar,
		float  *sortedDiamBins,
        uint   *gridParticleHash,
        uint   *gridParticleIndex,
        uint   *cellStart,
        uint   *cellEnd,
        int    *numParticles,
        int    *lessParticles)
	//int   &numParticles,
	//int   &lessParticles)
{
    uint gindex = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int3 ii;
	//ii.x = gindex/params.numCells.z;
	//ii.z = gindex%params.numCells.z;
	ii.x = gindex%params.numCells.x;
	ii.z = gindex/params.numCells.x;
	ii.y = 0;
	//uint gindex = calcNewGridHash(ii);
    uint i = ii.x;
    uint k = ii.z;
    //printf("i,k %2d %2d\n",i,k);
    int indexi;
    int3 iii;
    float3 fff;

    if (gindex >= params.numTotalCells) return;

    int partsInCell=-1;
    // INITIALIZE j TO ZERO?
	int j;   // this is tracking all particles above and beyond what we started with (partsInCell)
	int total = 0;

	if (cellStart[gindex]!=-1) {
		//printf("Yay\n");
		partsInCell = cellEnd[gindex]-cellStart[gindex];
		j=0;
	} //else { */
	//printf("pic: %d %d %d %d %d %d %d\n",i,k,gindex,cellStart[gindex],cellEnd[gindex],partsInCell,numParticles);

	// Need to check and make sure there are particles in the cell
	if (partsInCell<=0) {
		//printf("nay\n");
		printf("No particles in cell! %d %d\n",i,k);
		//cin.ignore();
		partsInCell = 0;
		indexi = atomicAdd(numParticles,1);
		cellStart[gindex] = indexi;
		cellEnd[gindex] = indexi;
		// RANDOMIZED sortedPos[indexi*4  ] = (params.m_R[i+1]-params.m_R[i]) * frand() + params.m_R[i];
		sortedPos[indexi].x  = params.cellSize.x * ((float)i+0.5);
		sortedPos[indexi].y  = 0.0; //p.y;
		// RANDOMIZED sortedPos[indexi*4+2] = (params.m_Z[k+1]-params.m_Z[k]) * frand() + params.m_Z[k]; //p.z;
		sortedPos[indexi].z  = params.cellSize.z * ((float)k+0.5); //p.z;
		sortedMom[indexi].x  = 0.0;
		sortedMom[indexi].y  = 0.0;
		sortedMom[indexi].z  = 0.0;
		sortedAngMom[indexi].x  = 0.0;
		sortedAngMom[indexi].y  = 0.0;
		sortedAngMom[indexi].z  = 0.0;
		sortedMass[indexi]   = 0.0; //volume(gindex)*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
		sortedScalar[indexi] = 0.0;
		// Not anymore now that bins are tracking number of particles
		//sortedDiamBins[indexi*params.numBins] = 1.0;
		sortedDiamBins[indexi*params.numBins] = 0.0;
		for (int b=1; b<params.numBins; b++) {
			sortedDiamBins[indexi*params.numBins+b] = 0.0;
		}

		//numParticles++;
		//partsInCell = 1;
		total++;
		j=1;

	}

	// check if there are enough particles in the cell; otherwise add
	while (partsInCell+j<params.minParticlesPerCell) {
		//printf("check1\n");
		indexi = atomicAdd(numParticles,1);
		//printf("in %d %d %d\n",i,k,indexi);
		// RANDOMIZED sortedPos[indexi*4  ] = (params.m_R[i+1]-params.m_R[i]) * frand() + params.m_R[i];
		//sortedPos[indexi].x = params.cellSize.x * ((float)i+0.5);
		sortedPos[indexi].y = 0.0; //p.y;
		// RANDOMIZED sortedPos[indexi*4+2] = (params.m_Z[k+1]-params.m_Z[k]) * frand() + params.m_Z[k]; //p.z;
		//sortedPos[indexi].z = params.cellSize.z * ((float)k+0.5); //p.z;
		//printf("check4\n");
		//sortedPos[indexi].w = 0.0; //volume(gindex)*m_params.celldensity/m_params.avgParticlesPerCell; //0.01;
		//sortedScalar[indexi] = 0.0;
		if (j<partsInCell) {
			sortedPos   [indexi].x = sortedPos   [(cellStart[gindex]+j)].x;
			sortedPos   [indexi].z = sortedPos   [(cellStart[gindex]+j)].z;
			sortedMass  [indexi]   = sortedMass  [(cellStart[gindex]+j)]/2.0;
			sortedMom   [indexi].x = sortedMom   [(cellStart[gindex]+j)].x;
			sortedMom   [indexi].y = sortedMom   [(cellStart[gindex]+j)].y;
			sortedMom   [indexi].z = sortedMom   [(cellStart[gindex]+j)].z;
			sortedAngMom[indexi].x = sortedAngMom[(cellStart[gindex]+j)].x;
			sortedAngMom[indexi].y = sortedAngMom[(cellStart[gindex]+j)].y;
			sortedAngMom[indexi].z = sortedAngMom[(cellStart[gindex]+j)].z;
			sortedScalar[indexi]   = sortedScalar[cellStart[gindex]+j];
			sortedMass  [(cellStart[gindex]+j)] = sortedMass   [indexi];
			for (int b=0; b<params.numBins; b++) {
				sortedDiamBins[indexi*params.numBins+b] = sortedDiamBins[(cellStart[gindex]+j)*params.numBins+b]/2.0;
				sortedDiamBins[(cellStart[gindex]+j)*params.numBins+b] = sortedDiamBins[indexi*params.numBins+b];
			}
			//sortedScalar[cellStart[gindex]+j] = sortedScalar[numParticles];
			//if (sortedPos   [indexi         *4+3]<0) {
			//printf("new cell mass: %f %d %d %d %d %f %d %d\n",sortedPos[indexi*4+3],j, gindex, cellStart[gindex], cellEnd[gindex],sortedPos[(cellStart[gindex]+j)*4+3],(cellStart[gindex]+j),m_hCellStart[gindex]);
			//}
		} else {
			sortedPos[indexi].x    = params.cellSize.x * ((float)i+0.5);
			sortedPos[indexi].z    = params.cellSize.z * ((float)k+0.5);
			sortedMom[indexi].x    = 0.0;
			sortedMom[indexi].y    = 0.0;
			sortedMom[indexi].z    = 0.0;
			sortedAngMom[indexi].x = 0.0;
			sortedAngMom[indexi].y = 0.0;
			sortedAngMom[indexi].z = 0.0;
			sortedMass  [indexi]   = 0.0;
			sortedScalar[indexi]   = 0.0;
			//sortedDiamBins[indexi*params.numBins] = 1.0;
			sortedDiamBins[indexi*params.numBins] = 0.0;
			for (int b=1; b<params.numBins; b++) {
				sortedDiamBins[indexi*params.numBins+b] = 0.0;
			}
		}
		//printf("check5\n");

		//numParticles += 1;
		//partsInCell += 1;
		j++;
		total++;
		//printf("check6\n");
	}

	// If there are too many particles, combine them by summing momenta and mass averaging the position and scalar
	// NOTE: For now, we're assuming we never end up with twice (or more) as many particles as we need
	while (partsInCell>params.maxParticlesPerCell) {
		//printf("check7\n");
		if ((sortedMass[(cellStart[gindex]+j)]+sortedMass[(cellEnd[gindex]-1)])>0.0) {
			sortedPos[cellStart[gindex]+j].x = (sortedPos[cellStart[gindex]+j].x*sortedMass[cellStart[gindex]+j] + sortedPos[cellEnd[gindex]-1].x*sortedMass[cellEnd[gindex]-1])
												/ (sortedMass[cellStart[gindex]+j]+sortedMass[(cellEnd[gindex]-1)]);
			sortedPos[cellStart[gindex]+j].z = (sortedPos[cellStart[gindex]+j].z*sortedMass[cellStart[gindex]+j] + sortedPos[cellEnd[gindex]-1].z*sortedMass[cellEnd[gindex]-1])
												/ (sortedMass[cellStart[gindex]+j]+sortedMass[(cellEnd[gindex]-1)]);
			sortedMom[cellStart[gindex]+j].x = sortedMom[cellStart[gindex]+j].x + sortedMom[cellEnd[gindex]-1].x;
			sortedMom[cellStart[gindex]+j].y = sortedMom[cellStart[gindex]+j].y + sortedMom[cellEnd[gindex]-1].y;
			sortedMom[cellStart[gindex]+j].z = sortedMom[cellStart[gindex]+j].z + sortedMom[cellEnd[gindex]-1].z;
			sortedAngMom[cellStart[gindex]+j].x = sortedAngMom[cellStart[gindex]+j].x + sortedAngMom[cellEnd[gindex]-1].x;
			sortedAngMom[cellStart[gindex]+j].y = sortedAngMom[cellStart[gindex]+j].y + sortedAngMom[cellEnd[gindex]-1].y;
			sortedAngMom[cellStart[gindex]+j].z = sortedAngMom[cellStart[gindex]+j].z + sortedAngMom[cellEnd[gindex]-1].z;
			sortedScalar[cellStart[gindex]+j] = (sortedScalar[cellStart[gindex]+j]*sortedMass[(cellStart[gindex]+j)] + sortedScalar[cellEnd[gindex]-1]*sortedMass[(cellEnd[gindex]-1)])
												/ (sortedMass[(cellStart[gindex]+j)]+sortedMass[(cellEnd[gindex]-1)]);
			for (int b=0; b<params.numBins; b++) {
				sortedDiamBins[(cellStart[gindex]+j)*params.numBins+b] += sortedDiamBins[(cellEnd[gindex]-1)*params.numBins+b];
			}
			sortedMass[(cellStart[gindex]+j)] += sortedMass[(cellEnd[gindex]-1)];
		} else {
			printf("Why the negative mass?\n");
			//cin.ignore();
			sortedScalar[cellStart[gindex]+j] = 0.0;
			//sortedDiamBins[(cellStart[gindex]+j)*params.numBins] = 1.0;
			sortedDiamBins[(cellStart[gindex]+j)*params.numBins] = 0.0;
			for (int b=1; b<params.numBins; b++) {
				sortedDiamBins[(cellStart[gindex]+j)*params.numBins+b] = 0.0;
			}
		}

		// Eliminate the extra particle from bookkeeping
		sortedMass[(cellEnd[gindex]-1)] = -1.0;
		cellEnd[gindex] -= 1;
		atomicAdd(lessParticles,1);
		partsInCell -= 1;
		j++;
		total--;
	}

	// don't know why, but this seems to give print bad numbers even if everything seems to be okay...
	//printf("balanceD returned with %d, %d, %d\n",partsInCell,lessParticles,numParticles);

}

__device__
// function for Antoine's equation; returns pressure in Pa, given temp in C
float AE(float temperature) {
	const float AE_A = 8.05573;
	const float AE_B = 1723.64;
	const float AE_C = 233.076;

	return powf(10.0,(AE_A-AE_B/(temperature+AE_C))) * 133.32239;
}
__device__
// function for Antoine's equation; returns pressure in mm Hg, given temp in C
float AE_mmHg(float temperature) {
	const float AE_A = 8.05573;
	const float AE_B = 1723.64;
	const float AE_C = 233.076;

	return powf(10.0,(AE_A-AE_B/(temperature+AE_C)));
}
__device__
// no idea if this is right
float AEinv(float pressure) {
	const float AE_A = 8.05573;
	const float AE_B = 1723.64;
	const float AE_C = 233.076;

	return -AE_B/(log10(pressure / 133.32239)-AE_A) - AE_C;
}
__device__
float scale1toC(float temperature) {
	return temperature*(params.tj-params.t0)+params.t0;
}
__device__
float scaleCto1(float temperature) {
	return (temperature-params.t0)/(params.tj-params.t0);
}
__device__
float scale1toPa(float humidity) {
	return humidity*(params.ppwj-params.ppw0)+params.ppw0;
}
__device__
float scalePato1(float humidity) {
	return (humidity-params.ppw0)/(params.ppwj-params.ppw0);
}
__device__
float w2p(float w) {
	return w*params.patm/(w+1);
}
__device__
float ss_calc(float scalar,float *radii,float *bins) {
	float delta_press = 0.0;
	float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	for(int b=0;b<params.numBins;b++) {
		delta_temp  += params.lambda_h2o/params.Cp * ((4.0/3.0*3.14159265359*powf(radii[b],3.0)-params.bVols[0]) * params.density_h2o/params.density_air) * bins[b]*1.0e6;
		delta_press += -w2p( (4.0/3.0*3.14159265359*powf(radii[b],3.0)-params.bVols[0]) * params.density_h2o/params.density_air) * bins[b]*1.0e6;
	}
	return (scale1toPa(scalar)+delta_press)/AE(scale1toC(scalar)+delta_temp);
}
__device__
float ss_calck(float scalar,float *radii,float *bins,float *k,float dt) {
	float delta_press = 0.0;
	float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	for(int b=0;b<params.numBins;b++) {
		delta_temp  += params.lambda_h2o/params.Cp * ((4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-params.bVols[0]) * params.density_h2o/params.density_air) * bins[b]*1.0e6;
		delta_press += -w2p( (4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-params.bVols[0]) * params.density_h2o/params.density_air) * bins[b]*1.0e6;
	}
	//printf("deltas: %f %f %f %f %f\n",scale1toC(scalar),delta_temp,AE(scale1toC(scalar)+delta_temp),scale1toPa(scalar),delta_press);
	return (scale1toPa(scalar)+delta_press)/AE(scale1toC(scalar)+delta_temp);
}
__device__
float ss_calc_enthalpy(float scalar,float *radii,float *bins,float particle_mass) {
	//float delta_press = 0.0;
	//float delta_temp = 0.0;
	//float conc_nuc = scalar*(params.jet_conc_nuc-params.conc_nuc_vol)+params.conc_nuc_vol;
	float X = (1.0-scalar)*params.x0 + scalar*params.xj;  // mass total water / total particle mass
	float hp = (1.0-scalar)*params.h0 + scalar*params.hj; // calculate enthalpy of particle
	float Xs = 0.0;
	for (int b=0;b<params.numBins;b++) {
		Xs += (4.0/3.0*3.14159265359*powf(radii[b],3.0)-params.bVols[0]) * params.density_h2o * bins[b]; //*1.0e6;
	}
	Xs = Xs/particle_mass;
	//Xs = 0.0;
	float T = (hp - (X-Xs)*params.hwe)/((1.0-X)*params.Cp+(X-Xs)*params.Cpw+Xs*params.Cw);
	float Ps = 760.0*(1.0 - (1.0-X)/((1.0-X)+(X-Xs)*params.Mair/params.Mwater));
	//printf("%f, %f, %f, %f, %f\n",Ps,T,Ps/AE_mmHg(T),X,Xs);
	for (int b=0;b<params.numBins;b++) {
		if (radii[b]>10.0e-6 && bins[b]>0.0) {
			printf("so TIRED.... %d  %e : %e X %e,%e,%e  %e,%e,%e,%e\n",b,radii[b],params.bDiams[b]/2e6,X,1.-X,Xs,(hp - (X)*params.hwe)/((1.0-X)*params.Cp+(X)*params.Cpw),T,Ps/AE_mmHg(T),bins[b]);
		}
	}
	return Ps/AE_mmHg(T);
}
__device__
float ss_calck_enthalpy(float scalar,float *radii,float *bins,float *k,float dt,float particle_mass) {
	float X = (1.0-scalar)*params.x0 + scalar*params.xj;  // mass total water / total particle mass
	float hp = (1.0-scalar)*params.h0 + scalar*params.hj; // calculate enthalpy of particle
	float Xs = 0.0;
	for (int b=0;b<params.numBins;b++) {
		Xs += (4.0/3.0*3.14159265359*powf(radii[b]+k[b]*dt,3.0)-params.bVols[0]) * params.density_h2o * bins[b]; //*1.0e6;
	}
	Xs = Xs/particle_mass;
	//Xs = 0.0;
	float T = (hp - (X-Xs)*params.hwe)/((1.0-X)*params.Cp+(X-Xs)*params.Cpw+Xs*params.Cw);
	float Ps = 760.0*(1.0 - (1.0-X)/((1.0-X)+(X-Xs)*params.Mair/params.Mwater));
	return Ps/AE_mmHg(T);
}
//__device__
//float drdt(float r,float ss) {
//	return params.gamma*(ss-1.0)/r;
//}
__device__
void drdt_bins(float scalar,float *bins,float dt,float *radii,float particle_mass) {
	//float k1[20];  float k2[20];  float k3[20];  float k4[20];
	//float gammaOverRadius[20];
	float k1[NUM_DIAM_BINS];
	float k2[NUM_DIAM_BINS];
	float k3[NUM_DIAM_BINS];
	float k4[NUM_DIAM_BINS];
	float gammaOverRadius[NUM_DIAM_BINS];
	//float* k1;  float* k2;  float* k3;  float* k4;
	//float* gammaOverRadius;
	//k1 = new float[params.numBins];  k2 = new float[params.numBins];  k3 = new float[params.numBins];  k4 = new float[params.numBins];
	//gammaOverRadius = new float[params.numBins];

	int bfirst;

	for (int b=0;b<params.numBins;b++) {
		gammaOverRadius[b] = params.gamma / radii[b];
	}

	float ss = ss_calc_enthalpy(scalar,radii,bins,particle_mass)-1.0;
	//float ss = ss_calc(scalar,radii,bins)-1.0;
	//printf("ss1: %f\n",ss);
	if (ss>0.0) { bfirst = 0; } else { bfirst = 1; k1[0] = 0.0; k2[0] = 0.0; k3[0] = 0.0; k4[0] = 0.0; }
	for (int b=bfirst;b<params.numBins;b++) {
		k1[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k1,dt/2.0,particle_mass)-1.0;
	//ss = ss_calck(scalar,radii,bins,k1,dt/2.0)-1.0;
	//printf("ss2: %f\n",ss);
	for (int b=bfirst;b<params.numBins;b++) {
		k2[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k2,dt/2.0,particle_mass)-1.0;
	//ss = ss_calck(scalar,radii,bins,k2,dt/2.0)-1.0;
	//printf("ss3: %f\n",ss);
	for (int b=bfirst;b<params.numBins;b++) {
		k3[b] = ss * gammaOverRadius[b];
	}
	ss = ss_calck_enthalpy(scalar,radii,bins,k3,dt    ,particle_mass)-1.0;
	//ss = ss_calck(scalar,radii,bins,k3,dt    )-1.0;
	//printf("ss4: %f\n",ss);
	for (int b=bfirst;b<params.numBins;b++) {
		k4[b] = ss * gammaOverRadius[b];
	}

	for (int b=0;b<params.numBins;b++) {
		//if (b==0 || b==1) {
		//	printf("%e %e %e %e %e %e %e\n",radii[b],k1[b],k2[b],k3[b],k4[b],(k1[b]+2*k2[b]+2*k3[b]+k4[b])*dt/6.0,gammaOverRadius[b]);
		//}
		radii[b] = max(radii[b] + (k1[b]+2*k2[b]+2*k3[b]+k4[b])*dt/6.0,params.minBin/2e6);
		if (radii[b]>20.0e-6 && bins[b]>0.0) {
			printf("so tired.... %d  %e : %e ks %e,%e,%e,%e\n",b,radii[b],params.bDiams[b]/2e6,k1[b],k2[b],k3[b],k4[b]);
		}
	}
	//delete [] k1;  delete [] k2;  delete [] k3;  delete [] k4;
	//delete [] gammaOverRadius;
}
__device__
void RK4(float scalar,float *bins,float dt,float particle_mass) {
	const int N = 10;
	float ddt = dt/N;
	//float radii[20];
	//float newBins[20];
	float radii[NUM_DIAM_BINS];
	float newBins[NUM_DIAM_BINS];
	//float* radii;
	//float* newBins;
    //newBins = new float[params.numBins];
    //radii = new float[params.numBins];
	const float threshold = 1e-4/NUM_DIAM_BINS;

    for (int b=0;b<params.numBins;b++) {
		radii[b] = params.bDiams[b]/2.0e6;
		newBins[b] = 0.0;
	}
	// Break up dt into smaller timesteps for integrating
	//printf("bins1: %f %f %f %f %f\n",bins[0],bins[1],bins[2],bins[3],bins[4]);
	for (int n=0;n<N;n++) {
		drdt_bins(scalar,bins,ddt,radii,particle_mass);
		//printf("bins: %f %f %f %f %f\n",bins[0],bins[1],bins[2],bins[3],bins[4]);
	}
	//printf("bins2: %f %f %f %f %f\n",bins[0],bins[1],bins[2],bins[3],bins[4]);

	float Vnew;
	int b1, b2;
	float y1, y2;
	for (int b=0;b<params.numBins;b++) {
		if (bins[b]<=threshold) {  continue;  }
		// check if bin b shrunk below the minimum diameter
		if (radii[b] <= params.minBin/1e6) {
			newBins[0] += bins[b];
			continue;
		}
		Vnew = 3.14159265359*powf(radii[b],3.0) *4./3.;  // in cubic meters
		b1 = floor((radii[b]*2.0e6) / params.binSize);  // calculate new drop diameter bin
		if (b1>=params.numBins-1) { // we've got bigger drops than our bins can handle
			//printf("bin emergency %f %f\n",radii[b]*2e6,params.binSize);
			b1 = params.numBins-2;  b2 = b1+1;
			y1 = 0.0;
			y2 = bins[b];
			//radii[b] = (b1 * params.binSize + params.binSize/2.0)/2e6;
		} else {
			b2 = b1+1;
			// Calculate the relative distribution between the two bins
			y2 = bins[b] * (Vnew-params.bVols[b1]) / (params.bVols[b2]-params.bVols[b1]);
			y1 = bins[b] - y2;
		}
		// Don't let just a very small amount be added
		if (fabs(y1)<threshold) {
			y1 = 0.0;
			y2 = bins[b];
		} else if (fabs(y2)<threshold) {
			y2 = 0.0;
			y1 = bins[b];
		} else if (y1<0 || y2<0) {
			printf("y12: %e %e %e %e %e %d %d %d %e %e %e\n",y1,y2,bins[b1],bins[b2],radii[b],b,b1,b2,(Vnew-params.bVols[b1]) / (params.bVols[b2]-params.bVols[b1]),params.bDiams[b1]/2e6,params.bDiams[b2]/2e6);
			//printf("r,d  %f, %f\n",radii[b]*2e6, params.binSize);
			//print 'Vs',Vnew,bvols[b1],bvols[b2]
			//print b,b1,b2,nbin
			//getchar();
		}
		newBins[b1] += y1;
		newBins[b2] += y2;
	}
	for (int b=0;b<params.numBins;b++) {
		bins[b] = newBins[b];
//		if (b>2 && bins[b]>0.0) {
//			printf("WHHHHHHYYYYYY %d, %e bins=%e; %e %e      %e %e\n",b,radii[b],bins[b],scalar,particle_mass,bins[0],bins[1]);
//		}
	}
	//printf("bins2: %f %f %f %f %f\n",bins[0],bins[1],bins[2],bins[3],bins[4]);
	//delete [] newBins;
	//delete [] radii;

}

__global__
void chemistryD(
		      float3 *oldPos,
		      float3 *oldMom,
		      float3 *oldAngMom,
		      float  *oldMass,
              float  *oldScalar,
              float  *oldDiamBins,
              float  *colors,
              uint    numParticles)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    float delta_temp = 0.0;
    float delta_press = 0.0;

    float y1,y2;
    float Vnew;
    int b1,b2;

    float newBins[NUM_DIAM_BINS];

    float scalar;
    float particle_mass;


    scalar = oldScalar[index];
    particle_mass = oldMass[index];

    //float3 pos = FETCH(oldPos, index);       // macro does either global read or texture fetch
    //particle_mass = pos.w;
    for (int b=0;b<params.numBins;b++) {
    	newBins[b] = oldDiamBins[index*params.numBins+b];
    }

    RK4(scalar,newBins,params.dt,particle_mass);

	float d10 = 0.0;
	for (int b=0;b<params.numBins;b++) {
		//TEMPORARY LINE:
		//newBins[b] = 0.0;
		oldDiamBins[index*params.numBins+b] = newBins[b];
		d10 += newBins[b]*params.bDiams[b];
	}

	if (index==1) {
		printf("d10 %f %f %f %f %f\n",d10,newBins[0],newBins[1],newBins[18],newBins[19]);
	}

	//colors[index] = ss;
	for (int b=0;b<params.numBins;b++) {
		newBins[b] = 0.0;
	}
	colors[index] = ss_calc_enthalpy(scalar,params.bDiams,newBins,particle_mass);
	//colors[index] = ss_calc(scalar,params.bDiams,newBins);
	//colors[index] = d10;

	//delete [] radii;
	//delete [] newBins;

}


#endif
