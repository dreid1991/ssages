#include "globalDefs.h"
#include "hook_kernels.h"
#define NTYPES 3
__global__ void copyToBuffer_cu(float4 *xs, float4 *vs, uint *ids, int *idToIdxs, char *buffer, uint *idsToCopy, int n) {
    int tid = GETIDX();
    if (tid < n * NTYPES) {
        int dataType = tid / n;
        int baseIdx = dataType * n;
        int nAtom = tid - baseIdx;
        int id = idsToCopy[nAtom];
        int idx = idToIdxs[id];
       // printf("tid %d dtype %d nAtom %d id %d idx %d\n", tid, dataType, nAtom, id, idx);
        if (dataType == 0) {
            float4 *bFloat = (float4 *) buffer;
            bFloat[nAtom] = xs[idx];
        } else if (dataType == 1) {
            float4 *bFloat = ((float4 *) buffer) + n;
            bFloat[nAtom] = vs[idx];
        } else if (dataType == 2) {
            uint *bInt = (uint *) (((float4 *) buffer) + 2*n);
            bInt[nAtom] = id; 
        }

//        int idx = 
    }
}

__global__ void unpackBuffer_cu(float4 *fs, int *idToIdxs, float4 *biasForces, uint *idsToCopy, int n) {
    int tid = GETIDX();
    if (tid < n) {
        int idx = idToIdxs[idsToCopy[tid]];
        float4 biasForce = biasForces[tid];
        float4 f = fs[idx];
        f.x += biasForce.x;
        f.y += biasForce.y;
        f.z += biasForce.z;
        fs[idx] = f;
    }
}


void copyToBuffer(float4 *xs, float4 *vs, uint *ids, int *idToIdxs, char *buffer, uint *idsToCopy, int n) {
    copyToBuffer_cu<<<NBLOCK(NTYPES * n), PERBLOCK>>>(xs, vs, ids, idToIdxs, buffer, idsToCopy, n);

}



void unpackBuffer(float4 *fs, int *idToIdxs, float4 *biasForces, uint *idsToCopy, int n) {
    unpackBuffer_cu<<<NBLOCK(n), PERBLOCK>>>(fs, idToIdxs, biasForces, idsToCopy, n);
}
