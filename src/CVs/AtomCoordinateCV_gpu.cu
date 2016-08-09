#include "AtomCoordinateCV_gpu.h"
#include "globalDefs.h"//from DANMD

__global__ void eval_cu(float4 *xs, int *idToIdxs, float *val, float4 *grad, int atomId, int index, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int id = idToIdxs[idx];
        float3 atomGrad = make_float4(0, 0, 0, 0);
        if (id == atomId) {
            if (index == 0) {
                atomGrad.x = 1;
                val[0] = xs[idx].x;
            } else if (index == 1) {
                atomGrad.y = 1;
                val[0] = xs[idx].y;
            } else if (index == 2) {
                atomGrad.z = 1;
                val[0] = xs[idx].z;
            }
        }
        grad[idx] = atomGrad;

    }
}

void AtomCoordinateCV_gpu::Evaluate(const Snapshot& snapshot) 
{
    // Gradient and value. 
    float4 *xs = snapshot._gpd.xs;
    int *idToIdxs = snapshot._gpd.idToIdxs;
    const auto& pos = snapshot.GetPositions(); 
    const auto& ids = snapshot.GetAtomIDs();
    int nAtoms = snapshot._gpd.nAtoms;
    // Loop through atom positions.
    eval_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(xs, idToIdxs, _val, _grad, _atomId, _index, nAtoms)

}


