#include "AtomCoordinateCV_gpu_kernels.h"
#include "globalDefs.h"//from DANMD


__global__ void call_atom_coordinate_eval_cu(float4 *xs, int *idToIdxs, float *val, float4 *grad, int atomId, int index, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int id = idToIdxs[idx];
        float4 atomGrad = make_float4(0, 0, 0, 0);
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
void call_atom_coordinate_eval(float4 *xs, int *idToIdxs, float *val, float4 *grad, int atomId, int index, int nAtoms) {
    SAFECALL((call_atom_coordinate_eval_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(xs, idToIdxs, val, grad, atomId, index, nAtoms)));
}
