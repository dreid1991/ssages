#include "AtomCoordinateCV_gpu.h"
#include "AtomCoordinateCV_gpu_kernels.h"
using namespace SSAGES;


void AtomCoordinateCV_gpu::Evaluate(const Snapshot& snapshot) 
{
    // Gradient and value. 
    float4 *xs = snapshot._gpd.xs;
    uint *ids = snapshot._gpd.ids;
    int nAtoms = snapshot._gpd.nAtoms;
    call_atom_coordinate_eval(xs, ids, _val.data(), _grad.data(), _atomid, _index, nAtoms);

}

