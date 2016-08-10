#include "AtomCoordinateCV_gpu.h"
#include "AtomCoordinateCV_gpu_kernels.h"
using namespace SSAGES;


void AtomCoordinateCV_gpu::Evaluate(const Snapshot& snapshot) 
{
    // Gradient and value. 
    float4 *xs = snapshot._gpd.xs;
    int *idToIdxs = snapshot._gpd.idToIdxs;
    const auto& pos = snapshot.GetPositions(); 
    const auto& ids = snapshot.GetAtomIDs();
    int nAtoms = snapshot._gpd.nAtoms;
    // Loop through atom positions.
    call_atom_coordinate_eval(xs, idToIdxs, _val, _grad.data(), _atomid, _index, nAtoms);

}

