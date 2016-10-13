#include "globalDefs.h"
#include "cutils_math.h"

__global__ void umbrella_eval(float4 *fs, float *val, float4 *grad, float center, float k, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float v = val[0];
        float4 f = fs[idx];
        float wOrig = f.w;
        float4 g = grad[idx];
        float mag = k * (v - center);
        g = g * mag;
        f -= g;
        //printf("g is %f %f %f %f\n", g.x, g.y, g.z, g.w);
        f.w = wOrig;
        fs[idx] = f;
    }
}

void call_umbrella_eval(float4 *fs, float *val, float4 *grad, float center, float k, int nAtoms) {
    umbrella_eval<<<NBLOCK(nAtoms), PERBLOCK>>>(fs, val, grad, center, k, nAtoms);

}
