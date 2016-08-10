#include "globalDefs.h"
#include "cutils_math.h"

__global__ void umbrella_eval(float4 *fs, float *val, float4 *grad, float center, float k, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float v = val[0];
        printf("from kernel, v is %f\n", v);
        float4 f = fs[idx];
        printf("idx is %d\n", idx);
        printf("fw is %f\n", f.w);
        float wOrig = f.w;
        float4 g = grad[idx];
        printf("g is %f\n", g.x);
        float mag = k * (v - center);
        printf("v is %f, center is %f, mag is %f\n", v, center, mag);
        g = g * mag;
        f -= g;
        f.w = wOrig;
        fs[idx] = f;
    }
}

void call_umbrella_eval(float4 *fs, float *val, float4 *grad, float center, float k, int nAtoms) {
    SAFECALL((umbrella_eval<<<NBLOCK(nAtoms), PERBLOCK>>>(fs, val, grad, center, k, nAtoms)));

}
