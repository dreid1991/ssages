#pragma once
#include "GPUArrayDeviceGlobal.h"
namespace SSAGES{
    class SnapshotGPUData {
        private:
            void processXs(){};//to be defined be engine for each field with nonconforming format
            void processVs(){};
            void processFs(){};
            void processIds(){};
            void processTypes(){};
        public:
            float4 *xs;
            float4 *vs;
            float4 *fs;
            int *ids;
            int *types;
            int nAtoms;


            GPUArrayDeviceGlobal<float> cvValues;
            bool xsFormatSame, vsFormatSame, fsFormatSame, idsFormatSame, typesFormatSame;

            bool xsIsProcessed, vsIsProcesses, fsIsProcessed, idsIsProcessed, typesIsProcessed;
            //only to be used by fields for which format is not as expected by package
            GPUArrayDeviceGlobal<float4> xsProcessed;
            GPUArrayDeviceGlobal<float4> vsProcessed;
            GPUArrayDeviceGlobal<float4> fsProcessed;
            GPUArrayDeviceGlobal<int> idsProcessed;
            GPUArrayDeviceGlobal<int> typesProcessed;
            resetIsProcessed() {
                xsIsProcessed = false;
                vsIsProcessed = false;
                fsIsProcessed = false;
                idsIsProcessed = false;
                typesIsProcessed = false;
            }
            float4 *getXs () {
                if (xsFormatSame) {
                    return xs;
                } else {
                    if (!xsIsProcessed) {
                        processXs();
                        xsIsProcessed = true;
                    }
                    return xsProcessed.ptr;
                }
            }
            float4 *getVs () {
                if (vsFormatSame) {
                    return vs;
                } else {
                    if (!vsIsProcessed) {
                        processVs();
                        vsIsProcessed = true;
                    }
                    return vsProcessed.ptr;
                }
            }
            float4 *getFs () {
                if (fsFormatSame) {
                    return fs;
                } else {
                    if (!fsIsProcessed) {
                        processFs();
                        fsIsProcessed = true;
                    }
                    return fsProcessed.ptr;
                }
            }
            int *getIds () {
                if (idsFormatSame) {
                    return ids;
                } else {
                    if (!idsIsProcessed) {
                        processIds();
                        idsIsProcessed = true;
                    }
                    return idsProcessed.ptr;
                }
            }
            int *getTypes () {
                if (typesFormatSame) {
                    return types;
                } else {
                    if (!typesIsProcessed) {
                        processTypes();
                        typesIsProcessed = true;
                    }
                    return types.ptr;
                }
            }



    };
}
