
#ROOT = '/home/daniel/Documents/md_engine/core'
print ' YOU MUST SET VARIABLE \'ROOT\' IN RUN.PY TO THE FOLDER WHERE THE MD ENGINE IS LOCATED'
import sys
sys.path = sys.path + ['%s/build/python/build/lib.linux-x86_64-2.7' % ROOT ]
sys.path.append('%s/util_py' % ROOT)
from Sim import *
from LAMMPS_Reader import LAMMPS_Reader
import re
from math import *
def setupSimulation():
    print 'HELLO'
    state = State()
    state.units.setReal()
#state.bounds = Bounds(state, lo = Vector(0, -20, -20), hi = Vector(dx*ndim+20, dy*ndim+20, dz*ndim+20))
#state.bounds = Bounds(state, lo = Vector(0, -20, -20), hi = Vector(40, 40, 40))#Vector(dx*ndim+20, dy*ndim+20, dz*ndim+20))


    state.rCut = 9
    state.padding = 2.0
    state.periodicInterval = 7
    state.shoutEvery = 100

    state.dt = 1.0

    ljcut = FixLJCut(state, 'ljcut')
    bondHarm = FixBondHarmonic(state, 'bondharm')
    angleHarm = FixAngleHarmonic(state, 'angleHarm')
    dihedralOPLS = FixDihedralOPLS(state, 'opls')
    improperHarm = FixImproperHarmonic(state, 'imp')

    state.activateFix(ljcut)
    state.activateFix(bondHarm)
    state.activateFix(angleHarm)
    state.activateFix(dihedralOPLS)
    state.activateFix(improperHarm)


    #ewald = FixChargeEwald(state, "chargeFix", "all")
#ewald.setError(0.01, state.rCut, 3)

#ewald.setParameters(64, 64, 64, state.rCut, 3)
#state.activateFix(ewald)

    writeconfig = WriteConfig(state, fn='poly_out', writeEvery=100, format='xyz', handle='writer')
    state.activateWriteConfig(writeconfig)

#Reading in chlorobenzene
    reader = LAMMPS_Reader(state=state, nonbondFix = ljcut, atomTypePrefix = 'poly_', setBounds=True, bondFix = bondHarm, angleFix = angleHarm, dihedralFix = dihedralOPLS,improperFix=improperHarm)
    reader.read(dataFn = 'out.dat')

    state.setSpecialNeighborCoefs(0, 0, 0.5)

    InitializeAtoms.initTemp(state, 'all', 300)

    print 'GOING TO RETURN'
    return state


def runSimulation(state, numTurns):
#IF YOU INITIALIZE THE NOSE HOOVER THERMOSTAT IN setupSimulation, IT CRASHES.  NEED TO FIND OUT WHY
    fixNVT = FixNoseHoover(state, 'temp', 'all', [0, 1], [300, 300], 100)
    state.activateFix(fixNVT)
    integVerlet = IntegratorVerlet(state)
    integVerlet.run(numTurns)
    state.deactivateFix(fixNVT)

