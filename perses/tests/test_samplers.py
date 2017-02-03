"""
Samplers for perses automated molecular design.

TODO:
* Refactor tests into a test class so that AlanineDipeptideSAMS test system only needs to be constructed once for a battery of tests.
* Generalize tests of samplers to iterate over all PersesTestSystem subclasses

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from nose.plugins.attrib import attr

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import logging
from functools import partial

import perses.tests.testsystems

import perses.rjmc.topology_proposal as topology_proposal
import perses.bias.bias_engine as bias_engine
import perses.rjmc.geometry as geometry
import perses.annihilation.ncmc_switching as ncmc_switching

################################################################################
# TEST MCMCSAMPLER
################################################################################

def test_valence():
    """
    Test valence-only test system.
    """
    # TODO: Test that proper statistics (equal sampling, zero free energy differences) are obtained.

    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem']
    niterations = 2 # number of iterations to run
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test ExpandedEnsembleSampler samplers.
        #for environment in testsystem.environments:
        #    exen_sampler = testsystem.exen_samplers[environment]
        #    f = partial(exen_sampler.run, niterations)
        #    f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
        #    yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            testsystem.exen_samplers[environment].pdbfile = open('sams.pdb', 'w') # DEBUG
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler for implicit hydration free energy
        from perses.samplers.samplers import MultiTargetDesign
        # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
        for environment in testsystem.environments:
            target_samplers = { testsystem.sams_samplers[environment] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
            designer = MultiTargetDesign(target_samplers)
            f = partial(designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

def test_testsystems_travis():
    """
    Test samplers on basic test systems for travis.
    """
    # These tests have to work for the first paper.
    testsystem_names = ['ValenceSmallMoleculeLibraryTestSystem', 'AlkanesTestSystem', 'FusedRingsTestSystem', 'T4LysozymeInhibitorsTestSystem']

    # If TESTSYSTEMS environment variable is specified, test those systems.
    if 'TESTSYSTEMS' in os.environ:
        testsystem_names = os.environ['TESTSYSTEMS'].split(' ')

    run_samplers(testsystem_names)

def test_tractable_system():
    """
    Test that the tractable testsystem samples molecules with the correct probabilities.
    """
    from perses.tests.testsystems import TractableValenceSmallMoleculeTestSystem
    import pymbar
    from perses import storage
    import itertools

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    temperature = 300.0*unit.kelvin
    outfile_name = "tractable_test.nc"
    n_iterations = 100
    environment = 'vacuum'
    modname = 'ExpandedEnsembleSampler'


    #initialize the system with no NCMC, 5 MCMC steps/iterations, and run 10000 iterations of the ExpandedEnsembleSampler
    tractable_test_system = TractableValenceSmallMoleculeTestSystem(storage_filename=outfile_name)
    tractable_test_system.exen_samplers[environment].ncmc_engine.nsteps = 0
    tractable_test_system.mcmc_samplers[environment].nsteps = 5
    tractable_test_system.exen_samplers[environment].run(niterations=n_iterations)

    #read in the storage file
    storage = storage.NetCDFStorage(outfile_name, mode='r')

    chemical_states = tractable_test_system.proposal_engines['vacuum'].chemical_state_list

    #Create structure for storing logP_accepts
    logPs = {chemical_state : [] for chemical_state in chemical_states}
    logP_dict = {chemical_state : logPs for chemical_state in chemical_states}

    for i in range(n_iterations):
        state_key = storage.get_object(environment, modname, 'state_key', i)
        proposed_state_key = storage.get_object(environment, modname, 'proposed_state_key', i)
        logP_accept = storage.get_object(environment, modname, 'logP_accept', i)
        logP_dict[state_key][proposed_state_key].append(logP_accept)
        print('%8d %s' % (i, state_key))

    #estimate the free energies using BAR:
    for pair in itertools.combinations(chemical_states,2):
        w_f = -np.array(logP_dict[pair[0]][pair[1]])
        w_r = -np.array(logP_dict[pair[1]][pair[0]])
        deltaF, dDeltaF = pymbar.BAR(w_f, w_r)
        analytical_difference = -1.0 * (tractable_test_system._log_normalizing_constants[pair[0]] - tractable_test_system._log_normalizing_constants[pair[1]])
        if np.abs(deltaF - analytical_difference) > 6*dDeltaF:
            msg = "The computed free energy did not match the analytical difference\n"
            msg += "Analytical difference: {analytical} \n".format(analytical=analytical_difference)
            msg += "Computed difference: {deltaF} +/- {dDeltaF}".format(deltaF=deltaF, dDeltaF=dDeltaF)
            print(pair)
            print(msg)
            raise Exception(msg)

@attr('advanced')
def test_testsystems_advanced():
    """
    Test samplers on advanced test systems.
    """
    testsystem_names = ['ImidazoleProtonationStateTestSystem', 'AblImatinibResistanceTestSystem', 'KinaseInhibitorsTestSystem', 'AlanineDipeptideTestSystem', 'AblAffinityTestSystem', 'T4LysozymeMutationTestSystem']
    run_samplers(testsystem_names)

def run_samplers(testsystem_names, niterations=5):
    """
    Run sampler stack on named test systems.

    Parameters
    ----------
    testsystem_names : list of str
        Names of test systems to run
    niterations : int, optional, default=5
        Number of iterations to run

    """
    for testsystem_name in testsystem_names:
        import perses.tests.testsystems
        testsystem_class = getattr(perses.tests.testsystems, testsystem_name)
        # Instantiate test system.
        testsystem = testsystem_class()
        # Test MCMCSampler samplers.
        for environment in testsystem.environments:
            mcmc_sampler = testsystem.mcmc_samplers[environment]
            f = partial(mcmc_sampler.run, niterations)
            f.description = "Testing MCMC sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test ExpandedEnsembleSampler samplers.
        for environment in testsystem.environments:
            exen_sampler = testsystem.exen_samplers[environment]
            f = partial(exen_sampler.run, niterations)
            f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler, if present.
        if hasattr(testsystem, 'designer') and (testsystem.designer is not None):
            f = partial(testsystem.designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f

def test_hybrid_scheme():
    """
    Test ncmc hybrid switching
    """
    from perses.tests.testsystems import AlanineDipeptideTestSystem
    niterations = 5 # number of iterations to run

    if 'TESTSYSTEMS' in os.environ:
        testsystem_names = os.environ['TESTSYSTEMS'].split(' ')
        if 'AlanineDipeptideTestSystem' not in testsystem_names:
            return

    # Instantiate test system.
    testsystem = AlanineDipeptideTestSystem()
    # Test MCMCSampler samplers.
    testsystem.environments = ['vacuum']
    # Test ExpandedEnsembleSampler samplers.
    from perses.samplers.samplers import ExpandedEnsembleSampler
    for environment in testsystem.environments:
        chemical_state_key = testsystem.proposal_engines[environment].compute_state_key(testsystem.topologies[environment])
        testsystem.exen_samplers[environment] = ExpandedEnsembleSampler(testsystem.mcmc_samplers[environment], testsystem.topologies[environment], chemical_state_key, testsystem.proposal_engines[environment], geometry.FFAllAngleGeometryEngine(metadata={}), scheme='geometry-ncmc-geometry', options={'nsteps':1})
        exen_sampler = testsystem.exen_samplers[environment]
        exen_sampler.verbose = True
        f = partial(exen_sampler.run, niterations)
        f.description = "Testing expanded ensemble sampler with AlanineDipeptideTestSystem '%s'" % environment
        yield f


if __name__=="__main__":
    test_tractable_system()
#    for t in test_samplers():
#        print(t.description)
#        t()
