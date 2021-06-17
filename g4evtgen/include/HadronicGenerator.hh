//------------------------------------------------------------------------
// Class: HadronicGenerator
// Author: Alberto Ribon (CERN EP/SFT)
// Date: May 2020
//
// This class shows how to use Geant4 as a generator for simulating
// inelastic hadron-nuclear interactions.
// Some of the most used hadronic models are currently supported in
// this class:
// - the hadronic string models Fritiof (FTF) and Quark-Gluon-String (QGS)
//   coupled with Precompound/de-excitation
// - the intranuclear cascade models: Bertini (BERT), Binary Cascade (BIC),
//                                    and Liege (INCL)
// Combinations of these models are also available to "mimick" some of
// the most common Geant4 reference physics lists.
//
// The current version of this class does NOT support:
// -  hadron elastic interactions
// -  neutron capture and fission
// -  precise low-energy inelastic interactions of neutrons and
//    charged particles (i.e. ParticleHP)
// -  gamma/lepton-nuclear inelastic interactions
// -  inelastic nuclear interactions of generic-ions (i.e. projectile ions
//    heavier than deuterium, triton, He3 and alpha)
//
// This class does NOT use the Geant4 run-manager, and therefore should
// be usable in a multi-threaded application, with one instance of this
// class in each thread.
// 
// This class has been inspired by test30 (whose author is Vladimir
// Ivanchenko), with various simplifications and restricted to hadronic
// inelastic interactions.
//------------------------------------------------------------------------

#ifndef HadronicGenerator_h
#define HadronicGenerator_h 1

#include <iomanip>
#include "globals.hh"
#include "G4ios.hh"
#include "G4ThreeVector.hh"
#include <map>

class G4ParticleDefinition;
class G4VParticleChange;
class G4ParticleTable;
class G4Material;
class G4HadronicProcess;


class HadronicGenerator {
  // This class provides the functionality of a "hadronic generator"
  // for Geant4 final-state inelastic hadronic collisions.
  // Only a few of the available Geant4 final-state hadronic inelastic
  // "physics cases" are currently available in this class - but it can
  // be extended to other cases if needed.
  // It is important to notice that this class does NOT use the Geant4
  // run-manager, so it should work fine in a multi-threaded environment,
  // with a separate instance of this class in each thread.
  public:

    explicit HadronicGenerator( const G4String physicsCase = "FTFP_BERT_ATL" );
    // Currently supported final-state hadronic inelastic "physics cases":
    // -  Hadronic models :        BERT, BIC, IonBIC, INCL, FTFP, QGSP
    // -  "Physics-list proxies" : FTFP_BERT_ATL (default), FTFP_BERT,
    //                             QGSP_BERT, QGSP_BIC, FTFP_INCLXX
    //    (i.e. they are not real, complete physics lists - for instance
    //     they do not have: transportation, electromagnetic physics,
    //     hadron elastic scattering, neutron fission and capture, etc. -
    //     however, they cover all hadron types and all energies by
    //     combining different hadronic models, i.e. there are transitions
    //     between two hadronic models in well-defined energy intervals,
    //     e.g. "FTFP_BERT" has the transition between BERT and FTFP
    //     hadronic models; moreover, the transition intervals used in
    //     our "physics cases"might not be the same as in the corresponding
    //     physics lists).

    ~HadronicGenerator();

    G4bool IsPhysicsCaseSupported();
    // Returns "true" if the physicsCase is supported; "false" otherwise. 
  
    G4bool IsApplicable( const G4String &nameProjectile, const G4double projectileEnergy );
    G4bool IsApplicable( G4ParticleDefinition* projectileDefinition, const G4double projectileEnergy );
    // Returns "true" if the specified projectile (either by name or particle definition)
    // of given energy is applicable, "false" otherwise.

    G4VParticleChange* GenerateInteraction( const G4String &nameProjectile,
					    const G4double projectileEnergy,
					    const G4ThreeVector &projectileDirection ,
					    G4Material* targetMaterial );
    G4VParticleChange* GenerateInteraction( G4ParticleDefinition* projectileDefinition,
					    const G4double projectileEnergy,
					    const G4ThreeVector &projectileDirection ,
					    G4Material* targetMaterial );
    // This is the main method provided by the class:
    // in input it receives the projectile (either by name or particle definition),
    // its energy, its direction and the target material, and it returns one sampled
    // final-state of the inelastic hadron-nuclear collision as modelled by the
    // final-state hadronic inelastic "physics case" specified in the constructor.
    // If the required hadronic collision is not possible, then the method returns
    // immediately an empty "G4VParticleChange", i.e. without secondaries produced.
  
  private:

    G4String thePhysicsCase;
    G4bool thePhysicsCaseIsSupported;
    G4ParticleTable* thePartTable;
    std::map< G4ParticleDefinition*, G4HadronicProcess* > processMap;  
};


inline G4bool HadronicGenerator::IsPhysicsCaseSupported() {
  return thePhysicsCaseIsSupported;
}

#endif
