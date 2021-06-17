//------------------------------------------------------------------------
// Class: HadronicGenerator
// Author: Alberto Ribon (CERN EP/SFT)
// Date: May 2020
//------------------------------------------------------------------------
#include "HadronicGenerator.hh"
#include <iomanip>
#include "globals.hh"
#include "G4ios.hh"
#include "G4PhysicalConstants.hh"
#include "G4Material.hh"
#include "G4ProcessManager.hh"
#include "G4VParticleChange.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4DynamicParticle.hh"
#include "G4DecayPhysics.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4Step.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4StateManager.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"

#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4Proton.hh"
#include "G4Neutron.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
#include "G4Lambda.hh"
#include "G4SigmaPlus.hh"
#include "G4SigmaZero.hh"
#include "G4SigmaMinus.hh"
#include "G4XiMinus.hh"
#include "G4XiZero.hh"
#include "G4OmegaMinus.hh"
#include "G4AntiProton.hh"
#include "G4AntiNeutron.hh"
#include "G4AntiDeuteron.hh"
#include "G4AntiTriton.hh"
#include "G4AntiHe3.hh"
#include "G4AntiAlpha.hh"
#include "G4AntiLambda.hh"
#include "G4AntiSigmaPlus.hh"
#include "G4AntiSigmaZero.hh"
#include "G4AntiSigmaMinus.hh"
#include "G4AntiXiMinus.hh"
#include "G4AntiXiZero.hh"
#include "G4AntiOmegaMinus.hh"
#include "G4GenericIon.hh"

#include "G4HadronicProcess.hh"
#include "G4PionMinusInelasticProcess.hh"
#include "G4PionPlusInelasticProcess.hh"
#include "G4KaonMinusInelasticProcess.hh"
#include "G4KaonPlusInelasticProcess.hh"
#include "G4KaonZeroSInelasticProcess.hh"
#include "G4KaonZeroLInelasticProcess.hh"
#include "G4ProtonInelasticProcess.hh"
#include "G4NeutronInelasticProcess.hh"
#include "G4DeuteronInelasticProcess.hh"
#include "G4TritonInelasticProcess.hh"
#include "G4He3InelasticProcess.hh"
#include "G4AlphaInelasticProcess.hh"
#include "G4IonInelasticProcess.hh"
#include "G4LambdaInelasticProcess.hh"
#include "G4SigmaMinusInelasticProcess.hh"
#include "G4SigmaPlusInelasticProcess.hh"
#include "G4XiMinusInelasticProcess.hh"
#include "G4XiZeroInelasticProcess.hh"
#include "G4OmegaMinusInelasticProcess.hh"
#include "G4AntiProtonInelasticProcess.hh"
#include "G4AntiNeutronInelasticProcess.hh"
#include "G4AntiDeuteronInelasticProcess.hh"
#include "G4AntiTritonInelasticProcess.hh"
#include "G4AntiHe3InelasticProcess.hh"
#include "G4AntiAlphaInelasticProcess.hh"
#include "G4AntiLambdaInelasticProcess.hh"
#include "G4AntiSigmaMinusInelasticProcess.hh"
#include "G4AntiSigmaPlusInelasticProcess.hh"
#include "G4AntiXiMinusInelasticProcess.hh"
#include "G4AntiXiZeroInelasticProcess.hh"
#include "G4AntiOmegaMinusInelasticProcess.hh"

#include "G4CascadeInterface.hh"
#include "G4TheoFSGenerator.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4ExcitationHandler.hh"
#include "G4PreCompoundModel.hh"
#include "G4LundStringFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4FTFModel.hh"
#include "G4BinaryCascade.hh"
#include "G4BinaryLightIonReaction.hh"
#include "G4INCLXXInterface.hh"
#include "G4AblaInterface.hh"
#include "G4QuasiElasticChannel.hh"
#include "G4QGSMFragmentation.hh"
#include "G4QGSModel.hh"
#include "G4QGSParticipants.hh"

#include "G4VCrossSectionDataSet.hh"
#include "G4CrossSectionInelastic.hh"
#include "G4BGGNucleonInelasticXS.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4BGGPionInelasticXS.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4ChipsHyperonInelasticXS.hh"
#include "G4ComponentAntiNuclNuclearXS.hh"
#include "G4ComponentGGNuclNuclXsc.hh"


HadronicGenerator::HadronicGenerator( const G4String physicsCase ) :
  thePhysicsCase( physicsCase ), thePhysicsCaseIsSupported( false ), thePartTable( nullptr )
{
  // The constructor set-ups all the particles, models, cross sections and
  // hadronic inelastic processes.
  // This should be done only once for each application.
  // In the case of a multi-threaded application using this class,
  // the constructor should be invoked for each thread,
  // i.e. one instance of the class should be kept per thread.
  // The particles and processes that are created in this constructor
  // will then be used by the method GenerateInteraction at each interaction.
  // Notes:
  // - Neither the hadronic models nor the cross sections are used directly
  //   by the method GenerateInteraction, but they are associated to the
  //   hadronic processes and used by Geant4 to simulate the collision;
  // - Although the class generates only final states, but not free mean paths,
  //   inelastic hadron-nuclear cross sections are needed by Geant4 to sample
  //   the target nucleus from the target material.
  
  // Definition of particles
  G4GenericIon* gion = G4GenericIon::GenericIon();
  gion->SetProcessManager( new G4ProcessManager( gion ) );
  G4DecayPhysics* decays = new G4DecayPhysics;
  decays->ConstructParticle();  
  thePartTable = G4ParticleTable::GetParticleTable();
  thePartTable->SetReadiness();
  G4IonTable* ions = thePartTable->GetIonTable();
  ions->CreateAllIon();
  ions->CreateAllIsomer();

  // Build BERT model
  G4CascadeInterface* theBERTmodel = new G4CascadeInterface;

  // Build BIC model
  G4BinaryCascade* theBICmodel = new G4BinaryCascade;
  G4PreCompoundModel* thePreEquilib = new G4PreCompoundModel( new G4ExcitationHandler );
  theBICmodel->SetDeExcitation( thePreEquilib );

  // Build BinaryLightIon model
  G4PreCompoundModel* thePreEquilibBis = new G4PreCompoundModel( new G4ExcitationHandler );
  G4BinaryLightIonReaction* theIonBICmodel = new G4BinaryLightIonReaction( thePreEquilibBis );

  // Build the INCL model
  G4INCLXXInterface* theINCLmodel = new G4INCLXXInterface;
  const G4bool useAblaDeExcitation = false;  // By default INCL uses Preco: set "true" to use ABLA DeExcitation
  if ( theINCLmodel && useAblaDeExcitation ) {
    G4AblaInterface* theAblaInterface = new G4AblaInterface;
    theINCLmodel->SetDeExcitation( theAblaInterface );
  }
  
  // Build the FTFP model (FTF/Preco) : three instances with different energy intervals
  G4TheoFSGenerator* theFTFPmodel = new G4TheoFSGenerator;
  theFTFPmodel->SetMaxEnergy( 100.0*TeV );  // Needed to run above 25 GeV
  G4GeneratorPrecompoundInterface* theCascade1 = new G4GeneratorPrecompoundInterface;
  G4PreCompoundModel* thePreEquilib1 = new G4PreCompoundModel( new G4ExcitationHandler );
  theCascade1->SetDeExcitation( thePreEquilib1 );
  theFTFPmodel->SetTransport( theCascade1 );
  G4LundStringFragmentation* theLundFragmentation1 = new G4LundStringFragmentation;
  G4ExcitedStringDecay* theStringDecay1 = new G4ExcitedStringDecay( theLundFragmentation1 );
  G4FTFModel* theStringModel1 = new G4FTFModel;
  theStringModel1->SetFragmentationModel( theStringDecay1 );
  theFTFPmodel->SetHighEnergyGenerator( theStringModel1 );

  G4TheoFSGenerator* theFTFPmodel_constrained = new G4TheoFSGenerator;
  theFTFPmodel_constrained->SetMaxEnergy( 100.0*TeV );  // Needed to run above 25 GeV
  G4GeneratorPrecompoundInterface* theCascade2 = new G4GeneratorPrecompoundInterface;
  G4PreCompoundModel* thePreEquilib2 = new G4PreCompoundModel( new G4ExcitationHandler );
  theCascade2->SetDeExcitation( thePreEquilib2 );
  theFTFPmodel_constrained->SetTransport( theCascade2 );
  G4LundStringFragmentation* theLundFragmentation2 = new G4LundStringFragmentation;
  G4ExcitedStringDecay* theStringDecay2 = new G4ExcitedStringDecay( theLundFragmentation2 );
  G4FTFModel* theStringModel2 = new G4FTFModel;
  theStringModel2->SetFragmentationModel( theStringDecay2 );
  theFTFPmodel_constrained->SetHighEnergyGenerator( theStringModel2 );

  G4TheoFSGenerator* theFTFPmodel_halfConstrained = new G4TheoFSGenerator;
  theFTFPmodel_halfConstrained->SetMaxEnergy( 100.0*TeV );  // Needed to run above 25 GeV
  G4GeneratorPrecompoundInterface* theCascade3 = new G4GeneratorPrecompoundInterface;
  G4PreCompoundModel* thePreEquilib3 = new G4PreCompoundModel( new G4ExcitationHandler );
  theCascade3->SetDeExcitation( thePreEquilib3 );
  theFTFPmodel_halfConstrained->SetTransport( theCascade3 );
  G4LundStringFragmentation* theLundFragmentation3 = new G4LundStringFragmentation;
  G4ExcitedStringDecay* theStringDecay3 = new G4ExcitedStringDecay( theLundFragmentation3 );
  G4FTFModel* theStringModel3 = new G4FTFModel;
  theStringModel3->SetFragmentationModel( theStringDecay3 );
  theFTFPmodel_halfConstrained->SetHighEnergyGenerator( theStringModel3 );
 
  // Build the QGSP model (QGS/Preco)
  G4TheoFSGenerator* theQGSPmodel = new G4TheoFSGenerator;
  theQGSPmodel->SetMaxEnergy( 100.0*TeV );  // Needed to run above 25 GeV
  G4GeneratorPrecompoundInterface* theCascade4 = new G4GeneratorPrecompoundInterface;
  G4PreCompoundModel* thePreEquilib4 = new G4PreCompoundModel( new G4ExcitationHandler );
  theCascade4->SetDeExcitation( thePreEquilib4 );
  theQGSPmodel->SetTransport( theCascade4 );
  G4QGSMFragmentation* theQgsmFragmentation4 = new G4QGSMFragmentation;
  G4ExcitedStringDecay* theStringDecay4 = new G4ExcitedStringDecay( theQgsmFragmentation4 );
  G4VPartonStringModel* theStringModel4 = new G4QGSModel< G4QGSParticipants >;
  theStringModel4->SetFragmentationModel( theStringDecay4 );
  theQGSPmodel->SetHighEnergyGenerator( theStringModel4 );
  G4QuasiElasticChannel* theQuasiElastic = new G4QuasiElasticChannel;  // QGSP uses quasi-elastic
  theQGSPmodel->SetQuasiElasticChannel( theQuasiElastic );

  // Cross sections (needed by Geant4 to sample the target nucleus from the target material)
  G4VCrossSectionDataSet* thePionMinusXSdata = new G4BGGPionInelasticXS( G4PionMinus::Definition() );
  thePionMinusXSdata->BuildPhysicsTable( *(G4PionMinus::Definition()) );
  G4VCrossSectionDataSet* thePionPlusXSdata = new G4BGGPionInelasticXS( G4PionPlus::Definition() );
  thePionPlusXSdata->BuildPhysicsTable( *(G4PionPlus::Definition()) );
  G4VCrossSectionDataSet* theKaonMinusXSdata = new G4CrossSectionInelastic( new G4ComponentGGHadronNucleusXsc );
  theKaonMinusXSdata->BuildPhysicsTable( *(G4KaonPlus::Definition()) );
  G4VCrossSectionDataSet* theKaonPlusXSdata = theKaonMinusXSdata;
  theKaonPlusXSdata->BuildPhysicsTable( *(G4KaonPlus::Definition()) );
  G4VCrossSectionDataSet* theKaonZeroXSdata = theKaonMinusXSdata;
  theKaonZeroXSdata->BuildPhysicsTable( *(G4KaonZeroLong::Definition()) );
  G4VCrossSectionDataSet* theProtonXSdata = new G4BGGNucleonInelasticXS( G4Proton::Proton() );
  theProtonXSdata->BuildPhysicsTable( *(G4Proton::Definition()) );
  G4VCrossSectionDataSet* theNeutronXSdata = new G4NeutronInelasticXS;
  theNeutronXSdata->BuildPhysicsTable( *(G4Neutron::Definition()) );
  // For hyperon and anti-hyperons we can use either Chips or, for G4 >= 10.5, Glauber-Gribov cross sections
  //theHyperonsXSdata = new G4CrossSectionInelastic( new G4ComponentGGHadronNucleusXsc );
  G4VCrossSectionDataSet* theHyperonsXSdata = new G4ChipsHyperonInelasticXS;
  G4VCrossSectionDataSet* theAntibaryonsXSdata = new G4CrossSectionInelastic( new G4ComponentAntiNuclNuclearXS ); 
  G4VCrossSectionDataSet* theNuclNuclXSdata = new G4CrossSectionInelastic( new G4ComponentGGNuclNuclXsc );

  // Set up inelastic processes : store them in a map (with particle definition as key) for convenience
  typedef std::pair< G4ParticleDefinition*, G4HadronicProcess* > ProcessPair;
  G4HadronicProcess* thePionMinusInelasticProcess = new G4PionMinusInelasticProcess;
  processMap.insert( ProcessPair( G4PionMinus::Definition(), thePionMinusInelasticProcess ) );
  G4HadronicProcess* thePionPlusInelasticProcess = new G4PionPlusInelasticProcess;
  processMap.insert( ProcessPair( G4PionPlus::Definition(), thePionPlusInelasticProcess ) );
  G4HadronicProcess* theKaonMinusInelasticProcess = new G4KaonMinusInelasticProcess;
  processMap.insert( ProcessPair( G4KaonMinus::Definition(), theKaonMinusInelasticProcess ) );
  G4HadronicProcess* theKaonPlusInelasticProcess = new G4KaonPlusInelasticProcess;
  processMap.insert( ProcessPair( G4KaonPlus::Definition(), theKaonPlusInelasticProcess ) );
  G4HadronicProcess* theKaonZeroLInelasticProcess = new G4KaonZeroLInelasticProcess;
  processMap.insert( ProcessPair( G4KaonZeroLong::Definition(), theKaonZeroLInelasticProcess ) );
  G4HadronicProcess* theKaonZeroSInelasticProcess = new G4KaonZeroSInelasticProcess;
  processMap.insert( ProcessPair( G4KaonZeroShort::Definition(), theKaonZeroSInelasticProcess ) );
  G4HadronicProcess* theProtonInelasticProcess = new G4ProtonInelasticProcess;
  processMap.insert( ProcessPair( G4Proton::Definition(), theProtonInelasticProcess ) );
  G4HadronicProcess* theNeutronInelasticProcess = new G4NeutronInelasticProcess;
  processMap.insert( ProcessPair( G4Neutron::Definition(), theNeutronInelasticProcess ) );
  G4HadronicProcess* theDeuteronInelasticProcess = new G4DeuteronInelasticProcess;
  processMap.insert( ProcessPair( G4Deuteron::Definition(), theDeuteronInelasticProcess ) );
  G4HadronicProcess* theTritonInelasticProcess = new G4TritonInelasticProcess;
  processMap.insert( ProcessPair( G4Triton::Definition(), theTritonInelasticProcess ) );
  G4HadronicProcess* theHe3InelasticProcess = new G4He3InelasticProcess;
  processMap.insert( ProcessPair( G4He3::Definition(), theHe3InelasticProcess ) );
  G4HadronicProcess* theAlphaInelasticProcess = new G4AlphaInelasticProcess;
  processMap.insert( ProcessPair( G4Alpha::Definition(), theAlphaInelasticProcess ) );
  G4HadronicProcess* theIonInelasticProcess = new G4IonInelasticProcess;
  processMap.insert( ProcessPair( G4GenericIon::Definition(), theIonInelasticProcess ) );
  G4HadronicProcess* theLambdaInelasticProcess = new G4LambdaInelasticProcess;
  processMap.insert( ProcessPair( G4Lambda::Definition(), theLambdaInelasticProcess ) );
  G4HadronicProcess* theSigmaMinusInelasticProcess = new G4SigmaMinusInelasticProcess;
  processMap.insert( ProcessPair( G4SigmaMinus::Definition(), theSigmaMinusInelasticProcess ) );
  G4HadronicProcess* theSigmaPlusInelasticProcess = new G4SigmaPlusInelasticProcess;
  processMap.insert( ProcessPair( G4SigmaPlus::Definition(), theSigmaPlusInelasticProcess ) );
  G4HadronicProcess* theXiMinusInelasticProcess = new G4XiMinusInelasticProcess;
  processMap.insert( ProcessPair( G4XiMinus::Definition(), theXiMinusInelasticProcess ) );
  G4HadronicProcess* theXiZeroInelasticProcess = new G4XiZeroInelasticProcess;
  processMap.insert( ProcessPair( G4XiZero::Definition(), theXiZeroInelasticProcess ) );
  G4HadronicProcess* theOmegaMinusInelasticProcess = new G4OmegaMinusInelasticProcess;
  processMap.insert( ProcessPair( G4OmegaMinus::Definition(), theOmegaMinusInelasticProcess ) );
  G4HadronicProcess* theAntiProtonInelasticProcess = new G4AntiProtonInelasticProcess;
  processMap.insert( ProcessPair( G4AntiProton::Definition(), theAntiProtonInelasticProcess ) );
  G4HadronicProcess* theAntiNeutronInelasticProcess = new G4AntiNeutronInelasticProcess;
  processMap.insert( ProcessPair( G4AntiNeutron::Definition(), theAntiNeutronInelasticProcess ) );
  G4HadronicProcess* theAntiDeuteronInelasticProcess = new G4AntiDeuteronInelasticProcess;
  processMap.insert( ProcessPair( G4AntiDeuteron::Definition(), theAntiDeuteronInelasticProcess ) );
  G4HadronicProcess* theAntiTritonInelasticProcess = new G4AntiTritonInelasticProcess;
  processMap.insert( ProcessPair( G4AntiTriton::Definition(), theAntiTritonInelasticProcess ) );
  G4HadronicProcess* theAntiHe3InelasticProcess = new G4AntiHe3InelasticProcess;
  processMap.insert( ProcessPair( G4AntiHe3::Definition(), theAntiHe3InelasticProcess ) );
  G4HadronicProcess* theAntiAlphaInelasticProcess = new G4AntiAlphaInelasticProcess;
  processMap.insert( ProcessPair( G4AntiAlpha::Definition(), theAntiAlphaInelasticProcess ) );
  G4HadronicProcess* theAntiLambdaInelasticProcess = new G4AntiLambdaInelasticProcess;
  processMap.insert( ProcessPair( G4AntiLambda::Definition(), theAntiLambdaInelasticProcess ) );
  G4HadronicProcess* theAntiSigmaMinusInelasticProcess = new G4AntiSigmaMinusInelasticProcess;
  processMap.insert( ProcessPair( G4AntiSigmaMinus::Definition(), theAntiSigmaMinusInelasticProcess ) );
  G4HadronicProcess* theAntiSigmaPlusInelasticProcess = new G4AntiSigmaPlusInelasticProcess;
  processMap.insert( ProcessPair( G4AntiSigmaPlus::Definition(), theAntiSigmaPlusInelasticProcess ) );
  G4HadronicProcess* theAntiXiMinusInelasticProcess = new G4AntiXiMinusInelasticProcess;
  processMap.insert( ProcessPair( G4AntiXiMinus::Definition(), theAntiXiMinusInelasticProcess ) );
  G4HadronicProcess* theAntiXiZeroInelasticProcess = new G4AntiXiZeroInelasticProcess;
  processMap.insert( ProcessPair( G4AntiXiZero::Definition(), theAntiXiZeroInelasticProcess ) );
  G4HadronicProcess* theAntiOmegaMinusInelasticProcess = new G4AntiOmegaMinusInelasticProcess;
  processMap.insert( ProcessPair( G4AntiOmegaMinus::Definition(), theAntiOmegaMinusInelasticProcess ) );

  // Add the cross sections to the corresponding hadronic processes
  thePionMinusInelasticProcess->AddDataSet( thePionMinusXSdata );
  thePionPlusInelasticProcess->AddDataSet( thePionPlusXSdata );
  theKaonMinusInelasticProcess->AddDataSet( theKaonMinusXSdata );
  theKaonPlusInelasticProcess->AddDataSet( theKaonPlusXSdata );
  theKaonZeroLInelasticProcess->AddDataSet( theKaonZeroXSdata );
  theKaonZeroSInelasticProcess->AddDataSet( theKaonZeroXSdata );
  theProtonInelasticProcess->AddDataSet( theProtonXSdata );
  theNeutronInelasticProcess->AddDataSet( theNeutronXSdata );
  theDeuteronInelasticProcess->AddDataSet( theNuclNuclXSdata );
  theTritonInelasticProcess->AddDataSet( theNuclNuclXSdata );
  theHe3InelasticProcess->AddDataSet( theNuclNuclXSdata );
  theAlphaInelasticProcess->AddDataSet( theNuclNuclXSdata );
  theIonInelasticProcess->AddDataSet( theNuclNuclXSdata );
  theLambdaInelasticProcess->AddDataSet( theHyperonsXSdata );
  theSigmaMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theSigmaPlusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theXiMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theXiZeroInelasticProcess->AddDataSet( theHyperonsXSdata );
  theOmegaMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiProtonInelasticProcess->AddDataSet( theAntibaryonsXSdata );
  theAntiNeutronInelasticProcess->AddDataSet( theAntibaryonsXSdata );
  theAntiDeuteronInelasticProcess->AddDataSet( theAntibaryonsXSdata );
  theAntiTritonInelasticProcess->AddDataSet( theAntibaryonsXSdata );
  theAntiHe3InelasticProcess->AddDataSet( theAntibaryonsXSdata );
  theAntiAlphaInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiLambdaInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiSigmaMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiSigmaPlusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiXiMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiXiZeroInelasticProcess->AddDataSet( theHyperonsXSdata );
  theAntiOmegaMinusInelasticProcess->AddDataSet( theHyperonsXSdata );
  
  // Register the proper hadronic model(s) to the corresponding hadronic processes.
  // Note: hadronic models ("BERT", "BIC", "IonBIC", "INCL", "FTFP", "QGSP") are
  //       used for the hadrons and energies they are applicable
  //       (exception for INCL, which in recent versions of Geant4 can handle
  //        more hadron types and higher energies than considered here).
  //       For "physics-list proxies" ("FTFP_BERT", "FTFP_BERT_ATL", "QGSP_BERT",
  //       "QGSP_BIC", "FTFP_INCLXX"), all hadron types and all energies are covered
  //       by combining different hadronic models - similarly (but not identically)
  //       to the corresponding physics lists.
  if ( thePhysicsCase == "BIC" || thePhysicsCase == "QGSP_BIC" ) {
    // The BIC model is applicable to nucleons and pions,
    // whereas in the physics list QGSP_BIC it is used only for nucleons
    thePhysicsCaseIsSupported = true;
    theProtonInelasticProcess->RegisterMe( theBICmodel );
    theNeutronInelasticProcess->RegisterMe( theBICmodel );
    if ( thePhysicsCase == "BIC" ) {
      thePionMinusInelasticProcess->RegisterMe( theBICmodel );
      thePionPlusInelasticProcess->RegisterMe( theBICmodel );
    } else {
      thePionMinusInelasticProcess->RegisterMe( theBERTmodel );
      thePionPlusInelasticProcess->RegisterMe( theBERTmodel );
    }
  } else if ( thePhysicsCase == "INCL" || thePhysicsCase == "FTFP_INCLXX" ) {
    // We consider here for simplicity only nucleons and pions
    // (although recent versions of INCL can handle others particles as well)
    thePhysicsCaseIsSupported = true;
    thePionMinusInelasticProcess->RegisterMe( theINCLmodel );    
    thePionPlusInelasticProcess->RegisterMe( theINCLmodel );    
    theProtonInelasticProcess->RegisterMe( theINCLmodel );    
    theNeutronInelasticProcess->RegisterMe( theINCLmodel );    
  }
  if ( thePhysicsCase == "IonBIC"    || thePhysicsCase == "FTFP_BERT_ATL" ||
       thePhysicsCase == "FTFP_BERT" || thePhysicsCase == "FTFP_INCLXX"   ||
       thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC"  ) {
    // The Binary Light Ion model is used for light ions in all physics lists
    thePhysicsCaseIsSupported = true;
    theDeuteronInelasticProcess->RegisterMe( theIonBICmodel );    
    theTritonInelasticProcess->RegisterMe( theIonBICmodel );    
    theHe3InelasticProcess->RegisterMe( theIonBICmodel );    
    theAlphaInelasticProcess->RegisterMe( theIonBICmodel );  
  }
  if ( thePhysicsCase == "QGSP" || thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC" ) {
    // Although the QGSP model can handle also hyperons and anti-baryons,
    // in the physics lists it is used only for pions, kaons and nucleons
    thePhysicsCaseIsSupported = true;
    thePionMinusInelasticProcess->RegisterMe( theQGSPmodel );
    thePionPlusInelasticProcess->RegisterMe( theQGSPmodel );
    theKaonMinusInelasticProcess->RegisterMe( theQGSPmodel );
    theKaonPlusInelasticProcess->RegisterMe( theQGSPmodel );
    theKaonZeroLInelasticProcess->RegisterMe( theQGSPmodel );
    theKaonZeroSInelasticProcess->RegisterMe( theQGSPmodel );
    theProtonInelasticProcess->RegisterMe( theQGSPmodel );
    theNeutronInelasticProcess->RegisterMe( theQGSPmodel );
    if ( thePhysicsCase == "QGSP" ) {
      theLambdaInelasticProcess->RegisterMe( theQGSPmodel );
      theSigmaMinusInelasticProcess->RegisterMe( theQGSPmodel );
      theSigmaPlusInelasticProcess->RegisterMe( theQGSPmodel );
      theXiMinusInelasticProcess->RegisterMe( theQGSPmodel );
      theXiZeroInelasticProcess->RegisterMe( theQGSPmodel );
      theOmegaMinusInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiProtonInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiNeutronInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiDeuteronInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiTritonInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiHe3InelasticProcess->RegisterMe( theQGSPmodel );
      theAntiAlphaInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiLambdaInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiSigmaMinusInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiSigmaPlusInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiXiMinusInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiXiZeroInelasticProcess->RegisterMe( theQGSPmodel );
      theAntiOmegaMinusInelasticProcess->RegisterMe( theQGSPmodel );
    }
  }
  if ( thePhysicsCase == "BERT"      || thePhysicsCase == "FTFP_BERT_ATL" ||
       thePhysicsCase == "FTFP_BERT" || thePhysicsCase == "QGSP_BERT" ) {
    // The BERT model is used for pions and nucleons in all Bertini-based physics lists
    thePhysicsCaseIsSupported = true;
    thePionMinusInelasticProcess->RegisterMe( theBERTmodel );
    thePionPlusInelasticProcess->RegisterMe( theBERTmodel );
    theProtonInelasticProcess->RegisterMe( theBERTmodel );
    theNeutronInelasticProcess->RegisterMe( theBERTmodel );
  }
  if ( thePhysicsCase == "BERT"      || thePhysicsCase == "FTFP_BERT_ATL" ||
       thePhysicsCase == "FTFP_BERT" || thePhysicsCase == "FTFP_INCLXX"   ||
       thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC" ) {
    // The BERT model is used for kaons and hyperons in all physics lists, but not for light ions
    thePhysicsCaseIsSupported = true;
    theKaonMinusInelasticProcess->RegisterMe( theBERTmodel );
    theKaonPlusInelasticProcess->RegisterMe( theBERTmodel );
    theKaonZeroLInelasticProcess->RegisterMe( theBERTmodel );
    theKaonZeroSInelasticProcess->RegisterMe( theBERTmodel );
    theLambdaInelasticProcess->RegisterMe( theBERTmodel );
    theSigmaMinusInelasticProcess->RegisterMe( theBERTmodel );
    theSigmaPlusInelasticProcess->RegisterMe( theBERTmodel );
    theXiMinusInelasticProcess->RegisterMe( theBERTmodel );
    theXiZeroInelasticProcess->RegisterMe( theBERTmodel );
    theOmegaMinusInelasticProcess->RegisterMe( theBERTmodel );
    if ( thePhysicsCase == "BERT" ) {
      theDeuteronInelasticProcess->RegisterMe( theBERTmodel );
      theTritonInelasticProcess->RegisterMe( theBERTmodel );
      theHe3InelasticProcess->RegisterMe( theBERTmodel );
      theAlphaInelasticProcess->RegisterMe( theBERTmodel );
    }
  }
  if ( thePhysicsCase == "FTFP"      || thePhysicsCase == "FTFP_BERT_ATL" ||
       thePhysicsCase == "FTFP_BERT" || thePhysicsCase == "FTFP_INCLXX"   ||
       thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC" ) {
    // The FTFP model is applied for all hadrons, but in different energy intervals according
    // whether it is consider as a stand-alone hadronic model, or within physics lists
    thePhysicsCaseIsSupported = true;
    theAntiProtonInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiNeutronInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiDeuteronInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiTritonInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiHe3InelasticProcess->RegisterMe( theFTFPmodel );
    theAntiAlphaInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiLambdaInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiSigmaMinusInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiSigmaPlusInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiXiMinusInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiXiZeroInelasticProcess->RegisterMe( theFTFPmodel );
    theAntiOmegaMinusInelasticProcess->RegisterMe( theFTFPmodel );
    G4TheoFSGenerator* theFTFPmodelToBeUsed = theFTFPmodel_constrained;
    if ( thePhysicsCase == "FTFP" ) theFTFPmodelToBeUsed = theFTFPmodel;
    thePionMinusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    thePionPlusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theKaonMinusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theKaonPlusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theKaonZeroLInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theKaonZeroSInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theProtonInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theNeutronInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theFTFPmodelToBeUsed = theFTFPmodel_halfConstrained;
    if ( thePhysicsCase == "FTFP" ) theFTFPmodelToBeUsed = theFTFPmodel;
    theDeuteronInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theTritonInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theHe3InelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theAlphaInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theIonInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theLambdaInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theSigmaMinusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theSigmaPlusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theXiMinusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theXiZeroInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
    theOmegaMinusInelasticProcess->RegisterMe( theFTFPmodelToBeUsed );
  }

  if ( ! thePhysicsCaseIsSupported ) {
    G4cerr << "ERROR: Not supported final-state hadronic inelastic physics case !"
	   << thePhysicsCase << G4endl
           << "\t Re-try by choosing one of the following:" << G4endl
           << "\t - Hadronic models : BERT, BIC, IonBIC, INCL, FTFP, QGSP" << G4endl
           << "\t - \"Physics-list proxies\" : FTFP_BERT_ATL (default), FTFP_BERT, QGSP_BERT, QGSP_BIC, FTFP_INCLXX"
	   << G4endl;
  }

  // For the case of "physics-list proxies", select the energy range for each hadronic model.
  // Note: the transition energy between hadronic models vary between physics lists,
  //       type of hadrons, and version of Geant4. Here, for simplicity, we use an uniform
  //       energy transition for all types of hadrons and regarless of the Geant4 version;
  //       moreover, for "FTFP_INCLXX" we use a different energy transition range
  //       between FTFP and INCL than in the real physics list.
  if ( thePhysicsCase == "FTFP_BERT_ATL" || thePhysicsCase == "FTFP_BERT" || thePhysicsCase == "FTFP_INCLXX" ||
       thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC" ) {
    const G4double ftfpMinE = 3.0*CLHEP::GeV;
    const G4double bertMaxE = 6.0*CLHEP::GeV;
    const G4double ftfpMinE_ATL = 9.0*CLHEP::GeV;
    const G4double bertMaxE_ATL = 12.0*CLHEP::GeV;
    const G4double ftfpMaxE = 25.0*CLHEP::GeV;
    const G4double qgspMinE = 12.0*CLHEP::GeV;
    theFTFPmodel->SetMinEnergy( 0.0 );
    theIonBICmodel->SetMaxEnergy( bertMaxE );
    theFTFPmodel_halfConstrained->SetMinEnergy( ftfpMinE );
    if ( thePhysicsCase == "FTFP_BERT_ATL" ) {
      theBERTmodel->SetMaxEnergy( bertMaxE_ATL );
      theFTFPmodel_constrained->SetMinEnergy( ftfpMinE_ATL );
    } else {
      theBERTmodel->SetMaxEnergy( bertMaxE );
      theFTFPmodel_constrained->SetMinEnergy( ftfpMinE );
    }
    if ( thePhysicsCase == "FTFP_INCLXX" ) {
      theINCLmodel->SetMaxEnergy( bertMaxE );
    }
    if ( thePhysicsCase == "QGSP_BERT" || thePhysicsCase == "QGSP_BIC" ) {
      theFTFPmodel_constrained->SetMaxEnergy( ftfpMaxE );
      theQGSPmodel->SetMinEnergy( qgspMinE );
      theBICmodel->SetMaxEnergy( bertMaxE );
    }
  }

}


HadronicGenerator::~HadronicGenerator() {
  thePartTable->DeleteAllParticles();
}


G4bool HadronicGenerator::IsApplicable( const G4String &nameProjectile, const G4double projectileEnergy ) {
  G4ParticleDefinition* projectileDefinition = thePartTable->FindParticle( nameProjectile );
  return IsApplicable( projectileDefinition, projectileEnergy );
}


G4bool HadronicGenerator::IsApplicable( G4ParticleDefinition* projectileDefinition, const G4double projectileEnergy ) {
  G4bool isApplicable = true;
  // No restrictions for "physics list proxies" because they cover all hadron types and energies.
  // For the individual models, instead, we need to consider their limitations.
  if ( thePhysicsCase == "BERT" ) {
    // We consider BERT model below 15 GeV and not for antibaryons
    if ( projectileEnergy > 15.0*CLHEP::GeV                     ||
	 projectileDefinition == G4AntiProton::Definition()     ||
	 projectileDefinition == G4AntiNeutron::Definition()    ||
	 projectileDefinition == G4AntiDeuteron::Definition()   ||
	 projectileDefinition == G4AntiTriton::Definition()     ||
	 projectileDefinition == G4AntiHe3::Definition()        ||
	 projectileDefinition == G4AntiAlpha::Definition()      ||
	 projectileDefinition == G4AntiLambda::Definition()     ||
	 projectileDefinition == G4AntiSigmaMinus::Definition() ||
	 projectileDefinition == G4AntiSigmaPlus::Definition()  ||
	 projectileDefinition == G4AntiXiMinus::Definition()    ||
	 projectileDefinition == G4AntiXiZero::Definition()     ||
	 projectileDefinition == G4AntiOmegaMinus::Definition() ) {
      isApplicable = false;
    }
  } else if ( thePhysicsCase == "QGSP" ) {
    // We consider QGSP above 2 GeV and not for light ions or anti-ions
    if ( projectileEnergy < 2.0*CLHEP::GeV                    ||
	 projectileDefinition == G4Deuteron::Definition()     ||
	 projectileDefinition == G4Triton::Definition()       ||
	 projectileDefinition == G4He3::Definition()          ||
	 projectileDefinition == G4Alpha::Definition()        ||
	 projectileDefinition == G4AntiDeuteron::Definition() ||
	 projectileDefinition == G4AntiTriton::Definition()   ||
	 projectileDefinition == G4AntiHe3::Definition()      ||
	 projectileDefinition == G4AntiAlpha::Definition() ) {
      isApplicable = false;
    }
  } else if ( thePhysicsCase == "BIC" || thePhysicsCase == "INCL" ) {
    // We consider BIC and INCL models only for pions and nucleons below 10 GeV
    // (although in recent versions INCL is capable of handling more hadrons and up to higher energies) 
    if ( ( ( projectileDefinition != G4PionMinus::Definition() ) &&
	   ( projectileDefinition != G4PionPlus::Definition() ) &&
	   ( projectileDefinition != G4Proton::Definition() ) &&
	   ( projectileDefinition != G4Neutron::Definition() ) ) ||
	 ( projectileEnergy > 10.0*CLHEP::GeV ) ) {
      isApplicable = false;
    }
  } 
  return isApplicable;
}


G4VParticleChange* HadronicGenerator::GenerateInteraction( const G4String &nameProjectile,
					                   const G4double projectileEnergy,
					                   const G4ThreeVector &projectileDirection,
					                   G4Material* targetMaterial ) {
  G4ParticleDefinition* projectileDefinition = thePartTable->FindParticle( nameProjectile );
  return GenerateInteraction( projectileDefinition, projectileEnergy, projectileDirection, targetMaterial );
}


G4VParticleChange* HadronicGenerator::GenerateInteraction( G4ParticleDefinition* projectileDefinition,
							   const G4double projectileEnergy,
					                   const G4ThreeVector &projectileDirection,
					                   G4Material* targetMaterial ) {
  // This is the most important method of the HadronicGenerator class:
  // the method performs the specified hadronic interaction
  // (by invoking the "PostStepDoIt" method of the corresponding hadronic process)
  // and returns the final state, i.e. the secondaries produced by the collision.
  // It is a relatively short method because the heavy load of setting up all
  // possible hadronic processes - with their hadronic models, transition regions,
  // and cross sections (the latter is needed for sampling the target nucleus from
  // the target material) - was already done by the constructor of the class.
  G4VParticleChange* aChange = nullptr;

  if ( ! projectileDefinition ) {
    G4cerr << "ERROR: projectileDefinition is NULL !" << G4endl;
    return aChange;
  }

  // This print-out is helpful for debugging, but it can be safely removed/commented-out
  /** 
  G4cout << "\t" << projectileDefinition->GetParticleName() << "\t" << projectileEnergy/CLHEP::GeV
         << " GeV \t" << projectileDirection << "\t" << ( targetMaterial ? targetMaterial->GetName() : "NULL" );
  **/

  if ( ! IsApplicable( projectileDefinition, projectileEnergy ) ) {
    G4cout << " -> NOT applicable !" ; //<< G4endl;
    return aChange;
  }
  //G4cout << G4endl;
  
  // Check Geant4 state (not strictly needed)
  if ( ! G4StateManager::GetStateManager()->SetNewState( G4State_PreInit ) ) {
    G4cerr << "ERROR: No possible to set G4State_PreInit !" << G4endl;
    return aChange;
  }

  // Geometry definition (not strictly needed)
  const G4double dimX = 1.0*mm;
  const G4double dimY = 1.0*mm;
  const G4double dimZ = 1.0*mm;
  G4Box* sFrame = new G4Box( "Box", dimX, dimY, dimZ );
  G4LogicalVolume* lFrame = new G4LogicalVolume( sFrame, targetMaterial, "Box", 0, 0, 0 );
  G4PVPlacement* pFrame = new G4PVPlacement( 0, G4ThreeVector(), "Box", lFrame, 0, false, 0 );
  G4TransportationManager::GetTransportationManager()->SetWorldForTracking( pFrame );

  // Projectile track & step
  G4DynamicParticle dParticle( projectileDefinition, projectileDirection, projectileEnergy );
  const G4double aTime = 0.0;
  const G4ThreeVector aPosition = G4ThreeVector( 0.0, 0.0, 0.0 );
  G4Track* gTrack = new G4Track( &dParticle, aTime, aPosition );
  G4TouchableHandle fpTouchable( new G4TouchableHistory );  // Not strictly needed
  gTrack->SetTouchableHandle( fpTouchable );                // Not strictly needed
  G4Step* step = new G4Step;
  step->SetTrack( gTrack );
  gTrack->SetStep( step );
  G4StepPoint* aPoint = new G4StepPoint;
  aPoint->SetPosition( aPosition );
  aPoint->SetMaterial( targetMaterial );
  step->SetPreStepPoint( aPoint );
  dParticle.SetKineticEnergy( projectileEnergy );
  gTrack->SetStep( step );
  gTrack->SetKineticEnergy( projectileEnergy );

  // Change Geant4 state: from "PreInit" to "Idle" (not strictly needed)
  if ( ! G4StateManager::GetStateManager()->SetNewState( G4State_Idle ) ) {
    G4cerr << "ERROR: No possible to set G4State_Idle !" << G4endl;
    return aChange;
  }

  // Finally, the hadronic interaction
  G4HadronicProcess* theProcess = nullptr;
  auto position = processMap.find( projectileDefinition );
  if ( position != processMap.end() ) theProcess = position->second;
  if ( theProcess ) aChange = theProcess->PostStepDoIt( *gTrack, *step );
                    //**************************************************
  delete pFrame;
  delete lFrame;
  delete sFrame;

  return aChange;
}
