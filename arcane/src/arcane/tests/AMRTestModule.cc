// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRUnitTest.cc                                              (C) 2000-2022 */
/*                                                                           */
/* Service de test du raffinement/dérafinnement facon AMR.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/AMRComputeFunction.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/MeshVariableInfo.h"
#include "arcane/EntryPoint.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/Properties.h"
#include "arcane/Timer.h"
#include "arcane/SharedVariable.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/ItemPrinter.h"
#include "arcane/VariableCollection.h"

#include "arcane/IItemOperationByBasicType.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/IMeshWriter.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/Directory.h"
#include "arcane/IVariableMng.h"
#include "arcane/MeshStats.h"

#include "arcane/tests/AMRTest_axl.h"

#include "arcane/tests/AMR/ExactSolution.h"
#include "arcane/tests/AMR/ErrorEstimate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class AMRTestModule
: public ArcaneAMRTestObject
{
 public:

  explicit AMRTestModule(const ModuleBuildInfo& cb);
  ~AMRTestModule();

 public:

  void init();
  void compute();
  VariableCellReal* getData() { return old_data;}
  void transportFunction (Array<ItemInternal*>& old_items, AMROperationType op);

 private:

  void _refine(Integer nb_to_refine);
  void _coarsen(Integer nb_to_coarsen);
  void _loadBalance();

  void _writeMesh(const String& filename);
  void _postProcessAMR();
  void _checkCreateOutputDir();

  Integer _executeAnalyticAdaptiveLoop(RealArray& sol,IMesh* mesh);

  void _checkParents();

 private:

  IMesh* new_mesh = nullptr;
  // Post-processing
  RealUniqueArray times;
  RealUniqueArray m_error;
  VariableCellReal* new_data = nullptr;
  VariableCellReal* old_data = nullptr;
  Directory m_output_directory;
  bool m_output_dir_created;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(AMRTestModule,AMRTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRTestModule::
AMRTestModule(const ModuleBuildInfo& mb)
: ArcaneAMRTestObject(mb)
, m_output_dir_created(false)
{
	addEntryPoint(this,"Init",
                &AMRTestModule::init,
                IEntryPoint::WInit);
	addEntryPoint(this,"compute",
                &AMRTestModule::compute);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRTestModule::
~AMRTestModule()
{
  delete old_data;
  delete new_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
init()
{
  //! AMR

	Integer nb_cell = mesh()->nbCell();
  Real amr_ratio = options()->amrRatio();
  amr_ratio = math::min(1.0,amr_ratio);
  amr_ratio = math::max(0.0,amr_ratio);
  const Integer nb_cell_old = static_cast<Integer>(nb_cell*amr_ratio);
  info() << "AMR Test nb_cell=" << nb_cell << " nb_to_refine=" << nb_cell_old;

  old_data = new VariableCellReal(Arcane::VariableBuildInfo(mesh(), "Proc",
                                                            mesh()->cellFamily()->name(),
                                                            Arcane::IVariable::PNoDump|Arcane::IVariable::PNoNeedSync));
  //	debug() << "OLD DATA SIZE " <<  old_data->variable()->nbElement() << "\n";
	ENUMERATE_CELL(icell,allCells()) {
    (*old_data)[icell] = 1.;
  }
	// creation du functor de transport de donnees d'un maillage a l'autre
	AMRComputeFunction f(this,&AMRTestModule::transportFunction);
	// Enregistrement du functor par le manager des functors associe au maillage
	// NOTE: l'objet responsable de l'appel au raffinement qui doit faire cet enregistrement
	mesh()->modifier()->registerCallBack(&f);

	FaceGroup face_group = mesh()->allCells().outerFaceGroup();

	new_mesh = subDomain()->defaultMesh();
	{
    //_refine(nb_cell_old);
    // On peut faire autant de phases qu'on souhaite. Cependant actuellement
    // pour des raisons de performance on ne fait que 2 phases.
		for( Integer i=0; i<2; ++i ){
      _refine(nb_cell_old);
      _coarsen(nb_cell_old/2);
    }
    Integer nb_active = new_mesh->allCells().size();
    info() << "NB_ACTIVE=" << nb_active;
    m_error.resize(nb_active);
    m_error.fill(0.0);
		_executeAnalyticAdaptiveLoop(m_error,new_mesh);
	}

	const Integer nb_cell_new = new_mesh->nbCell();
	info() << "NB_CELL_OLD= " << nb_cell_old << " NB_CELL_NEW= " << nb_cell_new << "\n";

	// Statistiques sur le nouveau maillage
	MeshStats stats(traceMng(),new_mesh,subDomain()->parallelMng());
	stats.dumpStats();
	
	// NOTE: avant de detruire old_data, on peut le garder pour visualiser la projection
	// des donnees stocker dedans
	// old_data sert ici pour afficher les procs en parallele
  delete old_data;
	
	old_data = new VariableCellReal(VariableBuildInfo(new_mesh, "Proc",
                                                    new_mesh->cellFamily()->name(),
                                                    IVariable::PNoDump|IVariable::PNoNeedSync));
	//debug() << "OLD DATA SIZE " <<  old_data->variable()->nbElement() << "\n";
	ENUMERATE_CELL(icell,allCells()) {
    (*old_data)[icell] = icell->owner();
  }
	new_data = new VariableCellReal(VariableBuildInfo(new_mesh, "Sol", new_mesh->cellFamily()->name(),
                                                    IVariable::PNoDump|IVariable::PNoNeedSync));
	{
		StringBuilder filename = "amrmesh";
		filename += subDomain()->subDomainId();
		filename += ".mli";
		_writeMesh(filename.toString());
	}
	{
		_checkCreateOutputDir();
		IPostProcessorWriter* post_processor = options()->format();
		post_processor->setBaseDirectoryName(m_output_directory.path());
		_postProcessAMR();
	}

	mesh()->modifier()->unRegisterCallBack(&f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
compute()
{
	subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_refine(Integer nb_to_refine)
{
	Int32UniqueArray cells_local_id;
	const Integer nb_cell= mesh()->nbCell();
	ARCANE_ASSERT((nb_to_refine<=nb_cell),("NB CELL TO REFINE EXCEED NB CELL OF THE MESH"));
	// Recherche les nb_to_refine premières mailles de type IT_Hexaedron8
	ENUMERATE_CELL(icell,mesh()->ownActiveCells()){
		Cell cell = *icell;
		if (cell.type()==IT_Hexaedron8 || cell.type()==IT_Quad4){
			cells_local_id.add(cell.localId());
			nb_to_refine--;
			if (nb_to_refine == 0)
				break;
		}
	}
  info() << "NB_CELL_TO_REFINE=" << cells_local_id.size();
	mesh()->modifier()->flagCellToRefine(cells_local_id);
	mesh()->modifier()->adapt();
  _checkParents();
  MeshStats ms(traceMng(),mesh(),mesh()->parallelMng());
  ms.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_coarsen(Integer nb_to_coarsen)
{
  info() << "Coarsening cells nb_to_coarsen=" << nb_to_coarsen;
	Int32UniqueArray cells_local_id;
	// Recherche les nb_to_coarsen premières mailles de type IT_Hexaedron8
	Integer nb_child_to_coarsen= nb_to_coarsen*8;
	ENUMERATE_CELL(icell,mesh()->ownActiveCells()){
		Cell cell = *icell;
		if (cell.type()==IT_Hexaedron8 && cell.level()>0){

			Cell parent_cell= cell.hParent();
			for(Integer c=0;c<parent_cell.nbHChildren();c++){
				Cell child = parent_cell.hChild(c);
				cells_local_id.add(child.localId());
			}
			nb_child_to_coarsen--;
			if (nb_child_to_coarsen <= 0)
				break;
		}
	}
  info() << "Computed nb to coarsen=" << cells_local_id.size();
	mesh()->modifier()->flagCellToCoarsen(cells_local_id);
	mesh()->modifier()->adapt();
  _checkParents();
  MeshStats ms(traceMng(),mesh(),mesh()->parallelMng());
  ms.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_loadBalance()
{
  // Test de migration
  VariableItemInt32& cells_new_owner = mesh()->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  ENUMERATE_FACE(iface,allFaces()) {
    if (!iface->isOwn())
      for (CellLocalId icell : iface->cells())
        cells_new_owner[icell] = iface->owner();
  }
  info() << "Own cells before migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);
  cells_new_owner.synchronize();
  Integer moved_cell_count = 0;
  ENUMERATE_CELL(icell,ownCells()) {
    if (cells_new_owner[icell] != icell->owner())
      ++moved_cell_count;
  }
  info() << "Own cells to move in migration : " <<  moved_cell_count;
  mesh()->utilities()->changeOwnersFromCells();
//   for(IItemFamilyCollection::Enumerator i(mesh()->itemFamilies()); ++i;) {
//     IItemFamily * family = *i;
//     VariableItemInt32& owner = family->itemsNewOwner();
//     const Integer subDomainId = subDomain()->subDomainId();
//     UniqueArray<Integer> counts(subDomain()->nbSubDomain(),0);
//     ENUMERATE_ITEM(iitem,family->allItems().own())
//       ++counts[owner[iitem]];
//     parallelMng()->reduce(Parallel::ReduceMax,counts);
//     info() << "Total " << family->itemKind() << " for domain " << subDomainId << " : " << counts[subDomainId];
//   }

  mesh()->modifier()->setDynamic(true);
  bool compact = mesh()->properties()->getBool("compact");
  mesh()->properties()->setBool("compact", true);
  mesh()->toPrimaryMesh()->exchangeItems();
  mesh()->properties()->setBool("compact", compact);
  info() << "Own cells after migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_writeMesh(const String& filename)
{
  ServiceBuilder<IMeshWriter> sb(subDomain());
  auto mesh_io(sb.createReference("Lima",SB_AllowNull));
  if (mesh_io.get())
    mesh_io->writeMeshToFile(mesh(),filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_postProcessAMR()
{
  info() << "Post-process AMR";
  IPostProcessorWriter* post_processor = options()->format();
  times.add(m_global_time());
  post_processor->setTimes(times);
  post_processor->setMesh(new_mesh);

  /*m_data.fill(0.);
  Integer i=0;
  ENUMERATE_CELL(icell,new_mesh->allActiveCells()) {
    m_data[icell] = m_error[i++];
  }*/
  new_data->fill(0.);
  Integer i=0;
  ENUMERATE_CELL(icell,new_mesh->allActiveCells()) {
    (*new_data)[icell] = m_error[i++];
  }

  VariableList variables;
  variables.add(new_data->variable());
  variables.add(old_data->variable());
  post_processor->setVariables(variables);
  ItemGroupList groups;
  groups.add(allCells());
  post_processor->setGroups(groups);
  IVariableMng* vm = subDomain()->variableMng();
  vm->writePostProcessing(post_processor);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_checkCreateOutputDir()
{
  if (m_output_dir_created)
    return;
  m_output_directory = Directory(subDomain()->exportDirectory(),"depouillement3");
  m_output_directory.createDirectory();
  m_output_dir_created = true;
  info() << "Creating output dir '" << m_output_directory.path() << "' for export";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer AMRTestModule::
_executeAnalyticAdaptiveLoop(RealArray& sol,IMesh* mesh)
{
  // Parse les options d'adaptation
  const Integer max_adapt_iters = 4;
  const Integer max_level = 3;
  const Real refine_percentage = 0.5;
  const Real coarsen_percentage = 0.2;

  singularity = true;

  // Création de l'objet ExactSolution et  attachement des functors solution
  ErrorEstimate exact_sol;
  exact_sol.attachExactValue(exact3DSolution);
  exact_sol.attachExactGradient(exact3DGradient);

  // boucle adaptative.
  Integer adapt_iter;
  RealUniqueArray error;
  for (adapt_iter=0; adapt_iter<max_adapt_iters-1; adapt_iter++) {
	info() << "Beginning Adaptive Loop " << adapt_iter << "\n";
	//  bloc d'adaptation
	{
	  info() << "  Refining the mesh..." << "\n";

	  // un objet \p ErrorEstimate interroge une solution approchée
	  // et affecte à chaque maille une valeur d'erreur positive
	  // Cette valeur est utilisée pour la prise de décision raffinement
	  // déraffinement.
	  // Pour ce cas test simple, nous utilisons une erreur
	  // d'interpolation sur la solution exacte
	  // Pour des cas réels, nous avons besoins d'un indicateur d'erreur
	  // sur la solution approchée.

	  // Calcul de l'erreur pour chaque maille active en utilisant l'indicateur d'erreur
	  // Note: dans le cas général, il faut un estimateur d'erreur specifique
	  // à l'application.
	  exact_sol.computeError(error,mesh);

	  // infos
	  info() << "L2-Error is: " << exact_sol.l2Error() << "\n";
	  info() << "LInf-Error is: " << exact_sol.lInfError() << "\n";

	  // A partir de l'erreur calculée dans \p error on décide quelle maille va être
	  // raffinée ou déraffinée.  Dans cet exemple l'approche est la suivante:
	  // chaque maille avec un pourcentage de 20% de l'erreur maximale
	  // va être raffinée, et chaque maille avec 10% de l'erreur minimale va peut être déraffinée
	  // Il faut noter que les mailles flaguées pour raffinement vont être raffinées,
	  // mais les mailles  flaguées pour déraffinement peuvent être déraffinées.
	  // ErrorToFlagConverter(error);
	  exact_sol.errorToFlagConverter(error,refine_percentage,coarsen_percentage,max_level,mesh);

	  // Adapter le maillage en raffinant déraffinant les mailles flaguées.
  	  //  Projection des solutions, paramètres, etc.
	  // de l'ancien maillage au nouveau maillage. Pour cela des callbacks
	  // se font dans la classe MeshRefinement
	  mesh->modifier()->adapt();
	}
  }
  // la derniere iteration pour calcul l erreur
  {
    info() << "Beginning Adaptive Loop " << adapt_iter << "\n";

    // Calcul de l'erreur.
    exact_sol.computeError(error,mesh);
    info() << "L2-Error is: " << exact_sol.l2Error() << "\n";
    info() << "LInf-Error is: " << exact_sol.lInfError() << "\n";

    exact_sol.computeSol(sol,mesh);
  }

  // All done
  return 0;
}

// -------------------------------------------------------------------
// Exemple d'une fonction callback permettant la projection des
// variables/solutions lors d'une iteration AMR.
// Cette fonction sera enregistree par le module AMRTest et sera
// donc appelee tout au long des iterations AMR.
// --------------------------------------------------------------------
void AMRTestModule::
transportFunction (Array<ItemInternal*>& old_items, AMROperationType op)
{
  // Prolongement/Restriction en fonction de l'operation maillage
  VariableCellReal& data = *this->getData();
  switch (op){
  case Restriction:
    for(Integer i=0,is=old_items.size();i<is;i++){
      Cell parent = old_items[i];
      UInt32 nb_children = parent.nbHChildren() ;
      Real value = 0. ;
      for (UInt32 j = 0; j < nb_children; j++) {
        value += data[parent.hChild(j)] ;
      }
      data[parent] = value/nb_children ;
    }
    break;
  case Prolongation:
    for(Integer i=0,is=old_items.size();i<is;i++){
      Cell parent= old_items[i];
      // coarse-to-fine: interpolation
      Real value = data[parent] ;
      for (UInt32 j = 0, js = parent.nbHChildren(); j < js; j++){
        data[parent.hChild(j)] = value ;
      }
    }
    break;
  default:
    ARCANE_FATAL("No callback function should be called with this operation {1}",op);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_checkParents()
{
  ENUMERATE_CELL(icell,allCells()){
    Cell c = *icell;
    Cell parent = c.topHParent();
    if (parent.null())
      ARCANE_FATAL("No topHParent() for cell {0}",ItemPrinter(c));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
