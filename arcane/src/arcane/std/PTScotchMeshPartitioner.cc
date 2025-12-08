// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PTScotchMeshPartitioner.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Partitioneur de maillage utilisant la bibliothèque 'PTScotch'.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Iostream.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Service.h"
#include "arcane/core/Timer.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/CommonVariables.h"

#include "arcane_internal_config.h"

#include <stdint.h>

// Au cas où on utilise mpich ou openmpi
#define MPICH_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <mpi.h>
extern "C"
{
#include <ptscotch.h>
}

#include "arcane/std/MeshPartitionerBase.h"
#include "arcane/std/PTScotchMeshPartitioner_axl.h"
#include "arcane/std/PartitionConverter.h"
#include "arcane/std/GraphDistributor.h"

// TODO: supprimer les '#define' et utiliser des tests (if)
// pour être sur que tout est compilé

#define SCOTCH_SCALING
// #define SCOTCH_MAPPING

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partitionneur de maillage utilisant la bibliothèque PtScotch.
 */
class PTScotchMeshPartitioner
: public ArcanePTScotchMeshPartitionerObject
{
 public:

  explicit PTScotchMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  void build() override {}

 public:

  void partitionMesh(bool initial_partition) override;
  void partitionMesh(bool initial_partition,Int32 nb_part) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PTScotchMeshPartitioner::
PTScotchMeshPartitioner(const ServiceBuildInfo& sbi)
: ArcanePTScotchMeshPartitionerObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PTScotchMeshPartitioner::
partitionMesh(bool initial_partition)
{
  Int32 nb_part = mesh()->parallelMng()->commSize();
  partitionMesh(initial_partition,nb_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PTScotchMeshPartitioner::
partitionMesh(bool initial_partition,Int32 nb_part)
{
  ARCANE_UNUSED(initial_partition);

  info() << "Load balancing with PTScotch\n";

  IParallelMng* pm = mesh()->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  bool dumpGraph = false;
  bool checkGraph = false;
  if (options()) {
    dumpGraph = options()->dumpGraph();
    checkGraph = options()->checkGraph();
  }


  if (nb_part<nb_rank)
    throw ArgumentException(A_FUNCINFO,"partition with nb_part<nb_rank");

  // initialisations pour la gestion des contraintes (sauf initUidRef)
  initConstraints(false);

  // Contient les numéros uniques des entités dans la renumérotation
  // propre à scotch
  VariableCellInteger cell_scotch_uid(VariableBuildInfo(mesh(),"CellsScotchUid",IVariable::PNoDump|IVariable::PNoRestore));

  IntegerUniqueArray global_nb_own_cell(nb_rank);
  CellGroup own_cells = mesh()->ownCells();
  Integer nb_own_cell = nbOwnCellsWithConstraints(); // on tient compte des contraintes
  pm->allGather(IntegerConstArrayView(1,&nb_own_cell),global_nb_own_cell);
  Integer total_nb_cell = 0;
  UniqueArray<SCOTCH_Num> scotch_vtkdist(nb_rank+1);
  {
    scotch_vtkdist[0] = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      total_nb_cell += global_nb_own_cell[i];
      scotch_vtkdist[i+1] = static_cast<SCOTCH_Num>(total_nb_cell);
      //      info() << "SCOTCH VTKDIST " << (i+1) << ' ' << scotch_vtkdist[i+1];
    }
  }
  //info() << "Numéro métis de la première entité du domaine: " << scotch_vtkdist[my_rank] << " totalnbcell=" << total_nb_cell;
  // Nombre max de mailles voisine connectés aux mailles
  // en supposant les mailles connectées uniquement par les faces
  // Cette valeur sert à préallouer la mémoire pour la liste des mailles voisines
  Integer nb_max_face_neighbour_cell = 0;
  {
    // Renumérote les mailles pour que chaque sous-domaine
    // ait des mailles de numéro consécutifs
    Integer mid = static_cast<Integer>(scotch_vtkdist[my_rank]);
    ENUMERATE_CELL(i_item,own_cells){
      const Cell& item = *i_item;
      if (cellUsedWithConstraints(item)){
        cell_scotch_uid[item] = mid;
        ++mid;
      }
      nb_max_face_neighbour_cell += item.nbFace();
      //info() << " GLOBAL_SCOTCH_NUM n=" << mid << " item=" << ItemPrinter(item);
    }
    cell_scotch_uid.synchronize();
  }

  _initUidRef(cell_scotch_uid);

  // libération mémoire
  cell_scotch_uid.setUsed(false);

  SharedArray<SCOTCH_Num> scotch_xadj;
  scotch_xadj.reserve(nb_own_cell+1);

  SharedArray<SCOTCH_Num> scotch_adjncy;
  scotch_adjncy.reserve(nb_max_face_neighbour_cell);

  // Construction de la connectivité entre les cellules et leurs voisines en tenant compte des contraintes
  // (La connectivité se fait suivant les faces)

  UniqueArray<float> edgeWeights;
  edgeWeights.resize(0);
  UniqueArray<float>* edgeWeightsPtr = &edgeWeights;
//   if (initial_partition)
//     edgeWeightsPtr = NULL;

  Int64UniqueArray neighbour_cells;
  ENUMERATE_CELL(i_item,own_cells){
    const Cell& item = *i_item;

    if (!cellUsedWithConstraints(item))
      continue;

    scotch_xadj.add(scotch_adjncy.size());

    getNeighbourCellsUidWithConstraints(item, neighbour_cells, edgeWeightsPtr);

    for( Integer z=0; z<neighbour_cells.size(); ++z )
      scotch_adjncy.add(static_cast<SCOTCH_Num>(neighbour_cells[z]));
  }
  scotch_xadj.add(scotch_adjncy.size());

  int nparts = static_cast<int>(nb_part);

  // Scotch can only deal with one weight per vertex
  SharedArray<float> cells_weights;

  if (nbCellWeight() == 1) { // One criterion, we balance this criterion
    cells_weights = cellsWeightsWithConstraints(1, true);
  }
  else  { // We need multi-criteria partitioning, it's not available yet.
    // So we try to balance memory !
    cells_weights = cellsSizeWithConstraints();
  }

#ifdef SCOTCH_SCALING
  PartitionConverter<float,SCOTCH_Num> converter(pm, (double)SCOTCH_NUMMAX / 2.0, cells_weights);
  ArrayConverter<float,SCOTCH_Num,PartitionConverter<float,SCOTCH_Num> > scotch_vwgtConvert(cells_weights, converter);
#else
  ArrayConverter<float,SCOTCH_Num> scotch_vwgtConvert(cells_weights);
#endif
  SharedArray<SCOTCH_Num> scotch_vwgt(scotch_vwgtConvert.array().constView());



#ifdef SCOTCH_SCALING
  converter.reset();
  if (edgeWeights.size() == 0) // Avoid NULL pointer for Scotch.
    edgeWeights.add(0);
  converter.computeContrib(edgeWeights);
  ArrayConverter<float,SCOTCH_Num,PartitionConverter<float,SCOTCH_Num> > scotch_ewgtConvert(edgeWeights, converter);
#else
  ArrayConverter<float,SCOTCH_Num> scotch_ewgtConvert(edgeWeights);
#endif

  SharedArray<SCOTCH_Num> scotch_ewgt((UniqueArray<SCOTCH_Num>)scotch_ewgtConvert.array());


  MPI_Comm scotch_mpicomm = *(MPI_Comm*)getCommunicator();
#ifdef ARCANE_PART_DUMP
  {
    Integer iteration = mesh()->subDomain()->commonVariables().globalIteration();
    StringBuilder filename("mesh-");
    filename += iteration;
    dumpObject(filename.toString());
  }
#endif // ARCANE_PART_DUMP


  GraphDistributor gd(pm);
#ifndef SCOTCH_MAPPING
  gd.initWithOneRankPerNode(true);
#else // SCOTCH_MAPPING
  gd.initWithMaxRank(1);
#endif // SCOTCH_MAPPING

  // TODO: compute correct part number !
  SharedArray<SCOTCH_Num> scotch_part;

  scotch_xadj = gd.convert<SCOTCH_Num>(scotch_xadj, &scotch_part, true);
  scotch_vwgt = gd.convert<SCOTCH_Num>(scotch_vwgt);
  scotch_adjncy = gd.convert<SCOTCH_Num>(scotch_adjncy);
  scotch_ewgt = gd.convert<SCOTCH_Num>(scotch_ewgt);
  scotch_mpicomm = gd.getCommunicator();

  if (gd.contribute()) {

  int retval = 0;
#ifndef SCOTCH_MAPPING
  SCOTCH_Dgraph graph;
  retval = SCOTCH_dgraphInit(&graph,scotch_mpicomm);
  if (retval!=0)
    error() << "Error in dgraphInit() r=" << retval;

  info() << "Build Scotch graph";

  // TODO: Remove
  SCOTCH_randomReset(); // For debugging

  retval = SCOTCH_dgraphBuild(&graph,
                              0, /* const SCOTCH_Num baseval */
                              scotch_xadj.size()-1, /* const SCOTCH_Num vertlocnbr */
                              scotch_xadj.size()-1, /* const SCOTCH_Num vertlocmax */
                              scotch_xadj.data(), /* const SCOTCH_Num* vertloctab */
                              0, /* const SCOTCH_Num* vendloctab */
                              scotch_vwgt.data(), /* const SCOTCH_Num* veloloctab */
                              0, /* const SCOTCH_Num* vlblocltab */
                              scotch_adjncy.size(), /* const SCOTCH_Num edgelocnbr */
                              scotch_adjncy.size(), /* const SCOTCH_Num edgelocsiz */
                              scotch_adjncy.data(), /* const SCOTCH_Num* edgeloctab */
                              0, /* const SCOTCH_Num* edgegsttab */
                              scotch_ewgt.data() /* const SCOTCH_Num* edloloctab) */
                              );
  if (retval!=0)
    error() << "Error in dgraphBuild() r=" << retval;

  if (dumpGraph) {
    Integer iteration = mesh()->subDomain()->commonVariables().globalIteration();
    StringBuilder filename("graph-");
    filename += iteration;
    filename += "_";
    filename += my_rank;

    String name(filename.toString());
    FILE* ofile = ::fopen(name.localstr(),"w");
    SCOTCH_dgraphSave(&graph,ofile);
    ::fclose(ofile);
  }


  if (checkGraph) {
    // Vérifie que le maillage est correct
    info() << "Check Scotch graph";
    retval = SCOTCH_dgraphCheck(&graph);
    if (retval!=0)
      error() << "Error in dgraphCheck() r=" << retval;
  }

  SCOTCH_Strat strategy;
  retval = SCOTCH_stratInit(&strategy);
  if (retval!=0)
    error() << "Error in SCOTCH_stratInit() r=" << retval;

  if (options() && (!(options()->strategy().empty()))) {
    char* strat = (char*)malloc(options()->strategy().length()+1);
    ::strncpy(strat, options()->strategy().localstr(), options()->strategy().length()+1);
    retval = SCOTCH_stratDgraphMap(&strategy, strat);
    if (retval!=0)
      error() << "Error in SCOTCH_stratDgraphMap() r=" << retval;
  }

  // Effectue la partition
  info() << "Execute 'SCOTCH_dgraphPart'";
  retval = SCOTCH_dgraphPart(&graph,
                             nparts, /* const SCOTCH_Num partnbr */
                             &strategy, /* const SCOTCH_Strat * straptr */
                             scotch_part.unguardedBasePointer() /* SCOTCH_Num * partloctab */
                             );

  SCOTCH_stratExit(&strategy);
  SCOTCH_dgraphExit(&graph);
  if (retval!=0)
    error() << "Error in dgraphPart() r=" << retval;
#else // SCOTCH_MAPPING
  SCOTCH_Graph graph;
  SCOTCH_Arch  architecture;
  info() << "Build Scotch graph";

  // TODO: Remove
  SCOTCH_randomReset(); // For debugging

  retval = SCOTCH_graphBuild(&graph,
                              0, /* const SCOTCH_Num baseval */
                              scotch_xadj.size()-1, /* const SCOTCH_Num vertlocnbr */
                              scotch_xadj.unguardedBasePointer(), /* const SCOTCH_Num* vertloctab */
                              0, /* const SCOTCH_Num* vendloctab */
                              scotch_vwgt.begin(), /* const SCOTCH_Num* veloloctab */
                              0, /* const SCOTCH_Num* vlblocltab */
                              scotch_adjncy.size(), /* const SCOTCH_Num edgelocnbr */
                              scotch_adjncy.unguardedBasePointer(), /* const SCOTCH_Num* edgeloctab */
                              scotch_ewgt.begin() /* const SCOTCH_Num* edloloctab) */
                              );
  if (retval!=0)
    error() << "Error in graphBuild() r=" << retval;

  // Build hierarchical topology view.
  // TODO: discover topology automatically.
  SCOTCH_Num nb_nodes;
  int level = 3;
  Array<SCOTCH_Num> sizetab(level);
  Array<SCOTCH_Num> linktab(level);

  nb_nodes = nb_rank/32;
  sizetab[0] = nb_nodes;
  sizetab[1] = 4;  // 4 sockets
  sizetab[2] = 8;  // 8 cores

  retval =SCOTCH_archTleaf(&architecture, level, sizetab.unguardedBasePointer(), linktab.unguardedBasePointer());
  if (retval!=0)
    error() << "Error in archTleaf() r=" << retval;

  SCOTCH_Strat strategy;
  retval = SCOTCH_stratInit(&strategy);
  if (retval!=0)
    error() << "Error in SCOTCH_stratInit() r=" << retval;

  retval = SCOTCH_graphMap(&graph, &architecture, &strategy, scotch_part.unguardedBasePointer());


#endif // SCOTCH_MAPPING


  info() << "PART retval=" << retval;


#if 0
  {
    String filename("s_part");
    filename += my_rank;
    FILE* ofile = ::fopen(filename.localstr(),"w");
    for (Array<SCOTCH_Num>::const_iterator val(scotch_part.begin()) ; val != scotch_part.end() ; val++)
      ::fprintf(ofile, "%d\n", *val);
    info() << "GRAPH SAVED in '" << filename << "'";
    ::fclose(ofile);
  }
#endif

  } // if gd.contribute()
  scotch_part = gd.convertBack<SCOTCH_Num>(scotch_part, nb_own_cell);
#if 0
  {
    String filename("scotch_part");
    filename += my_rank;
    FILE* ofile = ::fopen(filename.localstr(),"w");
    for (Array<SCOTCH_Num>::const_iterator val(scotch_part.begin()) ; val != scotch_part.end() ; val++)
      ::fprintf(ofile, "%d\n", *val);
    info() << "GRAPH SAVED in '" << filename << "'";
    ::fclose(ofile);
  }
#endif

  VariableItemInt32& cells_new_owner = mesh()->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  {
    Integer index = 0;
//     Integer nb_new_owner = 0;
    ENUMERATE_CELL(i_item,own_cells){
      const Cell& item = *i_item;
      if (!cellUsedWithConstraints(item))
        continue;

      auto new_owner = static_cast<Int32>(scotch_part[index]);
      ++index;
      changeCellOwner(item, cells_new_owner, new_owner);
    }
  }


  // libération des tableau temporaires
  freeConstraints();

  cells_new_owner.synchronize();
  changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(PTScotchMeshPartitioner,
                        ServiceProperty("PTScotch",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

ARCANE_REGISTER_SERVICE_PTSCOTCHMESHPARTITIONER(PTScotch,PTScotchMeshPartitioner);

#if ARCANE_DEFAULT_PARTITIONER == PTSCOTCH_DEFAULT_PARTITIONER
ARCANE_REGISTER_SERVICE(PTScotchMeshPartitioner,
                        ServiceProperty("DefaultPartitioner",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));
ARCANE_REGISTER_SERVICE_PTSCOTCHMESHPARTITIONER(DefaultPartitioner,PTScotchMeshPartitioner);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
