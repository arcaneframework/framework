// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisMeshPartitioner.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Partitioneur de maillage utilisant la bibliothèque PARMetis.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FloatingPointExceptionSentry.h"

#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/ItemGroup.h"
#include "arcane/Service.h"
#include "arcane/Timer.h"
#include "arcane/FactoryService.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IItemFamily.h"
#include "arcane/MeshVariable.h"
#include "arcane/VariableBuildInfo.h"

#include "arcane/std/MeshPartitionerBase.h"
#include "arcane/std/MetisMeshPartitioner_axl.h"

// Au cas où on utilise mpich2 ou openmpi pour éviter d'inclure le C++
#define MPICH_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <parmetis.h>
// #define CEDRIC_LB_2

#if (PARMETIS_MAJOR_VERSION < 4)
#error "Your version of Parmetis is too old .Version 4.0+ is required"
#endif

typedef real_t realtype;
typedef idx_t idxtype;

#include "arcane/std/PartitionConverter.h"
#include "arcane/std/GraphDistributor.h"
#include "arcane/std/internal/MetisWrapper.h"

#include <chrono>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

using MetisCallStrategy = TypesMetisMeshPartitioner::MetisCallStrategy;
using MetisEmptyPartitionStrategy = TypesMetisMeshPartitioner::MetisEmptyPartitionStrategy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partitioneur de maillage utilisant la bibliothèque PARMetis.
 */
class MetisMeshPartitioner
: public ArcaneMetisMeshPartitionerObject
{
 public:

  explicit MetisMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  void build() override {}

 public:

  void partitionMesh(bool initial_partition) override;
  void partitionMesh(bool initial_partition,Int32 nb_part) override;


 private:

  IParallelMng* m_parallel_mng = nullptr;
  Integer m_nb_refine = -1;
  Integer m_random_seed = 15;
  bool m_disable_floatingexception = false;
  void _partitionMesh(bool initial_partition,Int32 nb_part);
  void _removeEmptyPartsV1(Int32 nb_part, Int32 nb_own_cell, ArrayView<idxtype> metis_part);
  void _removeEmptyPartsV2(Int32 nb_part,ArrayView<idxtype> metis_part);
  Int32 _removeEmptyPartsV2Helper(Int32 nb_part,ArrayView<idxtype> metis_part,Int32 algo_iteration);
  int _writeGraph(IParallelMng* pm,
                  Span<const idxtype> metis_vtkdist,
                  Span<const idxtype> metis_xadj,
                  Span<const idxtype> metis_adjncy,
                  Span<const idxtype> metis_vwgt,
                  Span<const idxtype> metis_ewgt,
                  const String& Name) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MetisMeshPartitioner::
MetisMeshPartitioner(const ServiceBuildInfo& sbi)
: ArcaneMetisMeshPartitionerObject(sbi)
{
  m_parallel_mng = mesh()->parallelMng();
  String s = platform::getEnvironmentVariable("ARCANE_DISABLE_METIS_FPE");
  if (s=="1" || s=="TRUE")
    m_disable_floatingexception = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisMeshPartitioner::
partitionMesh(bool initial_partition)
{
  Int32 nb_part = m_parallel_mng->commSize();
  partitionMesh(initial_partition,nb_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisMeshPartitioner::
partitionMesh(bool initial_partition,Int32 nb_part)
{
  // Signale que le partitionnement peut planter, car metis n'est pas toujours
  // tres fiable sur le calcul flottant.
  initial_partition = (subDomain()->commonVariables().globalIteration() <= 2);
  if (m_disable_floatingexception){
    FloatingPointExceptionSentry fpes(false);
    _partitionMesh(initial_partition,nb_part);
  }
  else
    _partitionMesh(initial_partition,nb_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisMeshPartitioner::
_partitionMesh(bool initial_partition,Int32 nb_part)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  IMesh* mesh = this->mesh();
  bool force_partition = false;

  if (nb_part<nb_rank)
    ARCANE_THROW(ArgumentException,"partition with nb_part ({0}) < nb_rank ({1})",
                 nb_part,nb_rank);

  if (nb_rank==1){
    info() << "INFO: Unable to test load balancing on a single sub-domain";
    return;
  }

  // TODO : comprendre les modifs de l'IFPEN
  // utilise toujours re-partitionnement complet.
  // info() << "Metis: params " << m_nb_refine << " " << imbalance() << " " << maxImbalance();
  // info() << "Metis: params " << (m_nb_refine>=10) << " " << (imbalance()>4.0*maxImbalance()) << " " << (imbalance()>1.0);
  bool force_full_repartition = false;
  //force_full_repartition = true;
  force_full_repartition |= (m_nb_refine < 0); // toujours la première fois, pour compatibilité avec l'ancienne version

  if (nb_part != nb_rank) { // Pas de "repartitionnement" si le nombre de parties ne correspond pas au nombre de processeur.
  	force_full_repartition = true;
  	force_partition = true;
  }

  info() << "WARNING: compensating the potential lack of 'Metis' options in case of manual instanciation";
  Integer max_diffusion_count = 10; // reprise des options par défaut du axl
  Real imbalance_relative_tolerance = 4;
  float tolerance_target = 1.05f;
  bool dump_graph = false;
  MetisCallStrategy call_strategy = MetisCallStrategy::one_processor_per_node;
  bool in_out_digest = false;
  
  // Variables d'environnement permettant de regler les options lorsqu'il n'y a pas de jeu de donnees
  
  String call_strategy_env = platform::getEnvironmentVariable("ARCANE_METIS_CALL_STRATEGY");
  if (call_strategy_env == "all-processors"){
    call_strategy = MetisCallStrategy::all_processors;
  } else if (call_strategy_env == "one-processor-per-node") {
    call_strategy = MetisCallStrategy::one_processor_per_node;
  } else if (call_strategy_env == "two-processors-two-nodes") {
    call_strategy = MetisCallStrategy::two_processors_two_nodes;
  } else if (call_strategy_env == "two-gathered-processors") {
    call_strategy = MetisCallStrategy::two_gathered_processors;
  } else if (call_strategy_env == "two-scattered-processors") {
    call_strategy = MetisCallStrategy::two_scattered_processors;
  } else if (!call_strategy_env.null()) {
    ARCANE_FATAL("Invalid value '{0}' for ARCANE_METIS_CALL_STRATEGY environment variable",call_strategy_env);
  }
  
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_METIS_INPUT_OUTPUT_DIGEST", true)) 
    in_out_digest = (v.value()!=0);
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_METIS_DUMP_GRAPH", true))
    dump_graph = (v.value()!=0);

  if (options()){
    max_diffusion_count = options()->maxDiffusiveCount();
    imbalance_relative_tolerance = options()->imbalanceRelativeTolerance();
    tolerance_target = (float)(options()->toleranceTarget());
    dump_graph = options()->dumpGraph();
    call_strategy = options()->metisCallStrategy();
    in_out_digest = options()->inputOutputDigest();
  }

  if (max_diffusion_count > 0)
    force_full_repartition |= (m_nb_refine>=max_diffusion_count);
  force_full_repartition |= (imbalance()>imbalance_relative_tolerance*maxImbalance());
  force_full_repartition |= (imbalance()>1.0);

  // initialisations pour la gestion des contraintes (sauf initUidRef)
  initConstraints(false);

  bool is_shared_memory = pm->isThreadImplementation();
  // En mode mémoire partagé, pour l'instant on force le fait d'utiliser un seul
  // processeur par noeud car on ne sait pas si on dispose de plusieurs rang par noeud.
  // Notamment, en mode mémoire partagé sans MPI on n'a obligatoirement qu'un seul noeud
  if (is_shared_memory)
    call_strategy = MetisCallStrategy::one_processor_per_node;

  idxtype nb_weight = nbCellWeight();

  info() << "Load balancing with Metis nb_weight=" << nb_weight << " initial=" << initial_partition
         << " call_strategy=" << (int)call_strategy
         << " is_shared_memory?=" << is_shared_memory
         << " disabling_fpe?=" << m_disable_floatingexception
         << " sizeof(idxtype)==" << sizeof(idxtype);

  if (nb_weight==0)
    initial_partition = true;

  UniqueArray<idxtype> metis_vtkdist(nb_rank+1);
  Integer total_nb_cell = 0;

  // Contient les numéros uniques des entités dans la renumérotation
  // propre à metis
  VariableCellInteger cell_metis_uid(VariableBuildInfo(mesh,"CellsMetisUid",IVariable::PNoDump));

  UniqueArray<Integer> global_nb_own_cell(nb_rank);
  CellGroup own_cells = mesh->ownCells();
  Integer nb_own_cell = nbOwnCellsWithConstraints(); // on tient compte des contraintes
  pm->allGather(ConstArrayView<Integer>(1,&nb_own_cell),global_nb_own_cell);
  Int32 nb_empty_part = 0;
  {
    //Integer total_first_uid = 0;
    metis_vtkdist[0] = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      //total_first_uid += global_nb_own_cell[i];
      Int32 n = global_nb_own_cell[i];
      if (n==0)
        ++nb_empty_part;
      total_nb_cell += n;
      metis_vtkdist[i+1] = static_cast<idxtype>(total_nb_cell);
      //      info() << "METIS VTKDIST " << (i+1) << ' ' << metis_vtkdist[i+1];
    }
  }
  // N'appelle pas Parmetis si on n'a pas de mailles car sinon cela provoque une erreur.
  info() << "Total nb_cell=" << total_nb_cell << " nb_empty_partition=" << nb_empty_part;
  if (total_nb_cell==0){
    info() << "INFO: mesh '" << mesh->name() << " has no cell. No partitioning is needed";
    freeConstraints();
    return;
  }

  // HP: Skip this shortcut because it produces undefined itemsNewOwner variable.
/*
  if (total_nb_cell < nb_rank) { // Dans ce cas, pas la peine d'appeler ParMetis.
	warning() << "There are no subdomain except cells, no reffinement";
	freeConstraints();
	return;
  }
*/

  // Nombre max de mailles voisines connectées aux mailles
  // en supposant les mailles connectées uniquement par les faces
  // (ou les arêtes pour une maille 2D dans un maillage 3D)
  // Cette valeur sert à préallouer la mémoire pour la liste des mailles voisines
  Integer nb_max_face_neighbour_cell = 0;
  {
    // Renumérote les mailles pour Metis pour que chaque sous-domaine
    // ait des mailles de numéro consécutifs
    Integer mid = static_cast<Integer>(metis_vtkdist[my_rank]);
    ENUMERATE_ (Cell, i_item, own_cells) {
      Cell item = *i_item;
      if (cellUsedWithConstraints(item)){
        cell_metis_uid[item] = mid;
        ++mid;
      }
      bool use_face = true;
      if (_isNonManifoldMesh()) {
        Int32 dim = item.typeInfo()->dimension();
        if (dim == 2 && _meshDimension() == 3) {
          nb_max_face_neighbour_cell += item.nbEdge();
          use_face = false;
        }
      }
      if (use_face)
        nb_max_face_neighbour_cell += item.nbFace();
    }
    cell_metis_uid.synchronize();
  }

  _initUidRef(cell_metis_uid);

  // libération mémoire
  cell_metis_uid.setUsed(false);

  SharedArray<idxtype> metis_xadj;
  metis_xadj.reserve(nb_own_cell+1);

  SharedArray<idxtype> metis_adjncy;
  metis_adjncy.reserve(nb_max_face_neighbour_cell);


  // Construction de la connectivité entre les cellules et leurs voisines en tenant compte des contraintes
  // (La connectivité se fait suivant les faces)
  Int64UniqueArray neighbour_cells;
  UniqueArray<float> edge_weights;
  edge_weights.resize(0);
  ENUMERATE_CELL(i_item,own_cells){
    Cell item = *i_item;

    if (!cellUsedWithConstraints(item))
      continue;

    metis_xadj.add(metis_adjncy.size());

    getNeighbourCellsUidWithConstraints(item, neighbour_cells, &edge_weights);
    for( Integer z=0; z<neighbour_cells.size(); ++z )
      metis_adjncy.add(static_cast<idxtype>(neighbour_cells[z]));
  }
  metis_xadj.add(metis_adjncy.size());

  idxtype wgtflag = 3; // Sommets et aretes
//2; // Indique uniquement des poids sur les sommets
  idxtype numflags = 0;
  idxtype nparts = static_cast<int>(nb_part);

  // Poids aux sommets du graphe (les mailles)
//   if (initial_partition)
//     nb_weight = 1;

  String s = platform::getEnvironmentVariable("ARCANE_LB_PARAM");
  if (!s.null() && (s.upper() == "PARTICLES"))
    nb_weight = 1; // Only one weight in this case !

#if CEDRIC_LB_2
  nb_weight = 1;
#endif
  bool cond = (nb_weight == 1);
  //  Real max_int = 1 << (sizeof(idxtype)*8-1);
  UniqueArray<realtype> metis_ubvec(CheckedConvert::toInteger(nb_weight));
  SharedArray<float> cells_weights;

  PartitionConverter<float,idxtype> converter(pm, (Real)IDX_MAX, cond);


  /** The following loop is used to fix a ParMetis bug:
      ParMetis does not know how to deal with several imbalances ...
      If we detect that load imbalance cannot be insured, we
      increase all load imbalances.
   */
  cells_weights = cellsWeightsWithConstraints(CheckedConvert::toInteger(nb_weight));
  // Déséquilibre autorisé pour chaque contrainte
  metis_ubvec.resize(CheckedConvert::toInteger(nb_weight));
  metis_ubvec.fill(tolerance_target);

  converter.reset(CheckedConvert::toInteger(nb_weight));

  do {
    cond = converter.isBalancable<realtype>(cells_weights, metis_ubvec, nb_part);
    if (!cond) {
      info() << "We have to increase imbalance :";
      info() << metis_ubvec;
      for (auto& tol: metis_ubvec ) {
        tol *= 1.5f;
      }
    }
  } while (!cond);

  ArrayConverter<float,idxtype,PartitionConverter<float,idxtype> > metis_vwgtConvert(cells_weights, converter);

  SharedArray<idxtype> metis_vwgt(metis_vwgtConvert.array().constView());

  const bool do_print_weight = false;
  if (do_print_weight){
    StringBuilder fname;
    Integer iteration = mesh->subDomain()->commonVariables().globalIteration();
    fname = "weigth-";
    fname += pm->commRank();
    fname += "-";
    fname += iteration;
    String f(fname);
    std::ofstream dumpy(f.localstr());
    for (int i = 0; i < metis_xadj.size()-1 ; ++i) {
      dumpy << " Weight uid=" << i;
      for( int j=0 ; j < nb_weight ; ++j ){
        dumpy << " w=" << *(metis_vwgt.begin()+i*nb_weight+j)
              << "(" << cells_weights[i*nb_weight+j] <<")";
      }
      dumpy << '\n';
    }
  }

  converter.reset(1); // Only one weight for communications !

  converter.computeContrib(edge_weights);
  ArrayConverter<float,idxtype,PartitionConverter<float,idxtype> > metis_ewgt_convert(edge_weights, converter);
  SharedArray<idxtype> metis_ewgt((UniqueArray<idxtype>)metis_ewgt_convert.array());

  SharedArray<float> cells_size;
  realtype itr=50; // In average lb every 20 iterations ?
  SharedArray<idxtype> metis_vsize;
  if (!initial_partition) {
    cells_size = cellsSizeWithConstraints();
    converter.computeContrib(cells_size, (Real)itr);
    metis_vsize.resize(metis_xadj.size()-1,1);
  }


  idxtype metis_options[4];
  metis_options[0] = 0;

  // By default RandomSeed is fixed

#ifdef CEDRIC_LB_2
  m_random_seed = Convert::toInteger((Real)MPI_Wtime());
#endif // CEDRIC_LB_2

  metis_options[0] = 1;
  metis_options[1] = 0;
  metis_options[2] = m_random_seed;
  metis_options[3] = 1; // Initial distribution information is implicit
  /*
   * Comment to keep same initial random seed each time !
   m_random_seed++;
   */

  idxtype metis_edgecut = 0;

  // TODO: compute correct part number !
  UniqueArray<idxtype> metis_part;
  
  std::chrono::high_resolution_clock clock;
  auto start_time = clock.now();

  GraphDistributor gd(pm);
  
  // Il y a actuellement 2 mecanismes de regroupement du graph : celui fait par "GraphDistributor" et
  // celui fait par le wrapper ParMetis. Il est possible de combiner les 2 (two_processors_two_nodes).
  // A terme, ces 2 mecanismes devraient fusionner et il ne faudrait conserver que le "GraphDistributor".
  // 
  // La redistribution par le wrapper n'est faite que dans les 2 cas suivants :
  //   - on veut un regroupement sur 2 processeurs apres un regroupement par noeuds (two_processors_two_nodes)
  //   - on veut un regroupement direct sur 2 processeurs, en esperant qu'ils soient sur 2 noeuds distincts (two_scattered_processors)
  
  bool redistribute = true; // redistribution par GraphDistributor
  bool redistribute_by_wrapper = false; // redistribution par le wrapper ParMetis
  
  if (call_strategy == MetisCallStrategy::all_processors || call_strategy == MetisCallStrategy::two_scattered_processors) {
    redistribute = false;
  }
  
  if (call_strategy == MetisCallStrategy::two_processors_two_nodes || call_strategy == MetisCallStrategy::two_scattered_processors) {
    redistribute_by_wrapper = true;
  }
  // Indique si on n'autorise de n'utiliser qu'un seul PE.
  // historiquement (avant avril 2020) cela n'était pas autorisé car cela revenait
  // à appeler 'ParMetis' sur un seul PE ce qui n'était pas supporté.
  // Maintenant, on appelle directement Metis dans ce cas donc cela ne pose pas de
  // problèmes. Cependant pour des raisons historiques on garde l'ancien comportement
  // sauf pour deux cas:
  // - si l'échange de messages est en mode mémoire partagé. Comme le partitionnement
  //   dans ce mode n'était pas supporté avant, il n'y a
  //   pas d'historique à conserver. De plus cela est indispensable si on n'utilise
  //   qu'un seul noeud de calcul car alors
  // - lors du partionnement initial et s'il y a des partitions vides. Ce cas n'existait
  //   pas avant non plus car on utilisait 'MeshPartitionerTester' pour faire un
  //   premier pré-partitionnement qui garantit aucune partition vide. Cela permet
  //   d'éviter ce pré-partitionnement.

  bool gd_allow_only_one_rank = false;
  if (is_shared_memory || (nb_empty_part!=0 && initial_partition))
    gd_allow_only_one_rank = true;

  // S'il n'y a qu'une seule partie non-vide on n'utilise qu'un seul rang. Ce cas
  // intervient normalement uniquement en cas de partitionnement initial et si
  // un seul rang (en général le rang 0) a des mailles.
  if ((nb_empty_part+1)==nb_rank){
    info() << "Initialize GraphDistributor with max rank=1";
    gd.initWithMaxRank(1);
  }
  else if (call_strategy == MetisCallStrategy::two_gathered_processors && (nb_rank > 2)) {
    // Seuls les 2 premiers processeurs seront utilisés.
    info() << "Initialize GraphDistributor with max rank=2";
    gd.initWithMaxRank(2);
  }
  else {
    info() << "Initialize GraphDistributor with one rank per node"
           << " (allow_one?=" << gd_allow_only_one_rank << ")";
    gd.initWithOneRankPerNode(gd_allow_only_one_rank);
  }
  
  IParallelMng* metis_pm = pm;

  info() << "Using GraphDistributor?=" << redistribute;
  if (redistribute){
    metis_xadj = gd.convert<idxtype>(metis_xadj, &metis_part, true);
    metis_vwgt = gd.convert<idxtype>(metis_vwgt);
    metis_adjncy = gd.convert<idxtype>(metis_adjncy);
    metis_ewgt = gd.convert<idxtype>(metis_ewgt);
    if (!initial_partition)
      metis_vsize = gd.convert<idxtype>(metis_vsize);
    metis_pm = gd.subParallelMng();

    metis_vtkdist.resize(gd.size()+1);
    if (gd.contribute()) {
      UniqueArray<Integer> buffer(gd.size());
      Integer nbVertices = metis_part.size();
      gd.subParallelMng()->allGather(ConstArrayView<Integer>(1,&nbVertices),buffer);
      metis_vtkdist[0] = 0;
      for (int i = 1 ; i < metis_vtkdist.size() ; ++i) {
        metis_vtkdist[i] = metis_vtkdist[i-1] + buffer[i-1];
      }
    }
    metis_options[3] = PARMETIS_PSR_UNCOUPLED ; // Initial distribution information is in metis_part
  }
  else{

    metis_part = UniqueArray<idxtype>(nb_own_cell, my_rank);
  }
  MPI_Comm metis_mpicomm = static_cast<MPI_Comm>(metis_pm->communicator());

  bool do_call_metis = true;
  if (redistribute)
    do_call_metis = gd.contribute();

  if (do_call_metis) {
    if (dump_graph) {
      Integer iteration = sd->commonVariables().globalIteration();
      StringBuilder name("graph-");
      name += iteration;
      _writeGraph(metis_pm, metis_vtkdist, metis_xadj, metis_adjncy, metis_vwgt, metis_ewgt, name.toString());
    }

    // ParMetis >= 4 requires to define tpwgt.
    const Integer tpwgt_size = CheckedConvert::toInteger(nb_part) * CheckedConvert::toInteger(nb_weight);
    UniqueArray<realtype> tpwgt(tpwgt_size);
    float fill_value = (float)(1.0/(double)nparts);
    tpwgt.fill(fill_value);

    int retval = METIS_OK;
    Timer::Action ts(sd,"Metis");

    MetisWrapper wrapper(metis_pm);
    force_partition |= initial_partition;
    force_full_repartition |= force_partition;
    if (force_full_repartition){
      m_nb_refine = 0;
      if (force_partition) {
        info() << "Metis: use a complete partitioning.";
        retval = wrapper.callPartKway(in_out_digest,
                                      redistribute_by_wrapper,
                                      metis_vtkdist.data(),
                                      metis_xadj.data(),
                                      metis_adjncy.data(),
                                      metis_vwgt.data(),
                                      metis_ewgt.data(),
                                      &wgtflag,&numflags,&nb_weight,
                                      &nparts, tpwgt.data(),
                                      metis_ubvec.data(),
                                      metis_options,&metis_edgecut,
                                      metis_part.data());
      }
      else {
        info() << "Metis: use a complete REpartitioning.";
        retval = wrapper.callAdaptiveRepart(in_out_digest,
                                            redistribute_by_wrapper,
                                            metis_vtkdist.data(),
                                            metis_xadj.data(),
                                            metis_adjncy.data(),
                                            metis_vwgt.data(),
                                            metis_vsize.data(), // Vsize !
                                            metis_ewgt.data(),
                                            &wgtflag,&numflags,&nb_weight,
                                            &nparts,tpwgt.data(),
                                            metis_ubvec.data(),
                                            &itr, metis_options,&metis_edgecut,
                                            metis_part.data());
      }
    }
    else{
      // TODO: mettre dans MetisWrapper et supprimer utilisation de .data()
      // (cela doit être faire par le wrapper)
      ++m_nb_refine;
      info() << "Metis: use a diffusive REpartitioning";
      retval = ParMETIS_V3_RefineKway(metis_vtkdist.data(),
                                      metis_xadj.data(),
                                      metis_adjncy.data(),
                                      metis_vwgt.data(),
                                      metis_ewgt.data(),
                                      &wgtflag,&numflags,&nb_weight,
                                      &nparts,tpwgt.data(),
                                      metis_ubvec.data(),
                                      metis_options,&metis_edgecut,
                                      metis_part.data(),
                                      &metis_mpicomm);
    }
    if (retval != METIS_OK) {
      ARCANE_FATAL("Error while computing ParMetis partitioning r='{0}'",retval);
    }
  }
  if (redistribute){
    metis_part = gd.convertBack<idxtype>(metis_part, nb_own_cell);
  }

  auto end_time = clock.now();
  std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  info() << "Time to partition using parmetis = " << duration.count() << " us ";

  // Stratégie à adopter pour supprimer les partitions vides.
  // A noter qu'il faut faire cela avant d'appliquer les éventuelles contraintes
  // pour garantir que ces dernières seront bien respectées
  if (options()){
    switch(options()->emptyPartitionStrategy()){
    case MetisEmptyPartitionStrategy::DoNothing:
      break;
    case MetisEmptyPartitionStrategy::TakeFromBiggestPartitionV1:
      _removeEmptyPartsV1(nb_part,nb_own_cell,metis_part);
      break;
    case MetisEmptyPartitionStrategy::TakeFromBiggestPartitionV2:
      _removeEmptyPartsV2(nb_part,metis_part);
      break;
    }
  }
  else
    _removeEmptyPartsV2(nb_part,metis_part);

  VariableItemInt32& cells_new_owner = mesh->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  {
    Integer index = 0;
    ENUMERATE_CELL(i_item,own_cells){
      Cell item = *i_item;
      if (!cellUsedWithConstraints(item))
        continue;

      Int32 new_owner = CheckedConvert::toInt32(metis_part[index]);
      ++index;
      changeCellOwner(item, cells_new_owner, new_owner);
    }
  }

  // libération des tableaux temporaires
  freeConstraints();

  cells_new_owner.synchronize();

  changeOwnersFromCells();

  m_nb_refine = m_parallel_mng->reduce(Parallel::ReduceMax, m_nb_refine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Comble les partitions vides (version 1).
 *
 * Cette version est la seule disponible dans les versions 2.18 (février 2020)
 * et antérieure de Arcane. Le sous-domaine qui a le plus de mailles en
 * donne une pour chaque partition vide. Cela ne fonctionne pas s'il y a plus
 * de partititions vide que de mailles dans le sous-domaine le plus remplit.
 * Pour éviter ce problème, la version 2 de l'algorithme applique itérativement
 * celui-ci.
 */
void MetisMeshPartitioner::
_removeEmptyPartsV1(const Int32 nb_part,const Int32 nb_own_cell,ArrayView<idxtype> metis_part)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();

  //The following code insures that no empty part will be created.
  // TODO: faire une meilleure solution, qui prend des elements sur la partie la plus lourde.
  UniqueArray<Int32> elem_by_part(nb_part,0);
  UniqueArray<Int32> min_part(nb_part);
  UniqueArray<Int32> max_part(nb_part);
  UniqueArray<Int32> sum_part(nb_part);
  UniqueArray<Int32> min_roc(nb_part);
  UniqueArray<Int32> max_proc(nb_part);
  for (int i =0 ; i < nb_own_cell ; i++) {
    elem_by_part[CheckedConvert::toInteger(metis_part[i])]++;
  }
  pm->computeMinMaxSum(elem_by_part, min_part, max_part, sum_part, min_roc, max_proc);

  int nb_hole=0;
  Int32 max_part_id = -1;
  Int32 max_part_nbr = -1;
  // Compute number of empty parts
  for(int i = 0; i < nb_part ; i++) {
    if (sum_part[i] == 0) {
      nb_hole ++;
    }
    if(max_part_nbr < sum_part[i]) {
      max_part_nbr = sum_part[i];
      max_part_id = i;
    }
  }
  info() << "Parmetis: number empty parts " << nb_hole;

  // Le processeur ayant la plus grosse partie de la plus
  // grosse partition comble les trous
  if(my_rank == max_proc[max_part_id]) {
    int offset = 0;
    for(int i = 0; i < nb_part ; i++) {
      // On ne comble que s'il reste des mailles
      if (sum_part[i] == 0 && offset < nb_own_cell) {
        while(offset < nb_own_cell && metis_part[offset] != max_part_id) {
          offset++;
        }
        // Si on est sorti du while car pas assez de mailles
        if(offset == nb_own_cell)
          break;
        // Le trou est comblé par ajout d'une seule maille
        metis_part[offset] = i;
        offset++;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique une itération de l'algorithme de suppression des partitions vides.
 *
 * Il s'agit du même algorithme que _removeEmptyPartsLegacy() mais en garantissant
 * qu'on ne laisse au moins une maille dans la partition qui donne ses mailles
 * aux partitions vides.
 */
Int32 MetisMeshPartitioner::
_removeEmptyPartsV2Helper(const Int32 nb_part,ArrayView<idxtype> metis_part,Int32 algo_iteration)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  const Int32 nb_own_cell = metis_part.size();
  // The following code insures that no empty part will be created.
  UniqueArray<idxtype> elem_by_part(nb_part,0);
  UniqueArray<idxtype> min_part(nb_part);
  UniqueArray<idxtype> max_part(nb_part);
  UniqueArray<idxtype> sum_part(nb_part);
  UniqueArray<Int32> min_rank(nb_part);
  UniqueArray<Int32> max_rank(nb_part);
  for (int i =0 ; i < nb_own_cell ; i++) {
    elem_by_part[metis_part[i]]++;
  }
  pm->computeMinMaxSum(elem_by_part, min_part, max_part, sum_part, min_rank, max_rank);

  // Rang des parties vides
  UniqueArray<Int32> empty_part_ranks;
  int nb_hole = 0;
  Int32 max_part_id = -1;
  Int64 max_part_nbr = -1;
  // Compute number of empty parts
  Int64 total_nb_cell = 0;
  for(int i = 0; i < nb_part ; i++) {
    Int64 current_sum_part = sum_part[i];
    total_nb_cell += current_sum_part;
    if (current_sum_part == 0) {
      empty_part_ranks.add(i);
      nb_hole ++;
    }
    if (max_part_nbr < current_sum_part) {
      max_part_nbr = current_sum_part;
      max_part_id = i;
    }
  }
  if (max_part_id<0)
    ARCANE_FATAL("Bad value max_part_id ({0})",max_part_id);
  info() << "Parmetis: check empty parts: (" << algo_iteration << ") nb_empty_parts=" << nb_hole
         << " nb_max_part=" << max_part_nbr
         << " max_part_rank=" << max_part_id
         << " max_proc_max_part_id=" << max_rank[max_part_id]
         << " empty_part_ranks=" << empty_part_ranks
         << " total_nb_cell=" << total_nb_cell;

  if (nb_hole==0)
    return 0;

  // Le processeur ayant la plus grosse partie de la plus
  // grosse partition comble les trous
  if (my_rank == max_rank[max_part_id]) {
    // On garantit qu'on n'enlèvera pas toutes nos mailles
    Int32 max_remove_cell = nb_own_cell - 1;
    int offset = 0;
    for(int i = 0; i < nb_part ; i++) {
      // On ne comble que s'il reste des mailles
      if (sum_part[i] == 0 && offset < nb_own_cell) {
        while (offset < max_remove_cell && metis_part[offset] != max_part_id) {
          offset++;
        }
        // Si on est sorti du while car pas assez de mailles
        if (offset == max_remove_cell)
          break;
        // Le trou est comblé par ajout d'une seule maille
        metis_part[offset] = i;
        offset++;
      }
    }
  }

  return nb_hole;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Comble les partitions vides (version 2).
 *
 * Cette version applique la version 1 de manière itérative pour garantir
 * qu'on ne laisse pas de partitions vide.
 */
void MetisMeshPartitioner::
_removeEmptyPartsV2(const Int32 nb_part,ArrayView<idxtype> metis_part)
{
  bool do_continue = true;
  Int32 last_nb_hole = 0;
  while (do_continue){
    Int32 nb_hole = _removeEmptyPartsV2Helper(nb_part,metis_part,0);
    info() << "Parmetis: nb_empty_partition=" << nb_hole << " last_nb_partition=" << last_nb_hole;
    if (nb_hole==0)
      break;
    // Garanti qu'on sort de la boucle si on a toujours le même nombre de trous
    // qu'avant. Cela permet d'éviter les boucles infinies.
    // Cela signifie aussi qu'on ne peut pas combler les trous et donc il y
    // aura surement des partitions vides
    if (last_nb_hole>0 && last_nb_hole<=nb_hole){
      pwarning() << "Can not remove all empty partitions. This is probably because you try"
                 << " to cut in too many partitions";
      break;
    }
    last_nb_hole = nb_hole;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * This function saves the graph in Scotch distributed file format.
 */
int MetisMeshPartitioner::
_writeGraph(IParallelMng* pm,
            Span<const idxtype> metis_vtkdist,
            Span<const idxtype> metis_xadj,
            Span<const idxtype> metis_adjncy,
            Span<const idxtype> metis_vwgt,
            Span<const idxtype> metis_ewgt,
            const String& name) const
{
  int retval = 0;

  idxtype nvtx = 0;
  idxtype nwgt = 0;
  bool have_ewgt = false;

  Int32 commrank = pm->commRank();
  Int32 commsize = pm->commSize();
  info() << "COMM_SIZE=" << commsize << " RANK=" << commrank;
  traceMng()->flush();
  //MPI_Comm_size(metis_comm, &commsize);
  //MPI_Comm_rank(metis_comm ,&commrank);
  // NOTE GG: la gestion des erreurs via METIS_ERROR ne fonctionne pas: cela produit un blocage
  // car il y a un MPI_Allreduce dans la partie sans erreur.
  // NOTE GG: Ne pas utiliser MPI directement.

  info() << "MetisVtkDist=" << metis_vtkdist;
  info() << "MetisXAdj   =" << metis_xadj;
  info() << "MetisAdjncy =" << metis_adjncy;
  info() << "MetisVWgt   =" << metis_vwgt;
  info() << "MetisEWgt   =" << metis_ewgt;

#define METIS_ERROR  ARCANE_FATAL("_writeGraph")

  StringBuilder filename(name);
  filename += "_";
  filename += commrank;
  std::ofstream file(filename.toString().localstr());

  if (metis_vtkdist.size() != commsize + 1)
    METIS_ERROR;

  nvtx = metis_vtkdist[commrank+1] - metis_vtkdist[commrank];
  if (metis_xadj.size() != nvtx + 1) {
  	std::cerr << "_writeGraph : nvtx+1 = " << nvtx << " != " << metis_xadj.size() << std::endl;
    METIS_ERROR;
  }

  if (nvtx != 0)
    nwgt = metis_vwgt.size()/nvtx;
  if (nwgt != 0 && metis_vwgt.size() % nwgt != 0)
    METIS_ERROR;

  have_ewgt = (metis_ewgt.size() != 0) ;
  if (have_ewgt && metis_ewgt.size() != metis_adjncy.size())
    METIS_ERROR;

  if (!file.is_open())
    METIS_ERROR;

  Int64 localEdges = metis_xadj[nvtx];
  Int64 globalEdges = pm->reduce(Parallel::ReduceSum,localEdges);

  file << "2" << std::endl;
  file << commsize << "\t" << commrank << std::endl;
  file << metis_vtkdist[commsize] << "\t" << globalEdges << std::endl;
  file << nvtx << "\t" << localEdges << std::endl;
  file << "0\t" << "0" << have_ewgt << nwgt << std::endl;

  /*
    Each of these lines begins with the vertex label,
    if necessary, the vertex load, if necessary, and the vertex degree, followed by the
    description of the arcs. An arc is defined by the load of the edge, if necessary, and
    by the label of its other end vertex.
  */

  //** Lbl Wgt Degree edWgt edLbl ...

  for (idxtype vertnum = 0 ; vertnum < nvtx ; vertnum++) {
    //    file << vertnum + metis_vtkdist[commrank] << " ";
    for (int dim = 0 ; dim < nwgt ; ++ dim)
      file << metis_vwgt[vertnum*nwgt+dim] << '\t';
    file << metis_xadj[vertnum + 1] - metis_xadj[vertnum];
    for (idxtype edgenum = metis_xadj[vertnum] ;
         edgenum < metis_xadj[vertnum + 1] ; ++edgenum) {
      if (have_ewgt)
        file << '\t' << metis_ewgt[edgenum];
      file << '\t' << metis_adjncy[edgenum];
    }
    file << std::endl;
  }
  file.close();

  pm->barrier();
  return retval;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MetisMeshPartitioner,
                        ServiceProperty("Metis",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

ARCANE_REGISTER_SERVICE_METISMESHPARTITIONER(Metis,MetisMeshPartitioner);

#if ARCANE_DEFAULT_PARTITIONER == METIS_DEFAULT_PARTITIONER
ARCANE_REGISTER_SERVICE(MetisMeshPartitioner,
                        ServiceProperty("DefaultPartitioner",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));
ARCANE_REGISTER_SERVICE_METISMESHPARTITIONER(DefaultPartitioner,MetisMeshPartitioner);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
