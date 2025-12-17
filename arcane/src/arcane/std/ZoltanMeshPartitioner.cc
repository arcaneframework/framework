// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ZoltanMeshPartitioner.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Partitioneur de maillage utilisant la bibliotheque Zoltan.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/StringBuilder.h"
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
#include "arcane/utils/HashTableMap.h"

#include "arcane_internal_config.h"

// Au cas ou on utilise mpich2 ou openmpi
#define MPICH_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <zoltan.h>

#include "arcane/std/MeshPartitionerBase.h"
#include "arcane/std/ZoltanMeshPartitioner_axl.h"

#include <set>

#ifdef WIN32
#include <Windows.h>
#define sleep Sleep
#endif

#define ARCANE_DEBUG_ZOLTAN

// GG: NOTE: L'implémentation actuelle ne supporte que les uniqueId() sur 32 bits.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


enum ZoltanModel
{
  ZOLTAN_HG = (1<<0),
  ZOLTAN_GRAPH = (1<<1),
  ZOLTAN_GEOM = (1<<2)
};

/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations pour le partitionnement avec Zoltan.
 */
class ZoltanInfo
  : public TraceAccessor
{
public:
  ZoltanInfo(MeshPartitionerBase* basePartitionner,ostream* ofile, int model=1, const Real edgeWeightMultiplier = 1.)
    : TraceAccessor(basePartitionner->mesh()->traceMng())
    , m_mesh_partitionner_base(basePartitionner)
    , m_nbEdges(0)
    , m_nbPins(0)
    , m_ofile(ofile)
    , m_model(model)
    , m_edgeGIDStart(0)
    , m_edge_weight_multiplier(edgeWeightMultiplier)
  {
    m_own_cells = m_mesh_partitionner_base->mesh()->ownCells();
    build();
  }
  MeshPartitionerBase* m_mesh_partitionner_base;
private:
  CellGroup m_own_cells;
  int m_nbEdges;
  int m_nbPins;
  ostream* m_ofile;
  int m_model;
  int m_edgeGIDStart;
  Real m_edge_weight_multiplier;
  std::set<std::pair<Int64, Int64> > m_weight_set;
public:
  void build()
  {
    info() << "ZoltanInfo::build()";
    if (m_model & ZOLTAN_GEOM) // No topological informations to build
      return;

    Integer nbOwnCells = m_own_cells.size();
    Integer nbOwnEdges = 0;

    if (m_model & ZOLTAN_HG) {
      nbOwnEdges += nbOwnCells;
    }
    if (m_model & ZOLTAN_GRAPH) {
      // Neighboors, each and then neighboorhood HE
      nbOwnEdges += nbOwnCells * 6;
    }

    // Compute m_edgeGIDStart = Sum_k<i nbOwnEdges
    IParallelMng* pm = m_mesh_partitionner_base->mesh()->parallelMng();
    Int32UniqueArray scanouille(pm->commSize());
    scanouille.fill(0);
    scanouille[pm->commRank()] = nbOwnEdges;
    pm->scan(Parallel::ReduceSum,scanouille);
    m_edgeGIDStart = 0;
    for (int i = 0 ; i < pm->commRank() ; ++i) {
      m_edgeGIDStart += scanouille[i];
    }

    m_nbEdges = 0;
    m_nbPins = 0;
    ENUMERATE_CELL(iCell,m_own_cells)
    {
      if (!m_mesh_partitionner_base->cellUsedWithConstraints(*iCell))
        continue;

      int nbNgb = m_mesh_partitionner_base->nbNeighbourCellsWithConstraints(*iCell);
      if (m_model & ZOLTAN_HG) {
	m_nbEdges ++;
	m_nbPins += nbNgb + 1;
      }
      if (m_model & ZOLTAN_GRAPH) {
        m_nbEdges += nbNgb;
        m_nbPins += (nbNgb*2);
      }
    }

    info() << "nbEdges=" << m_nbEdges << " ; nbPins=" << m_nbPins;

    if(m_mesh_partitionner_base->haveWeakConstraints())
    {
      MeshVariableScalarRefT<Face, Integer> weak_constraint(VariableBuildInfo(m_mesh_partitionner_base->mesh(), "EdgeWeight"));
      ENUMERATE_FACE(iface,m_mesh_partitionner_base->mesh()->ownFaces())
      {
	const Face& face = *iface;
	const Cell& bCell = iface->backCell();
	const Cell& fCell = iface->frontCell();
	if(bCell.null() || fCell.null())
	  continue;
	if(weak_constraint[face]==2)
	{
	  m_weight_set.insert(std::pair<Int64,Int64>(face.backCell().uniqueId(), face.frontCell().uniqueId()));
	  m_weight_set.insert(std::pair<Int64,Int64>(face.frontCell().uniqueId(), face.backCell().uniqueId()));
	}
      }
    }
  } // end build()


public:
  static int getHgNumVertices(void *data, int *ierr)
  {
    ARCANE_UNUSED(ierr);

    ZoltanInfo* zi = (ZoltanInfo*)data;

    /*
     * Supply this query function to Zoltan with Zoltan_Set_Num_Obj_Fn().
     * It returns the number of vertices that this process owns.
     *
     * The parallel hypergraph method requires that vertex global IDs and
     * weights are returned by the application with query functions.  The
     * global IDs should be unique, and no two processes should
     * return the same vertex.
     */
    return zi->m_mesh_partitionner_base->nbOwnCellsWithConstraints();
  }

  static void getHgVerticesAndWeights(void *data, int num_gid_entries,
                                      int num_lid_entries, ZOLTAN_ID_PTR gids, ZOLTAN_ID_PTR lids,
                                      int wgt_dim, float *obj_weights, int *ierr)
  {
    ARCANE_UNUSED(num_gid_entries);
    ARCANE_UNUSED(num_lid_entries);

    ZoltanInfo* zi = (ZoltanInfo*)data;
    /*
     * Supply this query function to Zoltan with Zoltan_Set_Obj_List_Fn().
     *
     * It supplies vertex global IDs, local IDs,
     * and weights to the Zoltan library.  The application has previously
     * indicated the number of weights per vertex by setting the
     * OBJ_WEIGHT_DIM parameter.
     */
    //int nb_cell = zi->m_own_cells.size();

    SharedArray<float> cells_weights;
    if (wgt_dim != 0) { // No weight, does not mean no limitation like for Arcane
      if (zi->m_mesh_partitionner_base->nbCellWeight() <= wgt_dim) { // We can use all criteria, so we do
	cells_weights = zi->m_mesh_partitionner_base->cellsWeightsWithConstraints(wgt_dim);
      }
      else  { // We need more criteria than available
	// So we try to balance memory !
	cells_weights = zi->m_mesh_partitionner_base->cellsSizeWithConstraints();
      }
      ArrayView<float> view_weights(cells_weights.size(), obj_weights);
      view_weights.copy(cells_weights);
    }

    int index = 0;
    ENUMERATE_CELL(icell,zi->m_own_cells)
    {
      if (!zi->m_mesh_partitionner_base->cellUsedWithConstraints(*icell))
        continue;

      gids[index] = (*icell).uniqueId().asInt32();
      lids[index] = zi->m_mesh_partitionner_base->localIdWithConstraints(*icell);

      if (zi->m_ofile) {
        for( int j=0; j< wgt_dim; ++j ){
          float weight = cells_weights[lids[index]*wgt_dim+j];
          *obj_weights++ = weight;
          // zi->info() << " Weight uid=" << gids[index] << " w=" << weight;
          (*zi->m_ofile) << " Weight uid=" << gids[index] << " w=" << weight << '\n';
        }
      }
      index ++;
    }
    *ierr = ZOLTAN_OK;
  }

  static void getHgSizeAndFormat(void *data, int *num_lists, int *num_pins, int *format, int *ierr)
  {
    ZoltanInfo* zi = (ZoltanInfo*)data;

    zi->info() << " ZoltanInfo::getHgSizeAndFormat() ";

    /*
     * Supply this query function to Zoltan with Zoltan_Set_HG_Size_CS_Fn().
     * It tells Zoltan the number of rows or columns to be supplied by
     * the process, the number of pins (non-zeroes) in the rows or columns,
     * and whether the pins are provided in compressed row format or
     * compressed column format.
     */
    *format = ZOLTAN_COMPRESSED_EDGE;
    *num_pins = zi->m_nbPins;
    *num_lists = zi->m_nbEdges;

    // if (zi->m_model == ZOLTAN_MY_HG) {
    //   *format = ZOLTAN_COMPRESSED_VERTEX;
    // }
    zi->info() << " ZoltanInfo::getHgSizeAndFormat() " << " num_list= " << *num_lists << " num_pins= " << *num_pins;
    *ierr =ZOLTAN_OK;
  }

  static void getHg(void *data,  int num_gid_entries,
                    int nrowcol, int npins, int format,
                    ZOLTAN_ID_PTR z_vtxedge_GID, int *z_vtxedge_ptr, ZOLTAN_ID_PTR z_pin_GID, int *ierr)
  {
    ZoltanInfo* zi = (ZoltanInfo*)data;

    zi->info() << " ZoltanInfo::getHg() "
               << " num_gid_entries= " << num_gid_entries
               << " nrowcol= " << nrowcol
               << " npins= " << npins
               << " format= " << format;

    // 6 neighbors max ? >> Not true for groups !
    Int64UniqueArray neighbour_cells;
    neighbour_cells.reserve(6);
    int indexPin = 0;
    int indexEdge = 0;
    int gid = zi->m_edgeGIDStart;
    z_vtxedge_ptr[0] = 0;
    ENUMERATE_CELL(i_item, zi->m_own_cells) {
      const Cell& item = *i_item;

      if (!zi->m_mesh_partitionner_base->cellUsedWithConstraints(item))
        continue;

      neighbour_cells.resize(0);
      zi->m_mesh_partitionner_base->getNeighbourCellsUidWithConstraints(item,
									neighbour_cells);

      if (zi->m_model & ZOLTAN_HG)
      {
        for( Integer z=0; z<neighbour_cells.size(); ++z )
          z_pin_GID[indexPin++] = CheckedConvert::toInteger(neighbour_cells[z]);
        z_pin_GID[indexPin++] = item.uniqueId().asInt32(); // Don't forget current cell
        z_vtxedge_GID[indexEdge] = gid++;
        // +1 because current cell is it's own neighbor !
        z_vtxedge_ptr[indexEdge+1] = z_vtxedge_ptr[indexEdge] + neighbour_cells.size() + 1;
        indexEdge ++;
      }

      if (zi->m_model & ZOLTAN_GRAPH)
      {
        // size 2 edges are face communications
        // other are neighborhood description
        for( Integer z=0; z<neighbour_cells.size(); ++z ) {
          z_pin_GID[indexPin++] = CheckedConvert::toInteger(neighbour_cells[z]);
          z_pin_GID[indexPin++] = (item.uniqueId().asInt32());
          z_vtxedge_GID[indexEdge] = gid++;
          z_vtxedge_ptr[indexEdge+1] = z_vtxedge_ptr[indexEdge] + 2;
          indexEdge++;
        }
      }
    }
    if (zi->m_ofile) {
      for (int i = 0 ; i < nrowcol ; ++i) {
        (*zi->m_ofile) << "*topo GID " << z_vtxedge_GID[i]
                       << " : "	<< z_vtxedge_ptr[i+1] - z_vtxedge_ptr[i];
        for (int j = z_vtxedge_ptr[i] ; j < z_vtxedge_ptr[i+1] ; ++j) {
          (*zi->m_ofile) << '\t' << z_pin_GID[j];
        }
        (*zi->m_ofile) << '\n';
      }
    }
    *ierr = ZOLTAN_OK;
  }


  /*
    For a list of objects, it returns the per-objects sizes (in bytes)
    of the data buffers needed to pack object data.
    void (*)(void*, int, int, int, ZOLTAN_ID_TYPE*, ZOLTAN_ID_TYPE*, int*, int*)
  */

  static void getHgVertexSizes (void *data, int num_gid_entries, int num_lid_entries,
                         int num_ids, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                         int *sizes, int *ierr)
  {
    ARCANE_UNUSED(num_gid_entries);
    ARCANE_UNUSED(num_lid_entries);
    ARCANE_UNUSED(global_ids);

    ZoltanInfo* zi = (ZoltanInfo*)data;

    SharedArray<float> cellSizes = zi->m_mesh_partitionner_base->cellsSizeWithConstraints();

    for (int i = 0 ; i < num_ids ; ++i)
    {
      sizes[i] = Convert::toInteger(cellSizes[local_ids[i]]);
    }

    *ierr = ZOLTAN_OK;
  }

  static void getHgEdgeWeightSize(void *data, int *num_edges, int *ierr)
  {
    ZoltanInfo* zi = (ZoltanInfo*)data;

    /*
     * Supply this query function to Zoltan with Zoltan_Set_HG_Size_Edge_Weights_Fn().
     * It tells Zoltan the number edges for which this process will supply
     * edge weights.  The number of weights per edge was specified with the
     * parameter EDGE_WEIGHT_DIM.  If more than one process provides a weight
     * for the same edge, the multiple weights are resolved according the
     * value for the PHG_EDGE_WEIGHT_OPERATION parameter.
     */

    *num_edges = zi->m_nbEdges;
    *ierr = ZOLTAN_OK;
  }

  static void getHgEdgeWeights(void *data,  int num_gid_entries,
                               int num_lid_entries, int nedges, int edge_weight_dim,
                               ZOLTAN_ID_PTR edge_GID, ZOLTAN_ID_PTR edge_LID, float *edge_weight, int *ierr)
  {
    ARCANE_UNUSED(num_lid_entries);

    ZoltanInfo* zi = (ZoltanInfo*)data;
    /*
     * Supply this query function to Zoltan with Zoltan_Set_HG_Edge_Weights_Fn().
     * It tells Zoltan the weights for some subset of edges.
     */

    zi->info() << " ZoltanInfo::getHgEdgeWeights() "
               << " num_gid_entries= " << num_gid_entries
               << " nedges= " << nedges
               << " edge_weight_dim= " << edge_weight_dim;

    UniqueArray<float> connectivityWeights;
    Int64UniqueArray neighbour_cells;
    neighbour_cells.reserve(6);
    connectivityWeights.reserve(6);
    int indexEdge = 0;

    ENUMERATE_CELL(i_item, zi->m_own_cells) {
      Cell item = *i_item;

      if (!zi->m_mesh_partitionner_base->cellUsedWithConstraints(item))
        continue;

      connectivityWeights.resize(0);
      neighbour_cells.resize(0);
      bool hg_model=(zi->m_model & ZOLTAN_HG);
      Real he_weight= 0;
      he_weight =
        zi->m_mesh_partitionner_base->getNeighbourCellsUidWithConstraints(item,
          neighbour_cells, &connectivityWeights, hg_model);

      if (zi->m_model & ZOLTAN_HG) {
        if (!(zi->m_model & ZOLTAN_GRAPH)) {
          // We have to sum-up de Weight of pins to define HyperEdge's one.
          for( Integer z=0; z<neighbour_cells.size(); ++z ) {
            he_weight += connectivityWeights[z];
          }
        }
        edge_GID[indexEdge] = indexEdge + zi->m_edgeGIDStart;
        edge_LID[indexEdge] = indexEdge;
        edge_weight[indexEdge++] = (float)he_weight;
      }

      if (zi->m_model & ZOLTAN_GRAPH) {
        // size 2 edges are face communications
        // other are neighborhood description
        for( Integer z=0; z<neighbour_cells.size(); ++z ){
          edge_GID[indexEdge] = indexEdge + zi->m_edgeGIDStart;
          edge_LID[indexEdge] = indexEdge;
          Real w = connectivityWeights[z];
          edge_weight[indexEdge] = (float)w;
          if (zi->m_mesh_partitionner_base->haveWeakConstraints()){
            std::pair<Int64,Int64> items(item.uniqueId(), neighbour_cells[z]);
            if(zi->m_mesh_partitionner_base->cellUsedWithWeakConstraints(items)){
              edge_weight[indexEdge] *= (float)zi->m_edge_weight_multiplier;
            }
          }
          ++indexEdge;
        }
      }
    }
    *ierr = ZOLTAN_OK;
  }

  // Return the dimension of a vertex, for geometric methods
  static int get_num_geometry(void *data, int *ierr)
  {
    ARCANE_UNUSED(data);
    *ierr = ZOLTAN_OK;
    return 3;
  }

  // Return the coordinates of my objects (vertices), for geometric methods.
  static void get_geometry_list(void *data, int sizeGID, int sizeLID,
                                int num_obj,
                                ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                int num_dim, double *geom_vec, int *ierr)
  {
    ARCANE_UNUSED(global_ids);
    ARCANE_UNUSED(local_ids);

    ZoltanInfo* zi = (ZoltanInfo*)data;

    if ( (sizeGID != 1) || (sizeLID != 1) || (num_dim > 3)){
      *ierr = ZOLTAN_FATAL;
      return;
    }

    VariableNodeReal3& coords(zi->m_mesh_partitionner_base->mesh()->nodesCoordinates());


    int i=0;
    ENUMERATE_CELL(icell,zi->m_own_cells){
      if (!zi->m_mesh_partitionner_base->cellUsedWithConstraints(*icell))
        continue;

      // on calcul un barycentre

      Real3 bar;

      for( Integer z=0, zs = (*icell).nbNode(); z<zs; ++z ){
        const Node& node = (*icell).node(z);
        bar += coords[node];
      }
      bar /= Convert::toDouble((*icell).nbNode());

      geom_vec[num_dim*i  ] = bar.x;
      geom_vec[num_dim*i+1] = bar.y;
      geom_vec[num_dim*i+2] = bar.z;
      i += 1;
    }

    String s = platform::getEnvironmentVariable("ZOLTAN_MODEL");
    if (!s.null() && (s == "MYGEOM")) {
      for (int i = 0 ; i < num_obj ; ++i) {
        geom_vec[num_dim*i+num_dim-1] = 0.0;
      }
    }

    *ierr = ZOLTAN_OK;
    return;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partitioneur de maillage utilisant la bibliotheque Zoltan.
 */
class ZoltanMeshPartitioner
: public ArcaneZoltanMeshPartitionerObject
{
 public:

  explicit ZoltanMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

 public:

  virtual void partitionMesh(bool initial_partition);
  virtual void partitionMesh(bool initial_partition,Int32 nb_part);

  virtual void notifyEndPartition();

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ZoltanMeshPartitioner::
ZoltanMeshPartitioner(const ServiceBuildInfo& sbi)
  : ArcaneZoltanMeshPartitionerObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ZoltanMeshPartitioner::
partitionMesh(bool initial_partition)
{
  Int32 nb_part = mesh()->parallelMng()->commSize();
  partitionMesh(initial_partition,nb_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ZoltanMeshPartitioner::
partitionMesh(bool initial_partition,Int32 nb_part)
{
  info() << "Load balancing with Zoltan\n";

  float ver;

  int rc = ::Zoltan_Initialize(0,0,&ver);
  Zoltan_Memory_Debug(2);
  if (rc != ZOLTAN_OK)
    fatal() << "Can not initialize zoltan (r=" << rc << ")";
  IMesh* mesh = this->mesh();
  IParallelMng* pm = mesh->parallelMng();
  Integer nb_rank = pm->commSize();

  if (nb_part<nb_rank)
    throw ArgumentException(A_FUNCINFO,"partition with nb_part<nb_rank");

  if (nb_rank==1){
    warning() << "Unable to test load balancing on a single sub-domain";
    return;
  }

  Integer nb_weight = nbCellWeight();

  struct Zoltan_Struct *zz;
  int changes;
  int numGidEntries;
  int numLidEntries;
  int numImport;
  ZOLTAN_ID_PTR importGlobalIds;
  ZOLTAN_ID_PTR importLocalIds;
  int *importProcs;
  int *importToPart;
  int numExport;
  ZOLTAN_ID_PTR exportGlobalIds;
  int *exportProcs;

  /******************************************************************
   ** Prepare to partition the example hypergraph using the Zoltan
   ** parallel hypergraph package.
   **
   ** Set parameter values, and supply to Zoltan the query functions
   ** defined in the example library which will provide to the Zoltan
   ** library the hypergraph data.
   ******************************************************************/
  zz = Zoltan_Create(*(MPI_Comm*)getCommunicator());

  Zoltan_Set_Param(zz, "RCB_REUSE", "1"); // Try to do "repartitioning"
  // option entre PHG et RCB (hypergraphe et coordonnee bissection)

  // Anciennes valeurs par defaut si pas de variable d'environnement (cf OLD_HG)
  bool usePHG = true;

  String s = platform::getEnvironmentVariable("ZOLTAN_MODEL");

  if (options()) {
    if (!s.null() && options()->model.isPresent())
      fatal() << "Conflicting configuration between ZOLTAN_MODEL environment variable and user data set";
    usePHG = options()->useHypergraphe();
    if (!usePHG)
      s = "RCB";
    else if (s.null())
      s = options()->model();
  } else if (s.null()) {
    s = "OLDHG";
  }
  // => s != null

  if (s == "HYBRID") {
    if (subDomain()->commonVariables().globalIteration() <= 2)
      s = "RCB";
    else
      s = "DUALHG";
  }

  String algo = "HYPERGRAPH";
  if (s == "MYHG" || s == "OLDHG" /* nuance avec OLD_HG ? */ || s == "DUALHG" || s == "GRAPH") {
    algo = "HYPERGRAPH";
    usePHG = true;
  } else if (s == "RIB" || s== "HSFC") {
    algo = s;
    usePHG = false;
  } else if (s == "RCB" || s == "MYGEOM") {
    algo = "RCB";
    usePHG = false;
  } else {
    fatal() << "Undefined zoltan model '" << s << "'";
  }

  int model = ZOLTAN_HG;
  if (usePHG == false) {
    model = ZOLTAN_GEOM;
  }

  if (s == "DUALHG") {
    model |= ZOLTAN_GRAPH;
  } else if (s == "GRAPH") {
    model = ZOLTAN_GRAPH;
  }

  if (usePHG) {
    nb_weight = 1; // up to 2 weights for RCB
  }

  Real edgeWeightMultiplier = 1;
  Integer repartitionFrequency = 10;
  Real imbalanceTol = 1.05;
  Real phgRepartMultiplier = 10;
  Integer phgOutputLevel = 0;
  Integer debugLevel = 0;

  if(options())
  {
    edgeWeightMultiplier = options()->edgeWeightMultiplier();
    repartitionFrequency = options()->repartFrequency();
    imbalanceTol = options()->imbalanceTol();
    phgRepartMultiplier = options()->phgRepartMultiplier();
    phgOutputLevel = options()->phgOutputLevel();
    debugLevel = options()->debugLevel();
  }

  /* General parameters */
  info() << "Zoltan: utilise un repartitionnement " << algo <<" (" << s << ").";
  Zoltan_Set_Param(zz, "LB_METHOD", algo.localstr());   /* partitioning method */
  Zoltan_Set_Param(zz, "HYPERGRAPH_PACKAGE", "PHG"); /* version of method */
  //if(!initial_partition)
  if(mesh->subDomain()->commonVariables().globalIteration()==1)
  {
    Zoltan_Set_Param(zz, "LB_APPROACH", "PARTITION"); /* version of method */
    info() << "Zoltan: Partition";
  }
  else if(mesh->subDomain()->commonVariables().globalIteration()%repartitionFrequency==0 && repartitionFrequency!=-1)
  {
    Zoltan_Set_Param(zz, "LB_APPROACH", "REPARTITION"); /* version of method */
    info() << "Zoltan: Repartition";
  }
  else
  {
    Zoltan_Set_Param(zz, "LB_APPROACH", "REFINE"); /* version of method */
    info() << "Zoltan: Refine";
  }

  String s_imbalaceTol(String::fromNumber(imbalanceTol));
  Zoltan_Set_Param(zz, "IMBALANCE_TOL", s_imbalaceTol.localstr()); /* version of method */

  String s_nb_part(String::fromNumber(nb_part));
  Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", s_nb_part.localstr());

  Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");/* global IDs are integers */
  Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");/* local IDs are integers */
  Zoltan_Set_Param(zz, "RETURN_LISTS", "EXPORT"); /* only export lists */
  String s_nb_weight(String::fromNumber(nb_weight));

  Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", s_nb_weight.localstr()); /* 2 vtx weight */

  /* Graph parameters */
  Zoltan_Set_Param(zz, "ADD_OBJ_WEIGHT", "NONE"); /* Don't calculate extra weights */

  if (usePHG) {
    /* Parallel hypergraph parameters */
    Zoltan_Set_Param(zz, "FINAL_OUTPUT", "0"); /* provide stats at the end */
    Zoltan_Set_Param(zz, "PHG_USE_TIMERS", "1");

    String s_phgOutputLevel(String::fromNumber(phgOutputLevel));
    Zoltan_Set_Param(zz, "PHG_OUTPUT_LEVEL", s_phgOutputLevel.localstr());

    // Utilisation en debug
    Zoltan_Set_Param(zz, "CHECK_HYPERGRAPH", "0");  /* see User's Guide */
    String s_debugLevel(String::fromNumber(debugLevel));
    Zoltan_Set_Param(zz, "DEBUG_LEVEL", s_debugLevel.localstr());

    // La methode par defaut (ipm) donne en theorie de meilleurs resultats
    // mais elle semble des fois bloquer avec la version 2.0 de zoltan

    Zoltan_Set_Param(zz, "PHG_COARSENING_METHOD", "IPM");
    //Zoltan_Set_Param(zz, "PHG_COARSEPARTITION_METHOD", "GREEDY");
    //Zoltan_Set_Param(zz, "PHG_CUT_OBJECTIVE", "HYPEREDGES");

    Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM", "1");
    Zoltan_Set_Param(zz, "PHG_EDGE_WEIGHT_OPERATION", "add");

    if(!initial_partition)
    {
      // Number of iterations between 2 load balances. Used to scale the cost
      // of data migration.
      String s_phgRepartMultiplier(String::fromNumber(phgRepartMultiplier));
      Zoltan_Set_Param(zz, "PHG_REPART_MULTIPLIER", s_phgRepartMultiplier.localstr());
    }

  }
  else {
    Zoltan_Set_Param(zz, "KEEP_CUTS", "0");
    Zoltan_Set_Param(zz, "RCB_OUTPUT_LEVEL", "0");
    Zoltan_Set_Param(zz, "RCB_RECTILINEAR_BLOCKS", "0");
    // The 2 following options are for multi-weights partitioning
    // Object weights are not comparable
    Zoltan_Set_Param(zz, "OBJ_WEIGHTS_COMPARABLE", "0");
    // 1 : minimize total (ie Sum over all phases)
    // 2 : between
    // 3 : Minimize worst case for each phase
    Zoltan_Set_Param(zz, "RCB_MULTICRITERIA_NORM", "3");
    Zoltan_Set_Param(zz, "RCB_MAX_ASPECT_RATIO", "10");
  }

  bool dump_infos = false;
  if (platform::getEnvironmentVariable("ARCANE_DEBUG_PARTITION")=="TRUE")
    dump_infos = true;
  ofstream ofile;
  if (dump_infos){
    StringBuilder fname;
    Integer iteration = mesh->subDomain()->commonVariables().globalIteration();
    fname = "weigth-";
    fname += pm->commRank();
    fname += "-";
    fname += iteration;
    String f(fname);
    ofile.open(f.localstr());
  }

  /* Application defined query functions (defined in exphg.c) */

  // initialisation pour la gestion des contraintes
  initConstraints();

  ostream* zofile = 0;
  if (dump_infos)
    zofile = &ofile;

  ScopedPtrT<ZoltanInfo> zoltan_info(new ZoltanInfo(this,zofile, model, edgeWeightMultiplier));
  Zoltan_Set_Num_Obj_Fn(zz,&ZoltanInfo::getHgNumVertices, zoltan_info.get());
  Zoltan_Set_Obj_List_Fn(zz, &ZoltanInfo::getHgVerticesAndWeights, zoltan_info.get());
  if (!initial_partition) {
    Zoltan_Set_Obj_Size_Multi_Fn(zz, &ZoltanInfo::getHgVertexSizes,
                                 zoltan_info.get());
  }
  if (usePHG){
    info() <<"Setting up HG Callbacks";
    Zoltan_Set_HG_Size_CS_Fn(zz, &ZoltanInfo::getHgSizeAndFormat, zoltan_info.get());
    Zoltan_Set_HG_CS_Fn(zz, &ZoltanInfo::getHg, zoltan_info.get());
    Zoltan_Set_HG_Size_Edge_Weights_Fn(zz, &ZoltanInfo::getHgEdgeWeightSize, zoltan_info.get());
    Zoltan_Set_HG_Edge_Weights_Fn(zz, &ZoltanInfo::getHgEdgeWeights, zoltan_info.get());
  }
  else
  {
    info() <<"Setting up Geom Callbacks";
    Zoltan_Set_Num_Geom_Fn(zz, ZoltanInfo::get_num_geometry, zoltan_info.get());
    Zoltan_Set_Geom_Multi_Fn(zz, ZoltanInfo::get_geometry_list, zoltan_info.get());
  }

  /* Parallel partitioning occurs now */

  int* export_partitions = 0;
  ZOLTAN_ID_PTR export_local_ids = 0;

  info() << "Doing partition";
 rc = Zoltan_LB_Partition(zz, &changes, &numGidEntries, &numLidEntries,
                          &numImport, &importGlobalIds, &importLocalIds,
                          &importProcs, &importToPart,
                          &numExport, &exportGlobalIds, &export_local_ids,
                          &exportProcs, &export_partitions);
  pm->barrier();

  VariableItemInt32& cells_new_owner = mesh->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  ENUMERATE_ITEM(icell,mesh->ownCells()){
    cells_new_owner[icell] = (*icell).owner();
  }

  invertArrayLid2LidCompacted();

  {
    CellInfoListView items_internal(m_cell_family);
    Integer nb_export = numExport;
    info() <<"numExport = "<<numExport;
    for( Integer i=0; i<nb_export; ++i ){
      Item item = items_internal[ localIdWithConstraints(export_local_ids[i]) ];
      // changement pour la maille ou tout le groupe s'il y a lieu
      changeCellOwner(item, cells_new_owner, export_partitions[i]);
      if (dump_infos) // ces infos ne tiennent pas compte des groupes/contraintes
        ofile << "EXPORT: uid=" << ItemPrinter(item) << " OLD=" << item.owner()
              << " NEW=" << cells_new_owner[item] << " PROC=" << exportProcs[i] << '\n';
    }
    //     pinfo() << "Proc " << my_rank << " nombre de mailles changeant de domaine: "
    //             << nb_export << " changes=" << changes
    //             << " ZoltanMem=" << Zoltan_Memory_Usage(ZOLTAN_MEM_STAT_MAXIMUM);
  }

  // Libere la memoire de zoltan_info
  zoltan_info = 0;
  if (dump_infos)
    ofile.close();

  if (numExport < 0) {
    sleep(3);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  Zoltan_LB_Free_Part(&exportGlobalIds, &export_local_ids,
                      &exportProcs, &export_partitions);

  // Zoltan_Destroy(&zz);

  // liberation des tableau temporaires
  freeConstraints();

  cells_new_owner.synchronize();
  changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ZoltanMeshPartitioner::
notifyEndPartition()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ZoltanMeshPartitioner,
                        ServiceProperty("Zoltan",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));
ARCANE_REGISTER_SERVICE_ZOLTANMESHPARTITIONER(Zoltan,ZoltanMeshPartitioner);

#if ARCANE_DEFAULT_PARTITIONER == ZOLTAN_DEFAULT_PARTITIONER
ARCANE_REGISTER_SERVICE(ZoltanMeshPartitioner,
                        ServiceProperty("DefaultPartitioner",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));
ARCANE_REGISTER_SERVICE_ZOLTANMESHPARTITIONER(DefaultPartitioner,ZoltanMeshPartitioner);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
